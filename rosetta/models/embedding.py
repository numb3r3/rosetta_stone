from collections import OrderedDict
import json
import os
from typing import Dict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers.tokenization_bert import load_vocab

from .. import helper
from ..data import wordembedding_utils
from ..data.tokenization import SPECIAL_TOKENS


class EmbeddingModel:
    """
    Embedding Model that combines
    - Embeddings
    - Config Object
    - Vocab
    Necessary to work with Bert and other LM style functionality
    """

    def __init__(self, embedding_file: str, vocab_file: str, config_dict: Dict):
        """
        :param embedding_file: filename of embeddings. Usually in txt format, with the word and associated vector on each line
        :type embedding_file: str
        :param config_dict: dictionary containing config elements
        :type config_dict: dict
        :param vocab_file: filename of vocab, each line contains a word
        :type vocab_file: str
        """
        self.config = config_dict
        self.vocab = load_vocab(vocab_file)

        temp = wordembedding_utils.load_embeddings(
            embedding_file=embedding_file, vocab=self.vocab
        )
        self.embeddings = torch.from_numpy(temp).float()
        assert "[UNK]" in self.vocab, "No [UNK] symbol in Wordembeddingmodel! Aborting"
        self.unk_idx = self.vocab["[UNK]"]

    def save(self, save_dir):
        # Save Weights
        save_name = os.path.join(save_dir, self.config["embedding_filename"])
        embeddings = self.embeddings.cpu().numpy()
        with open(save_name, "w") as f:
            for w, vec in tqdm(
                zip(self.vocab, embeddings),
                desc="saving embeddings",
                total=embeddings.shape[0],
            ):
                f.write(w + " " + " ".join(["%.6f" % v for v in vec]) + "\n")
        f.close()

        # Save vocab
        save_name = os.path.join(save_dir, self.config["vocab_filename"])
        with open(save_name, "w") as f:
            for w in self.vocab:
                f.write(w + "\n")
        f.close()


class WordEmbedding_LM(nn.Module):
    """
    A Language Model based only on word embeddings
    - Inside FARM, WordEmbedding Language Models must have a fixed vocabulary
    - Each (known) word in some text input is projected to its vector representation
    - Pooling operations can be applied for representing whole text sequences
    """

    def __init__(self, logger=None):
        super().__init__()
        if logger is None:
            logger = helper.get_logger(__name__)
        self.logger = logger

        self.name = "WordEmbedding_LM"
        self.model = None
        self.pooler = None

    @classmethod
    def load(cls, model_path, language=None, **kwargs):
        """
        Load a language model either by supplying
        * a local path of a model trained via FARM ("some_dir/farm_model")
        * the name of a remote model on s3
        :param model_path: path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model
        """
        embedding_lm_model = cls()
        embedding_lm_model.name = kwargs.get("model_name", "embedding_lm_model")

        config_file = os.path.join(model_path, "config.json")

        if os.path.exists(config_file):
            config = json.load(open(config_file, "r"))
            embedding_file = os.path.join(model_path, config["embedding_filename"])
            vocab_filename = os.path.join(model_path, config["vocab_filename"])
            embedding_lm_model.model = EmbeddingModel(
                embedding_file=embedding_file,
                vocab_file=vocab_filename,
                config_dict=config,
            )
            embedding_lm_model.language = config.get("language", None)
        else:
            raise AttributeError(f"The config file {config_file} doesn't exist!")

        # taking the mean for getting the pooled representation
        # TODO: extend this to other pooling operations or remove
        embedding_lm_model.pooler = lambda x: torch.mean(x, dim=0)
        return embedding_lm_model

    def save(self, save_dir):
        """
        Save the model embeddings and its config file so that it can be loaded again.
        # TODO make embeddings trainable and save trained embeddings
        # TODO save model weights as pytorch model bin for more efficient loading and saving
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        # save model
        self.model.save(save_dir=save_dir)
        # save config
        self.save_config(save_dir=save_dir)

    def save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "config.json")
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)

    def forward(self, input_ids, batch_norm=False, **kwargs):
        """
        Perform the forward pass of the wordembedding model.
        This is just the mapping of words to their corresponding embeddings
        """
        sequence_output = []
        pooled_output = []
        # TODO do not use padding items in pooled output
        for sample in input_ids:
            sample_embeddings = []
            for index in sample:
                # if index != self.model.unk_idx:
                sample_embeddings.append(self.model.embeddings[index])
            sample_embeddings = torch.stack(sample_embeddings)
            sequence_output.append(sample_embeddings)
            pooled_output.append(self.pooler(sample_embeddings))

        sequence_output = torch.stack(sequence_output)
        pooled_output = torch.stack(pooled_output)

        # batchnorm for stable learning
        if batch_norm:
            m = nn.BatchNorm1d(pooled_output.shape[1])
            pooled_output = m(pooled_output)

        return sequence_output, pooled_output

    def trim_vocab(self, token_counts, processor, min_threshold):
        """ Remove embeddings for rare tokens in your corpus (< `min_threshold` occurrences) to reduce model size"""
        self.logger.info(
            f"Removing tokens with less than {min_threshold} occurrences from model vocab"
        )
        new_vocab = OrderedDict()
        valid_tok_indices = []
        cnt = 0
        old_num_emb = self.model.embeddings.shape[0]
        for token, tok_idx in self.model.vocab.items():
            if token_counts.get(token, 0) >= min_threshold or token in SPECIAL_TOKENS:
                new_vocab[token] = cnt
                valid_tok_indices.append(tok_idx)
                cnt += 1

        self.model.vocab = new_vocab
        self.model.embeddings = self.model.embeddings[valid_tok_indices, :]

        # update tokenizer vocab in place
        processor.tokenizer.vocab = self.model.vocab
        processor.tokenizer.ids_to_tokens = OrderedDict()
        for k, v in processor.tokenizer.vocab.items():
            processor.tokenizer.ids_to_tokens[v] = k

        self.logger.info(
            f"Reduced vocab from {old_num_emb} to {self.model.embeddings.shape[0]}"
        )

    def normalize_embeddings(
        self,
        zero_mean=True,
        pca_removal=False,
        pca_n_components=300,
        pca_n_top_components=10,
        use_mean_vec_for_special_tokens=True,
        n_special_tokens=5,
    ):
        """ Normalize word embeddings as in https://arxiv.org/pdf/1808.06305.pdf
            (e.g. used for S3E Pooling of sentence embeddings)

        :param zero_mean: Whether to center embeddings via subtracting mean
        :type zero_mean: bool
        :param pca_removal: Whether to remove PCA components
        :type pca_removal: bool
        :param pca_n_components: Number of PCA components to use for fitting
        :type pca_n_components: int
        :param pca_n_top_components: Number of PCA components to remove
        :type pca_n_top_components: int
        :param use_mean_vec_for_special_tokens: Whether to replace embedding of special tokens with the mean embedding
        :type use_mean_vec_for_special_tokens: bool
        :param n_special_tokens: Number of special tokens like CLS, UNK etc. (used if `use_mean_vec_for_special_tokens`). 
                                 Note: We expect the special tokens to be the first `n_special_tokens` entries of the vocab.
        :type n_special_tokens: int
        :return: None
        """

        if zero_mean:
            self.logger.info("Removing mean from embeddings")
            # self.model.embeddings[:n_special_tokens, :] = torch.zeros((n_special_tokens, 300))
            mean_vec = torch.mean(self.model.embeddings, 0)
            self.model.embeddings = self.model.embeddings - mean_vec

            if use_mean_vec_for_special_tokens:
                self.model.embeddings[:n_special_tokens, :] = mean_vec

        if pca_removal:
            from sklearn.decomposition import PCA

            self.logger.info(
                "Removing projections on top PCA components from embeddings (see https://arxiv.org/pdf/1808.06305.pdf)"
            )
            pca = PCA(n_components=pca_n_components)
            pca.fit(self.model.embeddings.cpu().numpy())

            U1 = pca.components_
            explained_variance = pca.explained_variance_

            # Removing projections on top components
            PVN_dims = pca_n_top_components
            for emb_idx in tqdm(
                range(self.model.embeddings.shape[0]), desc="removing projections"
            ):
                for pca_idx, u in enumerate(U1[0:PVN_dims]):
                    ratio = (
                        explained_variance[pca_idx] - explained_variance[PVN_dims]
                    ) / explained_variance[pca_idx]
                    self.model.embeddings[emb_idx] = (
                        self.model.embeddings[emb_idx]
                        - ratio
                        * np.dot(u.transpose(), self.model.embeddings[emb_idx])
                        * u
                    )
