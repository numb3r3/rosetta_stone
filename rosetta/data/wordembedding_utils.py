import numpy as np
from tqdm import tqdm

from .. import helper


logger = logger = helper.get_logger(__name__)


def load_embeddings(embedding_file: str, vocab: list):
    words = set()
    hidden_size = None
    vectors = {}

    with open(embedding_file, "r") as f:
        for line in tqdm(f, desc="loading embeddings"):
            line = line.strip()
            if line:
                word, vec = line.split(" ", 1)
                # omit repetitions = speed up + debug
                if word in words:
                    continue

                try:
                    np_vec = np.fromstring(vec, sep=" ")
                    if hidden_size is None:
                        # word2vec includes number of vectors and its dimension as header
                        if len(np_vec) < 4:
                            logger.info("Skipping header")
                            continue
                        else:
                            hidden_size = len(np_vec)
                    assert len(np_vec) == hidden_size
                    vectors[word] = np_vec
                    words.add(word)
                except Exception as ex:
                    if logger is not None:
                        logger.debug(
                            "Embeddings reader: Could not convert line: {}".format(line)
                        )
                    else:
                        raise ex
    # reduce memory usage
    words.clear()

    embeddings = np.zeros((len(vocab), hidden_size))
    for i, w in enumerate(vocab):
        current = vectors.get(w, np.zeros(hidden_size))
        if w not in vectors:
            logger.warning(f"Could not load pretrained embedding for word: {w}")
        embeddings[i, :] = current
    return embeddings


def load_word2vec_vocab(vocab_filename: str):
    """Loads a vocabulary file into a list."""
    vocab = []
    with open(vocab_filename, "r") as reader:
        for l in reader:
            w, c = l.strip().split(" ")
            vocab.append(w.strip())
    return vocab


def s3e_pooling(
    token_embs,
    token_ids,
    token_weights,
    centroids,
    token_to_cluster,
    mask,
    svd_components=None,
):
    """Pooling of word/token embeddings as described by Wang et al in their
    paper "Efficient Sentence Embedding via Semantic Subspace Analysis"
    (https://arxiv.org/abs/2002.09620) Adjusted their implementation from here:
    https://github.com/BinWang28/Sentence-Embedding-S3E.

    This method takes a fitted "s3e model" and token embeddings from a language model and returns sentence embeddings
    using the S3E Method. The model can be fitted via `fit_s3e_on_corpus()`.

    Usage: See `examples/embeddings_extraction_s3e_pooling.py`

    :param token_embs: numpy array of shape (batch_size, max_seq_len, emb_dim) containing the embeddings for each token
    :param token_ids: numpy array of shape (batch_size, max_seq_len) containing the ids for each token in the vocab
    :param token_weights: dict with key=token_id, value= weight in corpus
    :param centroids: numpy array of shape (n_cluster, emb_dim) that describes the centroids of our clusters in the embedding space
    :param token_to_cluster: numpy array of shape (vocab_size, 1) where token_to_cluster[i] = cluster_id that token with id i belongs to
    :param svd_components: Components from a truncated singular value decomposition (SVD, aka LSA) to be
                           removed from the final sentence embeddings in a postprocessing step.
                           SVD must be fit on representative sample of sentence embeddings first and can
                           then be removed from all subsequent embeddings in this function.
                           We expect the sklearn.decomposition.TruncatedSVD.fit(<your_embeddings>)._components to be passed here.
    :return: embeddings matrix of shape (batch_size, emb_dim + (n_clusters*n_clusters+1)/2)
    """

    embeddings = []
    n_clusters = centroids.shape[0]
    emb_dim = token_embs.shape[2]
    # n_tokens = token_embs.shape[1]
    n_samples = token_embs.shape[0]
    # Mask tokens that should be ignored (e.g. Padding, CLS ...)
    token_ids[mask] = -1

    # Process each sentence in the batch at a time
    for sample_idx in range(n_samples):
        stage_vec = [{}]
        # 1) create a dict with key=tok_id, value = embedding
        for tok_idx, tok_id in enumerate(token_ids[sample_idx, :]):
            if tok_id != -1:
                stage_vec[-1][tok_id] = token_embs[sample_idx, tok_idx]

        # 2) create a second dict with key=cluster_id, val=[embeddings] (= C)
        stage_vec.append({})
        for k, v in stage_vec[-2].items():
            cluster = token_to_cluster[k]

            if cluster in stage_vec[-1]:
                stage_vec[-1][cluster].append(stage_vec[-2][k] * token_weights[k])
            else:
                stage_vec[-1][cluster] = []
                stage_vec[-1][cluster].append(stage_vec[-2][k] * token_weights[k])

        # VLAD for each cluster
        for k, v in stage_vec[-1].items():
            # Centroids
            centroid_vec = centroids[k]

            # Residual
            v = [wv - centroid_vec for wv in v]
            stage_vec[-1][k] = np.sum(v, 0)

        # Compute Sentence Embedding (weighted avg, dim = original embedding dim)
        sentvec = []
        vec = np.zeros((emb_dim))
        for key, value in stage_vec[0].items():
            # print(token_weights[key])
            vec = vec + value * token_weights[key]
        sentvec.append(vec / len(stage_vec[0].keys()))

        # Covariance Descriptor (dim = k*(k+1)/2, with k=n_clusters)
        matrix = np.zeros((n_clusters, emb_dim))
        for j in range(n_clusters):
            if j in stage_vec[-1]:
                matrix[j, :] = stage_vec[-1][j]
        matrix_no_mean = matrix - matrix.mean(1)[:, np.newaxis]
        cov = matrix_no_mean.dot(matrix_no_mean.T)

        # Generate Embedding
        iu1 = np.triu_indices(cov.shape[0])
        iu2 = np.triu_indices(cov.shape[0], 1)
        cov[iu2] = cov[iu2] * np.sqrt(2)
        vec = cov[iu1]

        vec = vec / np.linalg.norm(vec)

        sentvec.append(vec)

        # Concatenate weighted avg + covariance descriptors
        sentvec = np.concatenate(sentvec)

        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)

    # Post processing (removal of first principal component)
    if svd_components is not None:
        embeddings = (
            embeddings - embeddings.dot(svd_components.transpose()) * svd_components
        )
    return embeddings
