import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_bert import ACT2FN, BertLayerNorm


class FeedForwardBlock(torch.nn.Module):
    """A feed forward neural network of variable depth and width."""

    def __init__(self, layer_dims, **kwargs):
        # Todo: Consider having just one input argument
        super().__init__()
        self.layer_dims = layer_dims
        # If read from config the input will be string
        n_layers = len(layer_dims) - 1
        layers_all = []
        # TODO: IS this needed?
        self.output_size = layer_dims[-1]

        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = torch.nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = torch.nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits


class PredictionHead(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def logits_to_loss(self, logits, labels):
        """Implement this function in your special Prediction Head. Should
        combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param labels: labels, can vary in shape and type, depending on task
        :type labels: object
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        """Implement this function in your special Prediction Head. Should
        combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()


class RegressionHead(PredictionHead):
    def __init__(self, layer_dims=[768, 1], task_name="regression", **kwargs):
        super().__init__()
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        # num_labels is being set to 2 since it is being hijacked to store the scaling factor and the mean
        self.num_labels = 2
        self.ph_output_type = "per_sequence_continuous"
        self.model_type = "regression"
        self.loss_fct = MSELoss(reduction="none")
        self.task_name = task_name

    def forward(self, x):
        logits = self.feed_forward(x)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        return self.loss_fct(logits, labels.float())

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.cpu().numpy()
        return preds


class TextClassificationHead(PredictionHead):
    def __init__(
        self,
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        task_name="text_classification",
        **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_ignore_index:
        :param loss_reduction:
        :param task_name:
        :param kwargs:
        """
        super().__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        # if layer_dims:
        #     self.layer_dims = layer_dims
        #     # logger.warning("`layer_dims` will be deprecated in future releases")
        # elif num_labels:
        #     self.layer_dims = [768, num_labels]
        # else:
        #     raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.layer_dims = layer_dims
        self.num_labels = self.layer_dims[-1]
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        # logger.info(f"Prediction head initialized with size {self.layer_dims}")

        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = (
            task_name  # used for connecting with the right output of the processor
        )
        self.class_weights = class_weights

        if class_weights:
            # logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = torch.nn.Parameter(
                torch.tensor(class_weights), requires_grad=False
            )
        else:
            balanced_weights = None

        self.loss_fct = CrossEntropyLoss(
            weight=balanced_weights,
            reduction=loss_reduction,
            ignore_index=loss_ignore_index,
        )

        # self.generate_config()

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        return self.loss_fct(logits, labels.view(-1))

    def logits_to_probs(self, logits, return_probs: bool = True, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_probs:
            probs = probs
        else:
            probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [self.label_list[int(x)] for x in pred_ids]
        return preds


class MultiLabelTextClassificationHead(PredictionHead):
    def __init__(
        self,
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_reduction="none",
        task_name="text_classification",
        pred_threshold=0.5,
        **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_reduction:
        :param task_name:
        :param pred_threshold:
        :param kwargs:
        """
        super().__init__()
        # # num_labels could in most cases also be automatically retrieved from the data processor
        # if layer_dims:
        #     self.layer_dims = layer_dims
        #     logger.warning("`layer_dims` will be deprecated in future releases")
        # elif num_labels:
        #     self.layer_dims = [768, num_labels]
        # else:
        #     raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        # logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.ph_output_type = "per_sequence"
        self.model_type = "multilabel_text_classification"
        self.task_name = (
            task_name  # used for connecting with the right output of the processor
        )
        self.class_weights = class_weights
        self.pred_threshold = pred_threshold

        if class_weights:
            # logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            # TODO must balanced weight really be a instance attribute?
            self.balanced_weights = torch.nn.Parameter(
                torch.tensor(class_weights), requires_grad=False
            )
        else:
            self.balanced_weights = None

        self.loss_fct = BCEWithLogitsLoss(
            pos_weight=self.balanced_weights, reduction=loss_reduction
        )

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        # label_ids = kwargs.get(self.label_tensor_name).to(dtype=torch.float)
        loss = self.loss_fct(
            logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
        )
        per_sample_loss = loss.mean(1)
        return per_sample_loss

    def logits_to_probs(self, logits, **kwargs):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        probs = self.logits_to_probs(logits)
        # TODO we could potentially move this to GPU to speed it up
        pred_ids = [np.where(row > self.pred_threshold)[0] for row in probs]
        preds = []
        for row in pred_ids:
            preds.append([self.label_list[int(x)] for x in row])
        return preds


class TokenClassificationHead(PredictionHead):
    def __init__(self, layer_dims=None, num_labels=None, task_name="ner", **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param task_name:
        :param kwargs:
        """
        super().__init__()
        # if layer_dims:
        #     self.layer_dims = layer_dims
        #     logger.warning("`layer_dims` will be deprecated in future releases")
        # elif num_labels:
        #     self.layer_dims = [768, num_labels]
        # else:
        #     raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        # logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.ph_output_type = "per_token"
        self.model_type = "token_classification"
        self.task_name = task_name

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, labels, initial_mask, padding_mask=None, **kwargs):

        # Todo: should we be applying initial mask here? Loss is currently calculated even on non initial tokens
        active_loss = padding_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss = self.loss_fct(
            active_logits, active_labels
        )  # loss is a 1 dimemnsional (active) token loss
        return loss

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret

    def logits_to_preds(self, logits, initial_mask, **kwargs):
        preds_word_all = []
        preds_tokens = torch.argmax(logits, dim=2)
        preds_token = preds_tokens.detach().cpu().numpy()
        # used to be: padding_mask = padding_mask.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()

        for idx, im in enumerate(initial_mask):
            preds_t = preds_token[idx]
            # Get labels and predictions for just the word initial tokens
            preds_word_id = self.initial_token_only(preds_t, initial_mask=im)
            preds_word = [self.label_list[pwi] for pwi in preds_word_id]
            preds_word_all.append(preds_word)
        return preds_word_all

    def logits_to_probs(self, logits, initial_mask, return_class_probs, **kwargs):
        # get per token probs
        softmax = torch.nn.Softmax(dim=2)
        token_probs = softmax(logits)
        if return_class_probs:
            token_probs = token_probs
        else:
            token_probs = torch.max(token_probs, dim=2)[0]
        token_probs = token_probs.cpu().numpy()

        # convert to per word probs
        all_probs = []
        initial_mask = initial_mask.detach().cpu().numpy()
        for idx, im in enumerate(initial_mask):
            probs_t = token_probs[idx]
            probs_words = self.initial_token_only(probs_t, initial_mask=im)
            all_probs.append(probs_words)
        return all_probs


class BertLMHead(PredictionHead):
    def __init__(
        self, hidden_size, vocab_size, hidden_act="gelu", task_name="lm", **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.num_labels = vocab_size  # vocab size
        # TODO Check if weight init needed!
        # self.apply(self.init_bert_weights)
        self.ph_output_type = "per_token"

        self.model_type = "language_modelling"
        self.task_name = task_name

        # NN Layers
        # this is the "transform" module in the pytorch-transformers repo
        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.transform_act_fn = ACT2FN[self.hidden_act]
        self.LayerNorm = BertLayerNorm(self.hidden_size, eps=1e-12)

        # this is the "decoder" in the pytorch-transformers repo
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def set_shared_weights(self, shared_embedding_weights):
        self.decoder.weight = shared_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        lm_logits = self.decoder(hidden_states) + self.bias
        return lm_logits

    def logits_to_loss(self, logits, labels, **kwargs):
        batch_size = labels.shape[0]
        masked_lm_loss = self.loss_fct(
            logits.view(-1, self.num_labels), labels.view(-1)
        )
        per_sample_loss = masked_lm_loss.view(-1, batch_size).mean(dim=0)
        return per_sample_loss

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        lm_label_ids = kwargs.get(self.label_tensor_name).cpu().numpy()
        lm_preds_ids = logits.argmax(2)
        # apply mask to get rid of predictions for non-masked tokens
        assert lm_preds_ids.shape == lm_label_ids.shape
        lm_preds_ids[lm_label_ids == -1] = -1
        lm_preds_ids = lm_preds_ids.tolist()
        preds = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for pred_ids_for_sequence in lm_preds_ids:
            preds.append(
                [self.label_list[int(x)] for x in pred_ids_for_sequence if int(x) != -1]
            )
        return preds
