import copy
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from einops import repeat
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertLMPredictionHead,
    BertModel,
    BertOnlyNSPHead,
    BertPooler,
    BertPreTrainedModel,
)
from transformers.utils import ModelOutput

from llm.utils import max_neg_value

logger = logging.getLogger(__name__)


@dataclass
class BertForSegmentOrderingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    """loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        Total loss as the sum of the masked language modeling loss and the next sequence prediction
        (classification) loss.
    """
    mlm_loss: Optional[torch.FloatTensor] = None
    so_loss: Optional[torch.FloatTensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    so_logits: Optional[torch.FloatTensor] = None
    nsp_logits: Optional[torch.FloatTensor] = None
    nsp_loss: Optional[torch.FloatTensor] = None
    so_binary_loss: Optional[torch.FloatTensor] = None
    so_binary_logits: Optional[torch.FloatTensor] = None
    sp_binary_loss: Optional[torch.FloatTensor] = None
    sp_binary_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    """hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed 
    or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    """
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    """attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed 
    or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
    """


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, classifier_dropout: Optional[float] = None):
        super().__init__()

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.classifier(self.dropout(hidden_states))
        return logits


class SequentialClassifier(nn.Module):
    """Autoregressive decoder to take previously predicted labels into account for predicting the next label."""

    def __init__(
        self,
        num_labels: int,
        label_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        decoding_type: str,
        init_zeros: bool = False,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.label_embedding_dim = label_embedding_dim
        self.init_zeros = init_zeros
        self.hidden_dim = hidden_dim

        self.label_embeddings = nn.Embedding(num_labels, self.label_embedding_dim)
        self.drop = nn.Dropout(dropout)

        self.decoding_type = decoding_type
        if self.decoding_type == "rnn":
            self.linear = nn.Linear(self.hidden_dim, num_labels)
            self.rnn = nn.GRU(label_embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        elif self.decoding_type == "ar_linear":
            self.linear = nn.Linear(self.label_embedding_dim + self.hidden_dim, num_labels)
        else:
            raise ValueError(f"Decoding type {self.decoding_type} is not supported.")

    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # Pack input sequence, apply RNN and Unpack output
        batch_size = hidden_states.shape[0]

        if self.init_zeros:
            label_start_id = torch.zeros(batch_size, 1, device=hidden_states.device).long().fill_(0)
            label_start_embedding = torch.zeros(batch_size, 1, self.label_embedding_dim, device=hidden_states.device)
        else:
            label_start_id = torch.zeros(batch_size, 1, device=hidden_states.device).long().fill_(0)
            label_start_embedding = self.label_embeddings(label_start_id)

        if self.training and labels is not None:
            num_segments = torch.sum(labels != -100, dim=-1)
            label_ids_flat = labels[labels != -100]
            label_embeddings_flat = self.label_embeddings(label_ids_flat)
            label_embeddings = pad_sequence(torch.split(label_embeddings_flat, num_segments.tolist()), batch_first=True)

            label_embeddings_shifted = torch.cat([label_start_embedding, label_embeddings], dim=1)[:, :-1, :]
            label_ids_shifted = torch.cat([label_start_id, labels], dim=1)[:, :-1]

        else:
            label_embeddings_shifted = label_start_embedding
            label_ids_shifted = label_start_id

        if self.decoding_type == "rnn":
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=hidden_states.device)

        seqlens = hidden_states.shape[1]
        logits = []

        for i in range(seqlens):
            label_embedding = label_embeddings_shifted[:, i, :].unsqueeze(1)
            segment_embedding = hidden_states[:, i, :].unsqueeze(1)

            input_ = torch.cat([segment_embedding, label_embedding], dim=-1)
            if self.decoding_type == "rnn":
                output, hidden = self.rnn(input_, hidden)
                logit = self.linear(self.drop(output))
            elif self.decoding_type == "ar_linear":
                logit = self.linear(self.drop(input_))
            else:
                raise ValueError(f"Decoding type {self.decoding_type} is not supported.")

            probs = torch.softmax(logit, dim=-1)
            pred_label_id = torch.argmax(probs, dim=-1)

            if not self.training:
                label_embeddings_shifted = torch.cat(
                    [label_embeddings_shifted, self.label_embeddings(pred_label_id)], dim=1
                )
                label_ids_shifted = torch.cat([label_ids_shifted, pred_label_id], dim=1)

            logits.append(logit)

        logits = torch.cat(logits, dim=1)

        return logits


class BertForSequentialTextClassification(BertPreTrainedModel):
    def __init__(
        self,
        config,
        cls_method: str = "sep",
        decoding_type: str = "linear",
        label_embedding_dim: int = 32,
        init_zeros: bool = False,
        use_sep_pooler: Optional[bool] = None,
        use_sep_pos_embedding: Optional[bool] = None,
        use_pointer: Optional[bool] = None,
        loss_weights: Optional[list[float]] = None,
        use_loss_weighting: bool = False,
        mlp_hidden_dims: Optional[List] = None,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_weights = loss_weights
        self.use_loss_weighting = use_loss_weighting
        self.config = config
        self.cls_method = cls_method
        self.mlp_hidden_dims = mlp_hidden_dims

        if cls_method == "cls" and decoding_type != "linear":
            raise ValueError("Decoding type has to be 'linear' if cls method is set to 'cls'.")

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.sep_pooler = NonLinear(config) if use_sep_pooler else None

        self.use_pointer = use_pointer
        if self.use_pointer:
            self.init_success = False
            self.seg_order_head = SegmentOrderHead(config, use_cross_attn_ptr_net=True)

            self.pointer_attention = PointerAttention(hidden_size=config.hidden_size, use_cross_attn_ptr_net=True)
            self.value_layer = nn.Linear(config.hidden_size, config.hidden_size)

        if use_sep_pos_embedding:
            self.sep_pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.sep_pos_embedding = None

        if decoding_type == "linear":
            self.classifier = LinearClassifier(
                hidden_size=config.hidden_size, num_labels=self.num_labels, classifier_dropout=classifier_dropout
            )
        else:
            self.classifier = SequentialClassifier(
                hidden_dim=config.hidden_size,
                label_embedding_dim=label_embedding_dim,
                num_labels=self.num_labels,
                dropout=classifier_dropout,
                decoding_type=decoding_type,
                init_zeros=init_zeros,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def init_pointer_weights(self):
        pretrained_pointer_weights = self.seg_order_head.pointer_attention.state_dict()
        self.pointer_attention.load_state_dict(pretrained_pointer_weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        segment_sep_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # Init pointer weights if configured and not initialized before
        if self.use_pointer and not self.init_success:
            self.init_pointer_weights()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.cls_method == "sep":
            # number of segments per batch
            num_segments = segment_sep_mask.sum(dim=1)
            max_seg_len = num_segments.max()
            batch_size = sequence_output.shape[0]

            # all segment embeddings stacked
            segment_hidden_states = sequence_output[segment_sep_mask]

            # segment embeddings per batch and padded
            segment_hidden_states = pad_sequence(
                torch.split(segment_hidden_states, num_segments.tolist()), batch_first=True
            )

            if self.sep_pos_embedding is not None:
                sep_pos_ids = torch.arange(0, max_seg_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
                pos_embeddings = self.sep_pos_embedding(sep_pos_ids)
                segment_hidden_states = segment_hidden_states + pos_embeddings

            if self.sep_pooler is not None:
                segment_hidden_states = self.sep_pooler(segment_hidden_states)

            if self.use_pointer:
                # build segment input padding mask
                padding_mask = SegmentOrderHead.create_segment_padding_mask(
                    max_seg_len=max_seg_len,
                    batch_size=segment_hidden_states.shape[0],
                    hidden_states=segment_hidden_states,
                    num_segments=num_segments,
                )
                log_pointer_scores, _ = self.pointer_attention(
                    segment_hidden_states, segment_hidden_states, padding_mask
                )
                attn_scores = torch.softmax(log_pointer_scores, dim=-1)
                segment_hidden_states = self.value_layer(segment_hidden_states)
                segment_hidden_states = attn_scores @ segment_hidden_states

            # sequential text classification using the SEP token representations
            # TODO: labels might not be available during inference
            label_mask = labels != -100

            # multi-class classification
            if len(label_mask.shape) == 2:
                logits = self.classifier(segment_hidden_states, labels)
                # segment_hidden_states = segment_hidden_states.view(-1, segment_hidden_states.shape[-1])
                # labels = labels.view(-1)
            # multi-label
            else:
                segment_hidden_states = segment_hidden_states[label_mask.sum(dim=-1) > 0]
                labels = labels[label_mask].view(-1, labels.shape[-1])
                # labels needed for label embeddings in case of rnn decoding
                # TODO: rnn decoding for multi-label classification is NOT implemented
                logits = self.classifier(segment_hidden_states)

        else:
            cls_hidden_state = sequence_output[:, 0]
            logits = self.classifier(cls_hidden_state)

        loss = None
        if self.loss_weights is not None and self.use_loss_weighting:
            loss_weights = torch.tensor(self.loss_weights, device=logits.device)
        else:
            loss_weights = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=loss_weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=loss_weights)
                if labels.shape != logits.shape:
                    labels = labels.view(logits.shape)
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PointerAttention(nn.Module):
    def __init__(self, hidden_size, use_cross_attn_ptr_net: bool = False):
        super().__init__()

        self.use_cross_attn_ptr_net = use_cross_attn_ptr_net

        if use_cross_attn_ptr_net:
            self.key_size = int(hidden_size / 4)
            self.W1 = nn.Linear(hidden_size, self.key_size, bias=False)
            self.W2 = nn.Linear(hidden_size, self.key_size, bias=False)

        else:
            self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        if self.use_cross_attn_ptr_net:
            # CROSS ATTENTION POINTER NETWORK
            key = self.W1(decoder_state)
            query = self.W2(encoder_outputs)
            prod = key @ query.transpose(-1, -2) / math.sqrt(self.key_size)

        else:
            # CLASSIC POINTER NETWORK
            # (batch_size, max_seq_len, hidden_size)
            # encoder_transform = self.W1(encoder_outputs)
            # (B, Nd, Ne, D) <- (B, Ne, D)
            encoder_transform = self.W1(encoder_outputs).unsqueeze(1).expand(-1, decoder_state.shape[1], -1, -1)

            # (batch_size, 1 (unsqueezed), hidden_size)
            # decoder_transform = self.W2(decoder_state)
            # (B, Nd, 1, D) <- (B, Nd, D)
            decoder_transform = self.W2(decoder_state).unsqueeze(2)

            # 1st line of Eq.(3) in the paper
            # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
            # u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
            # (B, Nd, Ne) <- (B, Nd, Ne, D), (B, Nd, 1, D)
            prod = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts and prev predicted ids
        # log-softmax for a better numerical stability
        # (B, Nd, Ne) <- (B, Nd, Ne)
        log_score = prod
        log_score_masked = prod + mask
        # log_score = torch.nn.functional.log_softmax(prod + mask, dim=-1)

        return log_score_masked, log_score


class SegmentOrderHead(nn.Module):
    def __init__(
        self,
        config,
        mask_predicted_positions: Optional[bool] = None,
        seg_head_type: Optional[str] = "rnn+pointer",
        **pointer_params,
    ):
        super().__init__()
        self.config = config
        self.seg_head_type = seg_head_type
        if seg_head_type == "transformer+pointer":
            attn_config = copy.copy(config)
            attn_config.is_decoder = True
            attn_config.add_cross_attention = True
            self.decoder_attention = BertLayer(attn_config)
        elif seg_head_type == "rnn+pointer":
            self.decoder_rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        elif seg_head_type == "nonlinear+pointer":
            self.nonlinear = NonLinear(config)
        self.mask_predicted_positions = mask_predicted_positions
        self.pointer_attention = PointerAttention(hidden_size=config.hidden_size, **pointer_params)

    @staticmethod
    def create_segment_padding_mask(
        max_seg_len: int,
        batch_size: int,
        hidden_states: torch.Tensor,
        num_segments: torch.Tensor,
    ) -> torch.Tensor:
        range_tensor = torch.arange(max_seg_len, device=hidden_states.device, dtype=hidden_states.dtype).expand(
            batch_size, max_seg_len, max_seg_len
        )
        each_len_tensor = num_segments.view(-1, 1, 1).expand(-1, max_seg_len, max_seg_len)
        segment_mask_tensor = torch.zeros_like(range_tensor)
        mask_value = max_neg_value(segment_mask_tensor)
        segment_mask_tensor[(range_tensor >= each_len_tensor)] = mask_value
        return segment_mask_tensor

    @staticmethod
    def create_segment_mask(
        padding_mask: torch.Tensor,
        max_seg_len: int,
        hidden_states: torch.Tensor,
        ordered_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Inplace update of segment mask by setting previously predicted position indices to -inf."""
        segment_mask = padding_mask.clone()
        mask_value = max_neg_value(hidden_states)

        prev_mask = segment_mask[:, 0]
        for seg_id in range(1, max_seg_len):
            m = segment_mask[:, seg_id]
            ids = ordered_indices[:, seg_id - 1].unsqueeze(-1).expand_as(m)
            src = torch.full(
                ordered_indices.shape,
                mask_value,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            new_mask = prev_mask + m.scatter(-1, index=ids, src=src)

            # replace all inf rows with 0 to avoid nan
            all_inf_mask = new_mask.isinf().sum(dim=-1) == max_seg_len
            new_mask[all_inf_mask] = 0.0

            segment_mask[:, seg_id] = new_mask
            prev_mask = new_mask
        return segment_mask

    def transformer_pointer(
        self,
        segment_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        cls_hidden_states: torch.Tensor,
        max_seg_len: torch.Tensor,
        segment_order_labels: Optional[torch.Tensor] = None,
    ):
        mask_value = max_neg_value(segment_hidden_states)

        if self.training and segment_order_labels is not None:
            # copy segment order labels
            ordered_indices = segment_order_labels.clone()
            # replace pad id with last idx of segment_order_labels
            ordered_indices[ordered_indices == -100] = ordered_indices.shape[-1] - 1

            segment_mask = padding_mask.clone()

            # expand ordered_indices since torch.gather does not support broadcasting
            ordered_indices = ordered_indices.unsqueeze(-1).expand_as(segment_hidden_states)

            # ordered segment hidden states
            ordered_seg_hidden_states = segment_hidden_states.gather(dim=1, index=ordered_indices)

            # concat cls token embeddings at start and remove last segment embeddings
            # -> shifted ground truth segment order embedding sequence used for decoding with teacher forcing
            sos = cls_hidden_states
            dec_input_shifted = torch.cat([sos, ordered_seg_hidden_states], dim=1)[:, :-1]

            # create causal decoding mask to avoid seeing the future during training
            causal_mask = torch.full(
                (max_seg_len, max_seg_len),
                fill_value=mask_value,
                dtype=segment_hidden_states.dtype,
                device=segment_hidden_states.device,
            ).triu(1)

            # USE BERT TRANSFORMER DECODER LAYER
            padding_mask = repeat(padding_mask, "b d e -> b h d e", h=1)
            decoder_mask = causal_mask + padding_mask
            dec_hidden, past_key_value = self.decoder_attention(
                hidden_states=dec_input_shifted,
                attention_mask=decoder_mask,
                encoder_hidden_states=segment_hidden_states,
                encoder_attention_mask=padding_mask,
            )

            # pointer attention on top to get attention scores over the sequence
            log_pointer_scores, _ = self.pointer_attention(dec_hidden, segment_hidden_states, segment_mask)
            predicted_seg_order = log_pointer_scores.argmax(dim=-1)

        # greedy sequential decoding since we rely on each step's predictions
        else:
            log_pointer_scores = []
            predicted_seg_order = []
            decoder_hidden_states = []

            past_key_value = None
            decoder_input = cls_hidden_states
            decoder_padding_mask = padding_mask.clone()
            decoder_padding_mask = repeat(decoder_padding_mask, "b d e -> b h d e", h=1)

            for i in range(max_seg_len):
                # pass through decoder network
                mask_subset = padding_mask[:, : i + 1]
                decoder_mask = decoder_padding_mask[:, :, i, : i + 1].unsqueeze(2)
                encoder_attention_mask = decoder_padding_mask[:, :, i, :].unsqueeze(2)
                dec_hidden, past_key_value = self.decoder_attention(
                    hidden_states=decoder_input,
                    attention_mask=decoder_mask,
                    encoder_hidden_states=segment_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                )
                decoder_hidden_states.append(dec_hidden)
                dec_hidden_in = torch.cat(decoder_hidden_states, dim=1)

                # pass through pointer network
                log_pointer_score, _ = self.pointer_attention(dec_hidden_in, segment_hidden_states, mask_subset)
                predicted_seg = log_pointer_score.argmax(dim=-1)[:, -1]

                # append new predictions
                # log_pointer_scores_no_mask.append(log_pointer_score_no_mask[:, -1, :])
                log_pointer_scores.append(log_pointer_score[:, -1, :])
                predicted_seg_order.append(predicted_seg)

                # mask out predicted inputs
                if self.mask_predicted_positions:
                    new_seg_mask = torch.zeros(
                        (padding_mask.shape[0], padding_mask.shape[2]),
                        dtype=torch.bool,
                        device=padding_mask.device,
                    )
                    new_seg_mask = new_seg_mask.scatter(1, predicted_seg.unsqueeze(1), True)
                    new_seg_mask = repeat(new_seg_mask, "b e -> b d e", d=max_seg_len)
                    padding_mask[new_seg_mask] = mask_value

                # prepare inputs for next iteration
                next_seg_ids = repeat(predicted_seg, "b -> b d n", d=1, n=segment_hidden_states.shape[-1])
                decoder_input = torch.gather(segment_hidden_states, dim=1, index=next_seg_ids)

            log_pointer_scores = torch.stack(log_pointer_scores, dim=1)
            # log_pointer_scores_no_mask = torch.stack(log_pointer_scores_no_mask, dim=1)
            predicted_seg_order = torch.stack(predicted_seg_order, dim=1)
        return log_pointer_scores, predicted_seg_order

    def rnn_pointer(
        self,
        segment_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        max_seg_len: torch.Tensor,
    ):
        batch_size = segment_hidden_states.shape[0]
        mask_value = max_neg_value(segment_hidden_states)

        hidden = torch.zeros(1, batch_size, self.config.hidden_size, device=segment_hidden_states.device)

        log_pointer_scores = []
        predicted_seg_order = []

        for i in range(max_seg_len):
            segment_embedding = segment_hidden_states[:, i, :].unsqueeze(1)

            # encoding rnn
            output, hidden = self.decoder_rnn(segment_embedding, hidden)

            # pass through pointer network
            mask_subset = padding_mask[:, : i + 1]
            log_pointer_score, _ = self.pointer_attention(output, segment_hidden_states, mask_subset)
            predicted_seg = log_pointer_score.argmax(dim=-1)[:, -1]

            # append new predictions
            log_pointer_scores.append(log_pointer_score[:, -1, :])
            predicted_seg_order.append(predicted_seg)

            # mask out predicted inputs
            if not self.training and self.mask_predicted_positions:
                new_seg_mask = torch.zeros(
                    (padding_mask.shape[0], padding_mask.shape[2]),
                    dtype=torch.bool,
                    device=padding_mask.device,
                )
                new_seg_mask = new_seg_mask.scatter(1, predicted_seg.unsqueeze(1), True)
                new_seg_mask = repeat(new_seg_mask, "b e -> b d e", d=max_seg_len)
                padding_mask[new_seg_mask] = mask_value

        log_pointer_scores = torch.stack(log_pointer_scores, dim=1)
        predicted_seg_order = torch.stack(predicted_seg_order, dim=1)
        return log_pointer_scores, predicted_seg_order

    def nonlinear_pointer(
        self,
        segment_hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        if self.seg_head_type == "nonlinear+pointer":
            segment_hidden_states = self.nonlinear(segment_hidden_states)
        log_pointer_scores, _ = self.pointer_attention(segment_hidden_states, segment_hidden_states, padding_mask)
        predicted_seg_order = log_pointer_scores.argmax(dim=-1)[:, -1]
        return log_pointer_scores, predicted_seg_order

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_segments: torch.Tensor,
        max_seg_len: torch.Tensor,
        segment_hidden_states: torch.Tensor,
        segment_order_labels: Optional[torch.Tensor] = None,
    ):
        cls_hidden_states = hidden_states[:, 0].unsqueeze(1)

        # build segment input padding mask
        padding_mask = self.create_segment_padding_mask(
            max_seg_len=max_seg_len,
            batch_size=hidden_states.shape[0],
            hidden_states=hidden_states,
            num_segments=num_segments,
        )

        if self.seg_head_type == "transformer+pointer":
            log_pointer_scores, predicted_seg_order = self.transformer_pointer(
                segment_hidden_states=segment_hidden_states,
                padding_mask=padding_mask,
                cls_hidden_states=cls_hidden_states,
                max_seg_len=max_seg_len,
                segment_order_labels=segment_order_labels,
            )
        elif self.seg_head_type == "rnn+pointer":
            log_pointer_scores, predicted_seg_order = self.rnn_pointer(
                segment_hidden_states=segment_hidden_states,
                padding_mask=padding_mask,
                max_seg_len=max_seg_len,
            )
        elif self.seg_head_type in ["nonlinear+pointer", "pointer"]:
            log_pointer_scores, predicted_seg_order = self.nonlinear_pointer(
                segment_hidden_states=segment_hidden_states,
                padding_mask=padding_mask,
            )
        else:
            raise NotImplementedError

        return log_pointer_scores, predicted_seg_order  # , log_pointer_scores_no_mask


class NonLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForSegmentOrderPretraining(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"mlm_head.decoder.bias",
        r"mlm_head.decoder.weight",
    ]

    def __init__(
        self,
        config,
        nsp_so_loss_scaling: Optional[float] = None,
        tasks: Optional[List[str]] = None,
        pointer_params: Optional[Dict] = None,
        mlm_only_steps: Optional[int] = None,
        use_sep_pooler: Optional[bool] = None,
        use_sep_pos_embedding: Optional[bool] = None,
        seg_head_type: Optional[str] = "transformer+pointer",
    ):
        super().__init__(config)

        if pointer_params is None:
            pointer_params = {}

        if tasks is None:
            tasks = ["mlm", "so"]

        self.bert = BertModel(config, add_pooling_layer=False)

        if use_sep_pooler and "so" in tasks:
            self.sep_pooler = NonLinear(config)
        else:
            self.sep_pooler = None

        if use_sep_pos_embedding and "so" in tasks:
            self.sep_pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.sep_pos_embedding = None

        self.mlm_head = BertLMPredictionHead(config) if "mlm" in tasks else None
        self.seg_order_head = None
        if "so" in tasks:
            self.seg_order_head = SegmentOrderHead(config, seg_head_type=seg_head_type, **pointer_params)
        self.nsp_pooler = BertPooler(config) if "nsp" in tasks else None
        self.nsp_cls = BertOnlyNSPHead(config) if "nsp" in tasks else None
        self.so_correct_pooler = BertPooler(config) if "so_binary" in tasks else None
        self.so_correct_cls = BertOnlyNSPHead(config) if "so_binary" in tasks else None
        self.sp_correct_pooler = NonLinear(config) if "sp_binary" in tasks else None
        self.sp_correct_cls = BertOnlyNSPHead(config) if "sp_binary" in tasks else None

        self.nsp_so_loss_scaling_factor = nsp_so_loss_scaling if nsp_so_loss_scaling is not None else 1.0
        self.mlm_only_steps = mlm_only_steps

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        if self.mlm_head is not None:
            return self.mlm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        if self.mlm_head is not None:
            self.mlm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        so_labels: Optional[torch.Tensor] = None,
        segment_sep_mask: Optional[torch.Tensor] = None,
        nsp_labels: Optional[torch.Tensor] = None,
        so_binary_labels: Optional[torch.Tensor] = None,
        sp_binary_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], BertForSegmentOrderingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        batch_size = sequence_output.shape[0]

        # MLM Task
        mlm_loss = None
        mlm_logits = None
        if self.mlm_head is not None:
            mlm_logits = self.mlm_head(sequence_output)
            if mlm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        nsp_loss = None
        nsp_logits = None
        so_loss = None
        so_logits = None
        so_binary_loss = None
        so_binary_logits = None
        sp_binary_loss = None
        sp_binary_logits = None
        if self.mlm_only_steps is None or (global_step is not None and global_step > self.mlm_only_steps):
            # NSP Task
            if self.nsp_pooler is not None:
                nsp_logits = self.nsp_cls(self.nsp_pooler(sequence_output))
                if nsp_labels is not None:
                    loss_fct = CrossEntropyLoss()  # -100 index = padding token
                    nsp_loss = loss_fct(nsp_logits, nsp_labels)

            # ORDERING TASK
            if self.seg_order_head is not None:
                # number of segments per batch
                num_segments = segment_sep_mask.sum(dim=1)
                max_seg_len = num_segments.max()

                # all segment embeddings stacked
                segment_hidden_states = sequence_output[segment_sep_mask]

                # segment embeddings per batch and padded
                segment_hidden_states = pad_sequence(
                    torch.split(segment_hidden_states, num_segments.tolist()), batch_first=True
                )

                if self.sep_pos_embedding is not None:
                    sep_pos_ids = torch.arange(0, max_seg_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
                    pos_embeddings = self.sep_pos_embedding(sep_pos_ids)
                    segment_hidden_states = segment_hidden_states + pos_embeddings

                if self.sep_pooler is not None:
                    segment_hidden_states = self.sep_pooler(segment_hidden_states)

                so_logits, predicted_seg_order = self.seg_order_head(
                    hidden_states=sequence_output,
                    num_segments=num_segments,
                    max_seg_len=max_seg_len,
                    segment_order_labels=so_labels,
                    segment_hidden_states=segment_hidden_states,
                )

                # so_logits = log_pointer_scores_no_mask if log_pointer_scores_no_mask is not None else log_pointer_scores
                if so_labels is not None:
                    loss_fct = CrossEntropyLoss()  # -100 index = padding token
                    try:
                        so_loss = loss_fct(
                            so_logits.view(-1, so_logits.shape[-1]),
                            so_labels.view(-1),
                        )
                    except RuntimeError:
                        print()

            # ORDER CORRECT TASK
            if self.so_correct_pooler is not None:
                so_binary_logits = self.so_correct_cls(self.so_correct_pooler(sequence_output))
                if so_binary_labels is not None:
                    loss_fct = CrossEntropyLoss()  # -100 index = padding token
                    so_binary_loss = loss_fct(so_binary_logits, so_binary_labels)

            # ORDER POSITIONS CORRECT TASK
            if self.sp_correct_pooler is not None:
                # number of segments per batch
                num_segments = segment_sep_mask.sum(dim=1)

                # all segment embeddings stacked
                segment_hidden_states = sequence_output[segment_sep_mask]

                # segment embeddings per batch and padded
                segment_hidden_states = pad_sequence(
                    torch.split(segment_hidden_states, num_segments.tolist()), batch_first=True
                )

                sp_binary_logits = self.sp_correct_cls(self.sp_correct_pooler(segment_hidden_states))
                if sp_binary_labels is not None:
                    loss_fct = CrossEntropyLoss()  # -100 index = padding token
                    sp_binary_loss = loss_fct(sp_binary_logits.view(-1, 2), sp_binary_labels.view(-1))

        # Loss aggregation over tasks
        total_loss = torch.zeros(1, device=input_ids.device)
        if so_loss is not None:
            total_loss += so_loss * self.nsp_so_loss_scaling_factor
        if mlm_loss is not None:
            total_loss += mlm_loss
        if nsp_loss is not None:
            total_loss += nsp_loss * self.nsp_so_loss_scaling_factor
        if so_binary_loss is not None:
            total_loss += so_binary_loss
        if sp_binary_loss is not None:
            total_loss += sp_binary_loss

        return BertForSegmentOrderingOutput(
            loss=total_loss,  # segment_order_loss, next_sentence_loss
            mlm_loss=mlm_loss,
            mlm_logits=mlm_logits,
            so_loss=so_loss,
            so_logits=so_logits,  # log_pointer_scores, seq_relationship_scores
            nsp_loss=nsp_loss,
            nsp_logits=nsp_logits,
            so_binary_loss=so_binary_loss,
            so_binary_logits=so_binary_logits,
            sp_binary_loss=sp_binary_loss,
            sp_binary_logits=sp_binary_logits,
            hidden_states=outputs,  # outputs.hidden_states,
            attentions=outputs,  # outputs.attentions,
        )
