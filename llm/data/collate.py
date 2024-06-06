import logging
from collections import defaultdict
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

from llm.data.collate_utils import (
    create_input_ids,
    create_token_type_ids,
    mask_whole_words,
    shuffle_segments,
)

logger = logging.getLogger(__name__)


class LMCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        device: str,
        segment_ordering: bool = True,
        mlm_probability: Optional[float] = None,
        mlm_apply_80_10_10: Optional[bool] = None,
        word_prefix: Optional[str] = None,
        subword_prefix: Optional[str] = None,
        next_sentence_prediction: bool = False,
        segment_order_correct_prediction: bool = False,
        segment_position_correct_prediction: bool = False,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.mlm_probability = mlm_probability
        self.mlm_apply_80_10_10 = mlm_apply_80_10_10
        self.word_prefix = word_prefix
        self.subword_prefix = subword_prefix
        self.segment_ordering = segment_ordering
        self.next_sentence_prediction = next_sentence_prediction
        self.segment_order_correct_prediction = segment_order_correct_prediction
        self.segment_position_correct_prediction = segment_position_correct_prediction

        self.tasks = []
        if self.mlm_probability:
            self.tasks.append("mlm")
        if self.segment_ordering:
            self.tasks.append("so")
        if self.next_sentence_prediction:
            self.tasks.append("nsp")
        if self.segment_order_correct_prediction:
            self.tasks.append("so_binary")
        if self.segment_position_correct_prediction:
            self.tasks.append("sp_binary")

        # set shuffling logic to prepare input ids and labels
        self.shuffle = None
        if self.segment_position_correct_prediction:
            self.shuffle = "half_half"
        elif self.segment_order_correct_prediction:
            self.shuffle = "half"
        elif self.segment_ordering or self.next_sentence_prediction:
            self.shuffle = "all"

        logger.info(f"Pretraining: {', '.join(self.tasks)}")

    def __call__(self, samples: tuple[tuple[dict], Optional[list[list[int]]]]) -> dict[str, list | torch.Tensor]:
        samples, memory_clearing_mask = samples

        batch = create_input_ids(samples, self.tokenizer, shuffle=self.shuffle)
        so_labels = batch.pop("so_labels", None)

        # convert and pad intput ids to torch tensor
        input_ids = [torch.tensor(sample, device=self.device) for sample in batch["input_ids"]]
        batch["input_ids"] = pad_sequence(input_ids, batch_first=True)

        if self.shuffle:
            segment_order_labels = [torch.tensor(sample, device=self.device) for sample in so_labels]
            segment_order_labels = pad_sequence(segment_order_labels, batch_first=True, padding_value=-100)
            segment_sep_mask = batch["input_ids"] == self.tokenizer.sep_token_id

            # segment ordering task
            if self.segment_ordering:
                batch["so_labels"] = segment_order_labels
                batch["segment_sep_mask"] = segment_sep_mask
            # next sentence prediction task
            if self.next_sentence_prediction:
                assert segment_order_labels.shape[-1] == 2
                batch["nsp_labels"] = segment_order_labels.argmax(dim=-1)
                batch["token_type_ids"] = create_token_type_ids(
                    input_ids=batch["input_ids"], sep_token_id=self.tokenizer.sep_token_id, device=self.device
                )
            if self.segment_order_correct_prediction or self.segment_position_correct_prediction:
                batch_size, num_segments = segment_order_labels.shape
                pad_mask = segment_order_labels != -100
                correct = torch.arange(0, num_segments, device=self.device).unsqueeze(0).repeat(batch_size, 1)
                shuffled = torch.clone(correct)

                shuffled[pad_mask] = segment_order_labels[pad_mask]
                so_position_correct_labels = (correct == shuffled).long()
                so_correct_labels = torch.all(so_position_correct_labels, dim=1).long()
                so_position_correct_labels[~pad_mask] = -100
                batch["so_binary_labels"] = so_correct_labels
                batch["sp_binary_labels"] = so_position_correct_labels
                batch["segment_sep_mask"] = segment_sep_mask

        # whole word masking for mlm
        if self.mlm_probability:
            batch["input_ids"], batch["mlm_labels"] = mask_whole_words(
                input_ids=batch["input_ids"],
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
                mlm_apply_80_10_10=self.mlm_apply_80_10_10,
                word_prefix=self.word_prefix,
                subword_prefix=self.subword_prefix,
                device=self.device,
            )

        batch["attention_mask"] = batch["input_ids"] != 0
        return batch


class SequentialTextCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        device: str,
        cls_method: str = "sep",
        apply_segment_shuffling: bool = False,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.cls_method = cls_method
        self.apply_segment_shuffling = apply_segment_shuffling
        self.label_map: dict[str, int] | None = None

    def __call__(self, samples: tuple[tuple[dict], Optional[list[list[int]]]]) -> dict[str, list | torch.Tensor]:
        samples, memory_clearing_mask = samples

        batch = defaultdict(list)
        for sample in samples:
            # TODO: This does not work if batches are no longer independent (e.g. using transformer-xl recurrence)
            if sample is None:
                continue

            input_ids = [self.tokenizer.cls_token_id]
            # Set to true only for cls method (1 segment per sample)
            if self.apply_segment_shuffling:
                sample = shuffle_segments(sample, keys_to_shuffle=["input_ids_per_segment", "labels"])
            for segment_ids in sample["input_ids_per_segment"]:
                input_ids.extend(segment_ids + [self.tokenizer.sep_token_id])

            batch["input_ids"].append(input_ids)

            batch["doc_ids"].append(sample.get("doc_id"))

            labels_formatted = []
            for label in sample["labels"]:
                if isinstance(label, list):  # multi-label classification
                    # create multi-hot label representation
                    labels_formatted.append([1.0 if req in label else 0.0 for req in self.label_map])
                else:
                    # get label id for multi-class prediction
                    labels_formatted.append(self.label_map[label])
            batch["labels"].append(labels_formatted)

        # convert and pad intput ids to torch tensor
        input_ids = [torch.tensor(sample, device=self.device) for sample in batch["input_ids"]]
        batch["input_ids"] = pad_sequence(input_ids, batch_first=True)

        # convert and pad labels to torch tensor
        labels = [torch.tensor(sample, device=self.device) for sample in batch["labels"]]
        batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100.0)

        if self.cls_method == "sep":
            batch["segment_sep_mask"] = batch["input_ids"] == self.tokenizer.sep_token_id
        elif self.cls_method == "cls":
            batch["token_type_ids"] = create_token_type_ids(
                input_ids=batch["input_ids"], sep_token_id=self.tokenizer.sep_token_id, device=self.device
            )
            batch["labels"] = batch["labels"].view(-1)
        else:
            raise ValueError(f"Classification method mus be either 'cls' or 'sep'. However, it is '{self.cls_method}'.")

        batch["attention_mask"] = batch["input_ids"] != 0

        return batch


COLLATORS = {
    "language_modeling": LMCollator,
    "sequential_text_clf": SequentialTextCollator,
}
