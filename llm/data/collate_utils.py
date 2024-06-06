import logging
import math
import random
from collections import defaultdict
from typing import Any, List, Optional

import torch
from transformers import BertTokenizerFast, PreTrainedTokenizerFast

from llm.utils import change_order

logger = logging.getLogger(__name__)


def flatten_and_shuffle_input_ids(
    sample_input_ids: list[list[int]],
    segment_order_labels: list[int],
    sep_token_id: int,
    bos_token_id: int,
) -> list[int]:
    shuffled_input_ids = [bos_token_id]
    for i in segment_order_labels:
        shuffled_input_ids.extend(sample_input_ids[i] + [sep_token_id])
    return shuffled_input_ids


def get_segment_order_labels(num_segments: int) -> List[int]:
    # change segment order for 50% of samples
    change_segment_order = random.randint(0, 1)
    if change_segment_order:
        # shuffle 50% of segments in sample
        num_segments_to_shuffle = math.ceil(num_segments / 2)
        segments_to_shuffle = sorted(random.sample(range(num_segments), k=num_segments_to_shuffle))
        change_order(segments_to_shuffle)

        segment_order_labels = []
        idx = 0
        for i in range(num_segments):
            if i in segments_to_shuffle:
                segment_order_labels.append(segments_to_shuffle[idx])
                idx += 1
            else:
                segment_order_labels.append(i)
    else:
        segment_order_labels = list(range(num_segments))

    return segment_order_labels


def create_input_ids(
    samples: tuple[dict], tokenizer: BertTokenizerFast, shuffle: Optional[str] = None
) -> dict[str, Any]:
    batch = defaultdict(list)
    for sample in samples:
        input_ids_per_segment = sample["input_ids_per_segment"]
        num_segments = len(input_ids_per_segment)

        if shuffle is not None:
            if shuffle == "all":
                # sample random order of article segments in sample
                segment_order_labels = random.sample(range(num_segments), k=num_segments)
            elif shuffle == "half":
                change_segment_order = random.randint(0, 1)
                if change_segment_order:
                    segment_order_labels = random.sample(range(num_segments), k=num_segments)
                else:
                    segment_order_labels = list(range(num_segments))
            elif shuffle == "half_half":
                # see logic in get_segment_order_labels fn
                segment_order_labels = get_segment_order_labels(num_segments)
            else:
                raise ValueError(f"'shuffle' argument has to be 'all' or 'half'. It is {shuffle}.")

            # shuffle segments and create flattened input ids with cls and sep tokens
            batch["input_ids"].append(
                flatten_and_shuffle_input_ids(
                    sample_input_ids=input_ids_per_segment,
                    segment_order_labels=segment_order_labels,
                    sep_token_id=tokenizer.sep_token_id,
                    bos_token_id=tokenizer.cls_token_id,
                )
            )
            batch["so_labels"].append(segment_order_labels)
        else:
            input_ids = [tokenizer.cls_token_id]
            for segment_ids in input_ids_per_segment:
                input_ids.extend(segment_ids + [tokenizer.sep_token_id])
            batch["input_ids"].append(input_ids)

    return batch


# functions to mask tokens for mlm (masked language modeling) task


def _is_new_word(piece: str, word_prefix: Optional[str] = None, subword_prefix: Optional[str] = None) -> bool:
    """Check if the current word piece is the starting piece (sentence piece).

    Examples:
        word_prefix Albert tokenization: "zyx" -> ['▁zy', 'x']
        subword_prefix BERT tokenization: "zyx" -> ['zy', '##x']

    Args:
        piece: A subword token from the input sequence.
        word_prefix: A prefix indicating the start of a new word. E.g. "▁" for AlBert or "Ġ" for GPT2.
        subword_prefix: A prefix indicating the start of a consecutive subword, e.g. "##" for BERT.

    Returns:
        Boolean if the word piece indicates a new word or not.
    """

    if not subword_prefix and not word_prefix:
        return True
    elif subword_prefix and not piece.startswith(subword_prefix):
        return True
    elif word_prefix and piece.startswith(word_prefix):
        return True
    else:
        return False


def _get_whole_word_mask(
    input_tokens: List[str],
    tokenizer: PreTrainedTokenizerFast,
    mlm_probability: float = 0.15,
    word_prefix: Optional[str] = None,
    subword_prefix: Optional[str] = None,
):
    """
    Get 0/1 labels for masked tokens with whole word mask proxy
    """
    cand_indexes = []
    for i, token in enumerate(input_tokens):
        if token in tokenizer.all_special_tokens_extended:
            continue
        if len(cand_indexes) >= 1 and not _is_new_word(token, word_prefix, subword_prefix):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)
    num_to_predict = max(1, int(round(len(input_tokens) * mlm_probability)))

    mask_labels = torch.zeros((len(input_tokens),), dtype=torch.long)
    covered_indexes = set()

    for index_set in cand_indexes:
        if len(covered_indexes) >= num_to_predict:
            break

        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(covered_indexes) + len(index_set) > num_to_predict:
            continue

        is_any_index_covered = any(index in covered_indexes for index in index_set)
        if is_any_index_covered:
            continue

        for index in index_set:
            covered_indexes.add(index)
            mask_labels[index] = 1

    return mask_labels


def _mask_tokens(
    inputs: torch.Tensor,
    whole_word_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    device: str = "cpu",
    mlm_apply_80_10_10: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    Set 'mask_labels' means we use whole word mask (WMM), we directly mask idxs according to it's ref.
    """
    # assert self.mlm
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)

    probability_matrix = whole_word_mask

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = probability_matrix.bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if mlm_apply_80_10_10:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # 110 as low value is picked to avoid sampling special tokens like [SEP]
        #  -> this would break the segment ordering task
        random_words = torch.randint(low=110, high=len(tokenizer), size=labels.shape, dtype=torch.long, device=device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    else:
        inputs[masked_indices] = tokenizer.mask_token_id
    return inputs, labels


def mask_whole_words(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    mlm_probability: float = 0.15,
    mlm_apply_80_10_10: bool = False,
    word_prefix: Optional[str] = None,
    subword_prefix: Optional[str] = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply whole-word masking.

    See https://arxiv.org/pdf/2202.08005.pdf for new findings regarding masking.
    """
    whole_word_mask = []
    for b in input_ids:
        ref_tokens = tokenizer.convert_ids_to_tokens(b)
        whole_word_mask.append(
            _get_whole_word_mask(ref_tokens, tokenizer, mlm_probability, word_prefix, subword_prefix)
        )

    whole_word_mask = torch.stack(whole_word_mask)

    input_ids, labels = _mask_tokens(input_ids, whole_word_mask, tokenizer, device, mlm_apply_80_10_10)
    return input_ids, labels


def create_token_type_ids(input_ids: torch.Tensor, sep_token_id: int, device: str) -> torch.Tensor:
    segment_indices = (input_ids == sep_token_id).cumsum(dim=1)
    # add 0 ids to left side of samples and cut of last element
    zeros = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=device)
    token_type_ids = torch.cat([zeros, segment_indices], dim=1)[:, :-1]
    # replace padding token positions with id 0
    token_type_ids[input_ids == 0] = 0
    return token_type_ids


def shuffle_segments(segments_dict: dict, keys_to_shuffle: list):
    seg_len = len(segments_dict[keys_to_shuffle[0]])
    indices = list(range(seg_len))
    random.shuffle(indices)

    for key in keys_to_shuffle:
        segments_dict[key] = [segments_dict[key][idx] for idx in indices]

    return segments_dict
