import random
from typing import (
    Any,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch


def max_neg_value(tensor: torch.Tensor) -> float:
    """Returns the maximum negative value of a given torch data type."""
    return -torch.finfo(tensor.dtype).max


def get_available_devices(use_cuda: bool = True, cuda_ids: Optional[int | list[int]] = None) -> list[str]:
    if cuda_ids is not None and isinstance(cuda_ids, int):
        cuda_ids = [cuda_ids]
    if (use_cuda or cuda_ids) and torch.cuda.is_available():
        if cuda_ids is not None:
            devices = [f"cuda:{id_}" for id_ in cuda_ids]
        else:
            devices = [f"cuda:{id_}" for id_ in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    return devices


def argsort(seq: Sequence, reverse: bool = False) -> list[int]:
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def batch_iterate_sequence(sequence: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(sequence), batch_size):
        yield sequence[i : i + batch_size]


def change_order(sequence: Sequence):
    if len(sequence) == 1:
        return None

    for i in reversed(range(1, len(sequence))):
        # pick an element in x[:i] with which to exchange x[i]
        j = int(random.random() * i)
        sequence[i], sequence[j] = sequence[j], sequence[i]


def assign_input_ids_to_samples(
    input_ids: list[list[int]],
    max_seq_len: int,
    max_segments_per_sample: Optional[int] = None,
    random_max_segments_per_sample: bool | None = None,
) -> list[list[list[int]]]:
    samples = []
    segments = []
    previous_seq_len = 1  # [CLS] token: added later during collate fn
    current_num_segments = 0

    max_segments_per_sample_options = [None, 3, 4, 5, 6, 7]

    for segment_input_ids in input_ids:
        segment_len = len(segment_input_ids) + 1  # [SEP] token: added later during collate fn
        current_seq_len = previous_seq_len + segment_len

        if random_max_segments_per_sample:
            max_segments_per_sample = random.choice(max_segments_per_sample_options)

        if (
            (max_segments_per_sample is not None and current_num_segments < max_segments_per_sample)
            or max_segments_per_sample is None
        ) and current_seq_len <= max_seq_len:
            segments.append(segment_input_ids)
            previous_seq_len = current_seq_len
            current_num_segments += 1

        else:
            samples.append(segments)
            segments = [segment_input_ids]
            previous_seq_len = segment_len + 1  # [CLS] token: added later during collate fn
            current_num_segments = 1
    samples.append(segments)
    return samples


def combine_k_segment_ids(sample_input_ids: list[list[int]], k: int) -> list[list[int]]:
    return [
        [id_ for segment in seg_ids for id_ in segment]
        for seg_ids in batch_iterate_sequence(sample_input_ids, batch_size=k)
    ]


def to_device(
    obj: Any,
    device: Union[str, torch.device],
    detach: bool = False,
    ignore_keys: Optional[Union[str, List[str]]] = None,
) -> Any:
    if ignore_keys is None:
        ignore_keys = []
    elif isinstance(ignore_keys, str):
        ignore_keys = [ignore_keys]

    if torch.is_tensor(obj):
        return obj.detach().to(device) if detach else obj.to(device)
    elif isinstance(obj, MutableMapping):
        return {k: to_device(v, device, detach) if k not in ignore_keys else v for k, v in obj.items()}
    elif isinstance(obj, (List, Tuple, Set)):
        return type(obj)(to_device(v, device, detach) for v in obj)
    else:
        return obj
