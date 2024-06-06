"""Module to hold dataset and data loading logic."""

import itertools
import logging
import random
from collections import defaultdict
from copy import copy
from functools import partial
from typing import Any, Callable, Iterator, List, Optional

import numpy as np
from torch.utils.data import IterableDataset

from llm.utils import argsort

logger = logging.getLogger(__name__)

DataIterator = Callable[..., Iterator[list[dict]]]
# """Protocol to define the output of a callable that returns an Iterator over documents.
#
# A document is defined as a list of dict, where each dict represents a single sample.
# """


def combine_iterators(iterator_fns: list[Callable[..., Iterator]], weights: Optional[list[float]] = None) -> Iterator:
    """Samples from multiple iterators, given a provided weight distribution.

    Args:
        iterator_fns: A list of iterator generating functions.
        weights: Holds the sampling probability of each iterator. If weights is None, a uniform distribution is assumed.

    Returns:
        A merged iterator.
    """

    if weights is None:
        weights = [1.0 / len(iterator_fns)] * len(iterator_fns)

    # create a list of dataset ids
    dataset_ids = list(range(len(iterator_fns)))
    # convert all iterator generating functions to iterators
    iterators = [iterator_fn() for iterator_fn in iterator_fns]
    # create a shallow copy of the provided weights so that only the copy is mutated but the original remains unchanged
    weights_copy = copy(weights)

    while True:
        # sample a data iterator id based on the provided weight distribution
        dataset_id = random.choices(dataset_ids, k=1, weights=weights_copy)[0]
        try:
            # yield the next sample from that sampled data iterator
            yield next(iterators[dataset_id])
        except StopIteration:
            # If a data iterator reaches the end, we set its sampling probability to 0
            weights_copy[dataset_id] = 0.0
            # If all sampling probabilities are set to 0 we know that each data iterator is exhausted, so we stop
            if sum(weights_copy) == 0:
                break


def get_data_iterator(
    iterator_fns: DataIterator | list[DataIterator],
    weights: Optional[list[float]] = None,
) -> DataIterator:
    if isinstance(iterator_fns, list) and len(iterator_fns) > 1:
        data_iterator = partial(combine_iterators, iterator_fns, weights)
    elif isinstance(iterator_fns, list) and iterator_fns:
        data_iterator = iterator_fns[0]
    else:
        data_iterator = iterator_fns
    return data_iterator


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        data_iterator: DataIterator | list[DataIterator],
        weights: Optional[list[float]] = None,
        buffer_size: int = 1000,
        batch_size: int = 1,
        shuffle_documents: Optional[bool] = None,
        shuffle_samples: Optional[bool] = None,
        drop_last: Optional[bool] = None,
        collate_fn: Optional[Callable] = None,
        apply_weighted_sampling: bool | str = False,
    ):
        super().__init__()

        self.data_iterator: DataIterator = get_data_iterator(iterator_fns=data_iterator, weights=weights)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle_documents = shuffle_documents
        self.shuffle_samples = shuffle_samples
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn else lambda batch: batch
        self.apply_weighted_sampling = apply_weighted_sampling

        # contains a list of independent samples (e.g. documents)
        # and each document is split up in subsequent chunks (e.g. segments)
        self._buffer: list[list[dict]] = []
        self._current_buffer_size = 0
        self._cache = defaultdict(list)
        self._current_cache_size = 0

        self._num_batches: Optional[int] = None

    #
    @property
    def num_batches(self):
        if self._num_batches is not None:
            return self._num_batches

        self._num_batches = sum(1 for _ in self.__iter__())
        return self._num_batches

    @staticmethod
    def _shuffle_buffer(buffer_partitions: List[List[Any]]):
        for partition in buffer_partitions:
            random.shuffle(partition)

    @staticmethod
    def _partition_greedily(
        sequence: List[List[Any]],
        num_partitions: int,
        cached_partitions: Optional[List[List[Any]]],
    ) -> List[List[Any]]:
        initial_partition_lens = [len(partition) for partition in cached_partitions]

        lens = [len(elem) for elem in sequence]
        indices = argsort(lens, reverse=True)

        partitions = [[] for _ in range(num_partitions)]

        for idx in indices:
            smallest_partition_idx = sorted(
                range(num_partitions),
                key=lambda ix: initial_partition_lens[ix],
                reverse=False,
            )[0]
            initial_partition_lens[smallest_partition_idx] += len(sequence[idx])
            partitions[smallest_partition_idx].append(sequence[idx])
        return partitions

    def _create_memory_clearing_mask(self, buffer: List[List[List[Any]]]) -> np.ndarray:
        num_buffer_batches = max([sum(len(article) for article in partition) for partition in buffer])

        # create memory clearing mask for buffer
        memory_clearing_mask = np.zeros((num_buffer_batches, self.batch_size), dtype=bool)
        for partition_id, partition in enumerate(buffer):
            batch_idx = 0
            for article in partition:
                memory_clearing_mask[batch_idx, partition_id] = 1
                batch_idx += len(article)

        # overwrite mask entries for cached batches
        for batch_idx, (cached_batch, cached_mask) in enumerate(zip(*self._cache.values())):
            for partition_idx, (sample, mask_value) in enumerate(zip(*(cached_batch, cached_mask))):
                if sample is not None:
                    memory_clearing_mask[batch_idx, partition_idx] = 0
                if mask_value == 1.0:
                    memory_clearing_mask[batch_idx, partition_idx] = 1

        return memory_clearing_mask

    def _process_buffer(self, buffer: list[list[dict]]):
        # shuffle samples (list os segments) within documents
        # TODO: This is incompatible with transformer-xl recurrence
        if self.shuffle_samples:
            self._shuffle_buffer(buffer)

        # load cached partitions (empty list if self._cache is empty)
        cached_partitions = [
            [elem for elem in partition if elem is not None] for partition in zip(*self._cache["batches"])
        ]
        if not cached_partitions:
            cached_partitions = [[] for _ in range(self.batch_size)]

        # group buffered articles in k (batch_size) partitions of approximately equal size
        buffer: List[List[List[Any]]] = self._partition_greedily(buffer, self.batch_size, cached_partitions)

        # shuffle documents within partitions
        if self.shuffle_documents:
            self._shuffle_buffer(buffer)

        # combine cached partitions with new buffered partitions
        buffer = [
            [cached_partition] + partition if cached_partition else partition
            for cached_partition, partition in zip(cached_partitions, buffer)
        ]

        # create a mask of shape (num_buffer_batches x batch_size) which indicates the start of a new document
        #  to clear the network's memory before that sample
        memory_clearing_mask = self._create_memory_clearing_mask(buffer)

        # flatten the samples in each partition
        buffer = [[sample for article in partition for sample in article] for partition in buffer]

        return buffer, memory_clearing_mask

    def _reset_buffer(self):
        self._buffer = []
        self._current_buffer_size = 0

    def _reset_cache(self):
        self._cache = defaultdict(list)
        self._current_cache_size = 0

    def _yield_buffer(self):
        for samples in self.data_iterator():
            if not samples:
                continue

            self._buffer.append(samples)
            self._current_buffer_size += 1
            if self._current_buffer_size == self.buffer_size:
                yield self._buffer

        # if the buffer is non-empty but file is exhausted -> yield the remaining buffer
        if self._buffer:
            yield self._buffer

    def _yield_batch_from_buffer(self):
        for buffer in self._yield_buffer():
            if self.apply_weighted_sampling == "ros":
                new_buffer = []
                for doc_samples in buffer:
                    pos_samples = []
                    neg_samples = []
                    for sample in doc_samples:
                        if sum(len(label) for labels in sample["labels"] for label in labels) > 0:
                            pos_samples.append(sample)
                        else:
                            neg_samples.append(sample)
                    num_pos_samples = len(pos_samples)
                    num_neg_samples = len(neg_samples)
                    pos_samples += random.choices(pos_samples, k=num_neg_samples - num_pos_samples)

                    new_doc_samples = pos_samples + neg_samples
                    random.shuffle(new_doc_samples)
                    new_buffer.append(new_doc_samples)

                buffer = new_buffer
            elif self.apply_weighted_sampling == "rous":
                new_buffer = []
                for doc_samples in buffer:
                    pos_samples = [
                        sample
                        for sample in doc_samples
                        if sum(len(label) for labels in sample["labels"] for label in labels) > 0
                    ]
                    neg_samples = [
                        sample
                        for sample in doc_samples
                        if sum(len(label) for labels in sample["labels"] for label in labels) == 0
                    ]

                    # Calculate the number of samples to select from each list to maintain the original buffer size
                    total_samples = len(doc_samples)
                    num_pos_samples = len(pos_samples)
                    num_neg_samples = len(neg_samples)

                    # If there are no positive or no negative samples, just use the original samples
                    if num_pos_samples == 0 or num_neg_samples == 0:
                        new_buffer.append(doc_samples)
                        continue

                    # Determine the number of samples to oversample/undersample for balance
                    if num_pos_samples > num_neg_samples:
                        # Oversample negative samples and undersample positive samples
                        neg_samples = random.choices(neg_samples, k=total_samples // 2)
                        pos_samples = random.sample(pos_samples, k=total_samples - len(neg_samples))
                    else:
                        # Oversample positive samples and undersample negative samples
                        pos_samples = random.choices(pos_samples, k=total_samples // 2)
                        neg_samples = random.sample(neg_samples, k=total_samples - len(pos_samples))

                    # Combine and shuffle the balanced samples
                    new_doc_samples = pos_samples + neg_samples
                    random.shuffle(new_doc_samples)
                    new_buffer.append(new_doc_samples)

                buffer = new_buffer

            buffer, buffer_memory_clearing_mask = self._process_buffer(buffer)

            self._reset_buffer()
            self._reset_cache()

            for batch, memory_clearing_mask in zip(itertools.zip_longest(*buffer), buffer_memory_clearing_mask):
                if None in batch:
                    self._cache["batches"].append(batch)
                    self._cache["memory_masks"].append(memory_clearing_mask)
                    self._current_cache_size += 1
                    continue

                yield batch, memory_clearing_mask

    def _yield_batch_from_cache(self):
        for batch, memory_clearing_mask in zip(*self._cache.values()):
            yield batch, memory_clearing_mask
        self._reset_cache()

    def _yield_batch(self):
        yield from self._yield_batch_from_buffer()

        if self.drop_last:
            self._reset_cache()
            return
        yield from self._yield_batch_from_cache()

    def __iter__(self):
        for batch, memory_clearing_mask in self._yield_batch():
            yield self.collate_fn((batch, memory_clearing_mask))
