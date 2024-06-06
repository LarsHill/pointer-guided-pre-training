import gzip
import json
from typing import Iterator, Optional

from transformers import PreTrainedTokenizerFast

from llm.utils import assign_input_ids_to_samples, combine_k_segment_ids


def create_samples(
    article: dict,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int,
    max_segments_per_sample: Optional[int] = None,
    combine_k_segments: Optional[int] = None,
    drop_short_samples: bool = False,
    random_max_segments_per_sample: bool | None = None,
) -> list[dict]:
    raw_segments: list[str] = article["raw_segments"]
    # tokenize the parsed segments
    input_ids: list[list[int]] = tokenizer(
        raw_segments, add_special_tokens=False, truncation=True, max_length=max_seq_len - 2
    )["input_ids"]

    # create samples based on mex_seq_len
    samples_input_ids: list[list[list[int]]] = assign_input_ids_to_samples(
        input_ids,
        max_seq_len=max_seq_len,
        max_segments_per_sample=max_segments_per_sample,
        random_max_segments_per_sample=random_max_segments_per_sample,
    )

    # combine k segment ids in 1 segment
    if combine_k_segments is not None:
        samples_input_ids = [combine_k_segment_ids(sample, k=combine_k_segments) for sample in samples_input_ids]

    # drop short samples
    if drop_short_samples:
        samples_input_ids = [sample for sample in samples_input_ids if len(sample) > 1]

    # assign labels to samples
    # NOTE: This assignment only works if combine_k_segments and drop_short_samples is None
    if "labels" in article and article["labels"]:
        samples_labels = []
        sentence_counter = 0
        for sample in samples_input_ids:
            sample_labels = []
            for _ in sample:
                sample_labels.append(article["labels"][sentence_counter])
                sentence_counter += 1
            samples_labels.append(sample_labels)
        return [
            {"input_ids_per_segment": sample, "labels": s_label}
            for sample, s_label in zip(samples_input_ids, samples_labels)
        ]

    return [{"input_ids_per_segment": sample, "article": article} for sample in samples_input_ids]


def wikitext_iterator(path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict) -> Iterator[list[dict]]:
    with gzip.open(path, "r") as f:
        for line in f:
            article: dict = json.loads(line)
            if "samples" in article:
                article["raw_segments"]: list[str] = [
                    seg["text"] for seg in article["samples"] if seg["text"] is not None
                ]
                samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
                if len(samples) > 0:
                    yield samples


def wiki_iterator(path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict) -> Iterator[list[dict]]:
    with gzip.open(path, "r") as f:
        for line in f:
            article: dict = json.loads(line)
            blobs = article.get("blobs")
            if blobs is None:
                blobs = article.get("segments")
            if blobs is not None and len(blobs) > 0:
                article["raw_segments"]: list[str] = [seg["value"] for seg in blobs if seg["value"] is not None]
                samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
                if len(samples) > 0:
                    yield samples


def news_en_iterator(path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict) -> Iterator[list[dict]]:
    with gzip.open(path, "r") as f:
        for line in f:
            article: dict = json.loads(line)
            if "blobs" in article and len(article["blobs"]) > 0:
                article["raw_segments"]: list[str] = article["blobs"]
                samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
                if len(samples) > 0:
                    yield samples


def process_banz(line, tokenizer: PreTrainedTokenizerFast, loading_params: dict):
    article: dict = json.loads(line)
    if "segments" in article and len(article["segments"]) > 0:
        article["raw_segments"]: list[str] = [seg["value"] for seg in article["segments"] if seg["value"] is not None]
        samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
        if len(samples) > 0:
            return samples


# Function to submit tasks to the pool asynchronously
def submit_tasks(path, pool, result_queue, process_banz_fn):
    def collect_result(result):
        if result:
            result_queue.put(result)

    with gzip.open(path, "r") as f:
        for line in f:
            pool.apply_async(process_banz_fn, args=(line,), callback=collect_result)


def banz_iterator(path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict) -> Iterator[list[dict]]:
    with gzip.open(path, "r") as f:
        for line in f:
            article: dict = json.loads(line)
            if "segments" in article and len(article["segments"]) > 0:
                article["raw_segments"]: list[str] = [
                    seg["value"] for seg in article["segments"] if seg["value"] is not None
                ]
                samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
                if len(samples) > 0:
                    yield samples


def news_de_iterator(path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict) -> Iterator[list[dict]]:
    with gzip.open(path, "r") as f:
        for line in f:
            article: dict = json.loads(line)
            if "sentences" in article and len(article["sentences"]) > 0:
                article["raw_segments"]: list[str] = article["sentences"]
                samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
                if len(samples) > 0:
                    yield samples


def csabstract_iterator(
    path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict, num_workers: int = None, worker_id: int = None
) -> Iterator[list[dict]]:
    main_process_only = True if num_workers is None and worker_id is None else False
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if main_process_only or (i + worker_id) % num_workers == 0:
                article: dict = json.loads(line)
                if "sentences" in article and len(article["sentences"]) > 0:
                    article["raw_segments"]: list[str] = article["sentences"]
                    samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)
                    yield samples


def process_ifrs(engagement, tokenizer, loading_params):
    doc = engagement["doc"]["documents"][0]

    blobs = doc["blobs"]
    recs = doc["recommendations"]

    raw_segments = [blob["value"] for blob in blobs]

    labels = []
    for blob in blobs:
        blob_labels = [rec["requirementId"] for rec in recs if rec["blobId"] == blob["blobId"] and rec["match"]]
        labels.append(blob_labels)

    article = {"raw_segments": raw_segments, "labels": labels}
    samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)

    # add document ids to samples
    for sample in samples:
        sample["doc_id"] = doc["name"]

    return samples


def ifrs_iterator(
    path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict, num_workers: int = None, worker_id: int = None
):
    engagements = json.load(open(path, "r"))

    # engagements = data["rows"]
    main_process_only = True if num_workers is None and worker_id is None else False

    for i, engagement in enumerate(engagements):
        if main_process_only or (i + worker_id) % num_workers == 0:
            doc = engagement["doc"]["documents"][0]

            blobs = doc["blobs"]
            recs = doc["recommendations"]

            raw_segments = [blob["value"] for blob in blobs]

            labels = []
            for blob in blobs:
                blob_labels = [rec["requirementId"] for rec in recs if rec["blobId"] == blob["blobId"] and rec["match"]]
                labels.append(blob_labels)

            article = {"raw_segments": raw_segments, "labels": labels}
            samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)

            # add document ids to samples
            for sample in samples:
                sample["doc_id"] = doc["name"]

            yield samples


def gri_de_iterator(
    path: str, tokenizer: PreTrainedTokenizerFast, loading_params: dict, num_workers: int = None, worker_id: int = None
) -> Iterator[list[dict]]:
    main_process_only = True if num_workers is None and worker_id is None else False
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if main_process_only or (i + worker_id) % num_workers == 0:
                article: dict = json.loads(line)
                if "raw_segments" in article and len(article["raw_segments"]) > 0:
                    samples: list[dict] = create_samples(article, tokenizer=tokenizer, **loading_params)

                    # add document ids to samples
                    for sample in samples:
                        sample["doc_id"] = article["id_"]

                    yield samples


DATA_ITERATORS = {
    "banz": banz_iterator,
    "news-de": news_de_iterator,
    "news-en": news_en_iterator,
    "wiki": wiki_iterator,
    "wikitext": wikitext_iterator,
    "cs_abstract": csabstract_iterator,
    "pubmed_20k": csabstract_iterator,
    "nicta_piboso": csabstract_iterator,
    "ifrs_en": ifrs_iterator,
    "gri_de": gri_de_iterator,
}
