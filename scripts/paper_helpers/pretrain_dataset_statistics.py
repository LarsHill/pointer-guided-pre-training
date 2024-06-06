import json
import multiprocessing
import os
from copy import deepcopy

from tqdm import tqdm

from llm import project_path
from llm.data.collate import COLLATORS
from llm.train import init_datasets, init_tokenizer


def compute_dataset_stats(path, base_dir, dataset_params_, iterator_params, tokenizer, collate_fn):
    dataset_params = deepcopy(dataset_params_)
    dataset_params["train_paths"] = [path]
    train_dataset, _ = init_datasets(dataset_params, iterator_params, tokenizer, collate_fn)

    num_documents = 0
    num_segments = 0
    num_samples = 0
    num_tokens = 0

    for samples in tqdm(train_dataset.data_iterator(), desc=f"Stats for {path}"):
        num_documents += 1
        num_samples += len(samples)
        num_segments += sum(len(sample["input_ids_per_segment"]) for sample in samples)
        num_tokens += sum(len(segment) for sample in samples for segment in sample["input_ids_per_segment"])

    segments_per_doc = num_segments / num_documents
    samples_per_doc = num_samples / num_documents
    tokens_per_doc = num_tokens / num_documents
    segments_per_sample = num_segments / num_samples

    # Save stats as json
    stats = {
        "num_documents": num_documents,
        "num_segments": num_segments,
        "num_samples": num_samples,
        "num_tokens": num_tokens,
        "segments_per_doc": segments_per_doc,
        "samples_per_doc": samples_per_doc,
        "tokens_per_doc": tokens_per_doc,
        "segments_per_sample": segments_per_sample,
    }

    print(path)
    print(json.dumps(stats, indent=4))

    save_dir, name = os.path.split(os.path.join(base_dir, path))
    name_formatted = name.split(".")[0]
    save_path = os.path.join(save_dir, f"{name_formatted}_stats.json")
    json.dump(stats, open(save_path, "w"), indent=4)

    return stats


def main():
    train_paths = [
        "wiki/enwiki.jsonl.gz",
        "wiki/dewiki.jsonl.gz",
        "banz/banz.jsonl.gz",
        "news/news_de.jsonl.gz",
        "news/news_en.jsonl.gz",
    ]

    iterator_params = {
        "max_seq_len": 512,
        "max_segments_per_sample": None,
        "drop_short_samples": False,
        "combine_k_segments": None,
    }

    collator_params = {
        "name": "language_modeling",
        "mlm_probability": None,
        "mlm_apply_80_10_10": False,
        "segment_ordering": True,
        "next_sentence_prediction": False,
    }

    dataset_params_ = {
        "base_dir": "/data/datasets",
        "eval_paths": [],
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "shuffle": True,
        "drop_last": False,
        "buffer_size": 1000,
    }

    base_dir = dataset_params_["base_dir"]

    hf_tokenizer_name = "/data/datasets/transformers_bert_tokenizer"
    hf_model_name = "bert-base-cased"

    # init tokenizer
    tokenizer = init_tokenizer(hf_model_name, hf_tokenizer_name)

    # init data collator
    name = collator_params.pop("name")
    collate_fn = COLLATORS[name](tokenizer, "cpu", **collator_params)

    # init training and evaluation dataset
    if not os.path.isabs(dataset_params_["base_dir"]):
        dataset_params_["base_dir"] = os.path.join(project_path, dataset_params_["base_dir"])

    # Use multiprocessing pool to compute stats in parallel
    with multiprocessing.Pool(processes=len(train_paths)) as pool:
        for path in train_paths:
            pool.apply_async(
                compute_dataset_stats, args=(path, base_dir, dataset_params_, iterator_params, tokenizer, collate_fn)
            )
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
