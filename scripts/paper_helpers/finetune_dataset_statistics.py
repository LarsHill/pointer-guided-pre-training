import json
import os
from collections import Counter, defaultdict
from typing import List

import numpy as np
import tqdm

from llm import project_path
from llm.data.dataset import DataIterator
from llm.train import init_iterators, init_tokenizer


def calculate_gini_impurity(class_distribution: List[int]) -> float:
    """Calculate the Gini impurity for a given class distribution.

    The Gini impurity is calculated as:
    \( G = 1 - \sum_{i=1}^{C} (p_i)^2 \)
    where \( p_i \) is the proportion of instances belonging to class \( i \),
    and \( C \) is the number of classes.

    A low Gini impurity indicates a more pure (less diverse) distribution, with
    a value of 0 representing no impurity (all instances belong to one class).
    A high Gini impurity indicates a more diverse or balanced class distribution.

    Args:
        class_distribution (List[int]): A list of integers where each element
            represents the number of instances in each class.

    Returns:
        float: The Gini impurity of the class distribution.
    """
    total_instances = sum(class_distribution)
    class_probabilities = [count / total_instances for count in class_distribution]
    gini_impurity = 1 - sum(p**2 for p in class_probabilities)
    return gini_impurity


def balance(class_distribution):
    from numpy import log

    n = sum(class_distribution)
    k = len(class_distribution)
    H = -sum([(count / n) * log((count / n)) for count in class_distribution])  # shannon entropy
    return H / log(k)


def calculate_entropy(class_distribution: List[int]) -> float:
    """Calculate the entropy for a given class distribution.

    The entropy is calculated as:
    \( H = -\sum_{i=1}^{C} p_i \log_2(p_i) \)
    where \( p_i \) is the proportion of instances belonging to class \( i \),
    and \( C \) is the number of classes.

    A low entropy indicates a more pure (less diverse) distribution, with
    a value of 0 representing no entropy (all instances belong to one class).
    A high entropy indicates a more diverse or balanced class distribution.

    Args:
        class_distribution (List[int]): A list of integers where each element
            represents the number of instances in each class.

    Returns:
        float: The entropy of the class distribution.
    """
    total_instances = sum(class_distribution)
    class_probabilities = [count / total_instances for count in class_distribution]
    entropy = -sum(p * np.log2(p) for p in class_probabilities if p > 0)  # Avoid log(0)
    return entropy


def calculate_imbalance_ratio(class_distribution: List[int]) -> float:
    """Calculate the Imbalance Ratio for a given class distribution.

    The Imbalance Ratio (IR) is calculated as:
    \( IR = \frac{\max(\text{{class_distribution}})}{\min(\text{{class_distribution}})} \)

    A low IR indicates a more balanced class distribution (approaching 1),
    while a high IR indicates a greater imbalance, with larger values showing
    more disparity between the majority and minority classes.

    Args:
        class_distribution (List[int]): A list of integers where each element
            represents the number of instances in each class.

    Returns:
        float: The Imbalance Ratio of the class distribution.
    """
    max_instances = max(class_distribution)  # Majority class
    min_instances = min(class_distribution)  # Minority class
    imbalance_ratio = max_instances / min_instances
    return imbalance_ratio


def get_dataset_statistics(data_iterator: DataIterator):  # split: Split
    all_labels = []
    number_blobs_annotated = 0
    number_blobs_annotated_list = []
    num_segments_per_req = []

    num_documents = 0
    num_segments = 0
    num_samples = 0
    num_tokens = 0

    for article_samples in data_iterator():
        num_documents += 1
        num_samples += len(article_samples)
        doc_num_segments = sum(len(sample["input_ids_per_segment"]) for sample in article_samples)
        num_segments += doc_num_segments
        num_tokens += sum(len(segment) for sample in article_samples for segment in sample["input_ids_per_segment"])

        number_blobs_annotated_doc = 0
        segments_per_req = defaultdict(int)

        for sample in article_samples:
            labels: list[str] | list[list[str]] = sample.get("labels")
            if labels:
                # add number of annotated blobs
                number_blobs_annotated += sum(1 for label in labels if label)
                number_blobs_annotated_doc += sum(1 for label in labels if label)

                # get all labels
                # multi-label setting: labels: List[List[str]]
                if isinstance(labels[0], list):
                    for label in labels:
                        for l in label:
                            segments_per_req[l] += 1
                        all_labels.extend(label)
                # multi-class setting: labels: List[str]
                else:
                    all_labels.extend(labels)
                    for label in labels:
                        segments_per_req[label] += 1

        number_blobs_annotated_list.append(number_blobs_annotated_doc / doc_num_segments)
        num_segments_per_req.append(np.mean(list(segments_per_req.values())))

    label_to_idx = {label: i for i, label in enumerate(sorted(set(all_labels)))}
    label_to_support = Counter(all_labels)
    label_support = np.array([label_to_support[k] for k in label_to_idx])
    label_no_support = number_blobs_annotated - label_support
    label_weights = label_no_support / label_support

    segments_per_doc = num_segments / num_documents
    samples_per_doc = num_samples / num_documents
    tokens_per_doc = num_tokens / num_documents
    segments_per_sample = num_segments / num_samples

    statistics = {
        "label_map": label_to_idx,
        "label_to_support": label_to_support,
        "label_weights": label_weights.tolist(),
        "segments_per_doc": segments_per_doc,
        "samples_per_doc": samples_per_doc,
        "tokens_per_doc": tokens_per_doc,
        "segments_per_sample": segments_per_sample,
        "num_documents": num_documents,
        "num_segments": num_segments,
        "num_segments_annotated": number_blobs_annotated,
        "num_segments_annotated_percent": 100 * number_blobs_annotated / num_segments,
        "num_segments_annotated_list": number_blobs_annotated_list,
        "num_segments_annotated_macro_avg_percent": 100 * np.mean(number_blobs_annotated_list),
        "num_samples": num_samples,
        "num_tokens": num_tokens,
        "num_classes": len(label_to_support),
        "num_labels": sum(label_to_support.values()),
        "gini_impurity": calculate_gini_impurity(list(label_to_support.values())),
        "imbalance_ratio": calculate_imbalance_ratio(list(label_to_support.values())),
        "num_segments_per_req_list": num_segments_per_req,
        "num_segments_per_req_macro_avg": np.mean(num_segments_per_req),
    }
    return statistics


def main():
    dataset_paths = {
        "gri_de": ["gri_de/train.jsonl", "gri_de/dev.jsonl", "gri_de/test.jsonl"],
        "ifrs_en": ["ifrs_en/new_train.json", "ifrs_en/new_dev.json", "ifrs_en/new_test.json"],
        "cs_abstract": ["cs_abstract/train.jsonl", "cs_abstract/dev.jsonl", "cs_abstract/test.jsonl"],
        "pubmed_20k": ["pubmed_20k/train.jsonl", "pubmed_20k/dev.jsonl", "pubmed_20k/test.jsonl"],
        "nicta_piboso": ["nicta_piboso/train.jsonl", "nicta_piboso/dev.jsonl", "nicta_piboso/test.jsonl"],
    }

    iterator_params = {
        "max_seq_len": 512,
        "max_segments_per_sample": None,
        "drop_short_samples": False,
        "combine_k_segments": None,
    }

    base_dir = os.path.join(project_path, "data", "finetuning")

    hf_tokenizer_name = os.path.join(project_path, "data", "transformers_bert_tokenizer")
    hf_model_name = "bert-base-cased"

    # init tokenizer
    tokenizer = init_tokenizer(hf_model_name, hf_tokenizer_name)

    all_stats = {}
    for name, paths in tqdm.tqdm(dataset_paths.items()):
        if name in ["gri_de", "ifrs_en"]:
            iterator_params["random_max_segments_per_sample"] = True
        else:
            iterator_params["random_max_segments_per_sample"] = False

        iterators = init_iterators(paths, base_dir, iterator_params, tokenizer)

        stats = {}
        for data_iterator, split in zip(iterators, ["train", "dev", "test"]):
            stats[split] = get_dataset_statistics(data_iterator)

        # aggregate stats across splits
        num_documents = sum(split_stats["num_documents"] for split_stats in stats.values())
        num_segments = sum(split_stats["num_segments"] for split_stats in stats.values())
        num_segments_annotated = sum(split_stats["num_segments_annotated"] for split_stats in stats.values())
        num_samples = sum(split_stats["num_samples"] for split_stats in stats.values())
        num_tokens = sum(split_stats["num_tokens"] for split_stats in stats.values())

        num_segments_annotated_macro_avg_percent = 100 * np.mean(
            [count for split_stats in stats.values() for count in split_stats["num_segments_annotated_list"]]
        )
        num_segments_per_req_macro_avg = np.mean(
            [count for split_stats in stats.values() for count in split_stats["num_segments_per_req_list"]]
        )

        label_to_support = defaultdict(int)
        for split_stats in stats.values():
            for label, support in split_stats["label_to_support"].items():
                label_to_support[label] += support

        segments_per_doc = num_segments / num_documents
        samples_per_doc = num_samples / num_documents
        tokens_per_doc = num_tokens / num_documents
        segments_per_sample = num_segments / num_samples
        num_segments_annotated_percent = (100 * num_segments_annotated / num_segments,)
        gini_impurity = calculate_gini_impurity(list(label_to_support.values()))
        imbalance_ratio = calculate_imbalance_ratio(list(label_to_support.values()))
        num_classes = len(label_to_support)
        num_labels = sum(label_to_support.values())

        stats["all"] = {
            "num_documents": num_documents,
            "num_segments": num_segments,
            "num_segments_annotated": num_segments_annotated,
            "num_samples": num_samples,
            "num_tokens": num_tokens,
            "label_to_support": label_to_support,
            "segments_per_doc": segments_per_doc,
            "samples_per_doc": samples_per_doc,
            "tokens_per_doc": tokens_per_doc,
            "segments_per_sample": segments_per_sample,
            "num_segments_annotated_percent": num_segments_annotated_percent,
            "num_segments_annotated_macro_avg_percent": num_segments_annotated_macro_avg_percent,
            "gini_impurity": gini_impurity,
            "imbalance_ratio": imbalance_ratio,
            "num_classes": num_classes,
            "num_labels": num_labels,
            "num_segments_per_req_macro_avg": num_segments_per_req_macro_avg,
            "tokens_per_segment": num_tokens / num_segments,
        }

        all_stats[name] = stats

    save_dir = os.path.join(project_path, "data", "finetuning_stats")
    os.makedirs(save_dir, exist_ok=True)

    json.dump(all_stats, open(os.path.join(save_dir, "all_stats.json"), "w"), indent=4)


if __name__ == "__main__":
    main()
