import json
import os
from collections import defaultdict


def main():
    base_dir = "/data/datasets"
    train_paths = [
        "wiki/enwiki_stats.json",
        "wiki/dewiki_stats.json",
        "banz/banz_stats.json",
        "news/news_de_stats.json",
        "news/news_en_stats.json",
    ]

    all_stats = defaultdict(list)
    for path in train_paths:
        stats_path = os.path.join(base_dir, path)
        stats = json.load(open(stats_path, "r"))
        for k, v in stats.items():
            all_stats[k].append(v)

    all_stats["documents_sum"] = sum(all_stats["num_documents"])
    all_stats["segments_sum"] = sum(all_stats["num_segments"])
    all_stats["samples_sum"] = sum(all_stats["num_samples"])
    all_stats["tokens_sum"] = sum(all_stats["num_tokens"])

    all_stats["tokens_fraction"] = [
        round((num_tokens / all_stats["tokens_sum"]) * 100, 2) for num_tokens in all_stats["num_tokens"]
    ]
    print(sum(all_stats["tokens_fraction"]))

    print(json.dumps(all_stats, indent=4))


if __name__ == "__main__":
    main()
