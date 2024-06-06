import argparse
import json
import os
from typing import Dict, List

import requests

from llm import project_path


# Function to parse raw data and generate JSONL format
def parse_raw_data(raw_data) -> List[Dict]:
    data_list = []
    current_abstract = {"abstract_id": 0, "sentences": [], "labels": []}

    for line in raw_data:
        if line.startswith("###"):
            # New abstract, reset current_abstract
            if current_abstract["sentences"]:
                data_list.append(current_abstract)
            current_abstract = {"abstract_id": int(line.strip("###")), "sentences": [], "labels": []}
        else:
            parts = line.split("\t")
            if len(parts) == 2:
                section, text = parts
                current_abstract["sentences"].append(text.strip())
                current_abstract["labels"].append(section.strip())

    # Add the last abstract
    if current_abstract["sentences"]:
        data_list.append(current_abstract)

    return data_list


def download_and_parse_data(base_url: str, output_dir: str):
    labels = ["dev", "test", "train"]

    for label in labels:
        # Construct the URL for the data
        url = f"{base_url}/{label}.txt"

        # Make an HTTP GET request to fetch the data
        response = requests.get(url)

        if response.status_code == 200:
            # Parse raw data
            raw_data = response.text.split("\n")
            parsed_data = parse_raw_data(raw_data)

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Write the parsed data to a JSONL file in the output directory
            output_file_path = os.path.join(output_dir, f"{label}.jsonl")
            with open(output_file_path, "w", encoding="utf-8") as jsonl_file:
                for item in parsed_data:
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Data for '{label}' has been parsed and saved to {output_file_path}")
        else:
            print(f"Failed to fetch data from URL: {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_url",
        # see here: https://github.com/Franck-Dernoncourt/pubmed-rct
        default="https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT",
        type=str,
        help="URL of the pubmed raw dataset",
    )

    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_path, "data", "finetuning", "pubmed_20k"),
        type=str,
        help="Specify the output path with filename",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    download_and_parse_data(args.base_url, args.output_dir)


if __name__ == "__main__":
    main()
