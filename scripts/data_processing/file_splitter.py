import argparse
import gzip
import json
import multiprocessing as mp
import os
from functools import partial
from typing import IO, Dict, Optional

from tqdm import std, tqdm


def file_reader(line: str) -> Dict:
    """
    File reader function
    Args:
        line: byte string containing the json data

    Returns: Dict formatted data

    """
    return json.loads(line)


def write_jsonl_gzip(line: Optional[dict], file: IO, pbar_total: std.tqdm, pbar_success: std.tqdm):
    """
    Writes the data to a zip file
    Args:
        line: dict containing the data content
        file: file object for target file
        pbar_success: update the progress bar
    """
    pbar_total.update()
    if line is not None:
        file.write(f"{json.dumps(line)}\n")
        pbar_success.update()


def split_jsonl_zip(
    input_file: str,
    train_ratio: float = 0.9,
    output_train_file: str = "train.jsonl.gz",
    output_val_file: str = "val.jsonl.gz",
) -> None:
    """
    Entry function for training a validation split
    Args:
        input_file: Absolute input file name
        train_ratio: Ratio into which the data is to be split
        output_train_file: Output training file name (absolute)
        output_val_file: Input training file name (absolute)
    """
    ctx = mp.get_context("spawn")

    with gzip.open(input_file, "r") as f_read, gzip.open(output_train_file, "wt") as f_train, gzip.open(
        output_val_file, "wt"
    ) as f_val:
        number_of_lines = sum(1 for _ in tqdm(f_read, desc="Reading file to get number of lines"))
        f_read.seek(0)

        # file_idxs = []
        counter = 0
        with tqdm(desc=f"Processed files", total=number_of_lines) as pbar_total, tqdm(
            desc="Successfully written"
        ) as pbar_success:
            train_callback = partial(write_jsonl_gzip, file=f_train, pbar_total=pbar_total, pbar_success=pbar_success)
            val_callback = partial(write_jsonl_gzip, file=f_val, pbar_total=pbar_total, pbar_success=pbar_success)

            # create training split
            with ctx.Pool() as pool:
                for line in f_read:
                    pool.apply_async(file_reader, args=(line,), callback=train_callback)
                    # file_idxs.append(idx)
                    counter += 1
                    if counter > (number_of_lines * train_ratio):
                        # if len(file_idxs) > (number_of_lines * train_ratio):
                        break
                pool.close()
                pool.join()

            # create validation split
            with ctx.Pool() as pool:
                for line in f_read:
                    pool.apply_async(file_reader, args=(line,), callback=val_callback)
                pool.close()
                pool.join()

    print("Splitting complete. Training data saved to", output_train_file)
    print("Validation data saved to", output_val_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        default="/data/wiki-en.jsonl.gz",
        type=str,
        help="Path processed file",
    )

    parser.add_argument(
        "--label",
        default="wiki-en",
        type=str,
        help="Specify the label of dataset",
    )

    parser.add_argument(
        "--train_ratio",
        default="0.99",
        type=str,
        help="Specify the split ratio of dataset",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    label = args.label

    file_dir = os.path.split(args.file)[0]
    output_train_file = os.path.join(file_dir, f"{label}_train.jsonl.gz")
    output_val_file = os.path.join(file_dir, f"{label}_val.jsonl.gz")

    split_jsonl_zip(
        args.file,
        train_ratio=float(args.train_ratio),
        output_train_file=output_train_file,
        output_val_file=output_val_file,
    )


if __name__ == "__main__":
    main()
