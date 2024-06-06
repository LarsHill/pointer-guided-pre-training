import gzip
import json
import multiprocessing as mp
from functools import partial
from typing import IO, Optional

import pandas as pd
from tqdm import std, tqdm


def filter_table_df(table_str: str) -> str:
    return table_str.replace("Unnamed", "")


def handle_table_data(table_str: str) -> str:
    try:
        df = pd.read_html(table_str)[0]
        if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
            cols = " [COL] ".join([" ".join(x) for x in list(df.columns)])
        else:
            cols = " [COL] ".join(list(df.columns.astype(str)))
        processed_rows = []
        df.fillna("", inplace=True)
        for idx, rows in df.iterrows():
            processed_rows.append(" [COL] ".join(list([str(row_item) for row_item in rows])))
        return filter_table_df("".join([cols, " [ROW] ", " [ROW] ".join(processed_rows)]))
    except ValueError as ve:
        print(f"Table not found error : {ve}, table_str: {table_str}")
    except IndexError as ie:
        print(f"Index error : {ie}, table_str: {table_str}")
    except Exception as e:
        print(e)


def write_jsonl_gzip(
    line: Optional[dict],
    file: IO,
    pbar_total: std.tqdm,
    pbar_success: std.tqdm,
):
    pbar_total.update()
    if line is not None:
        file.write(f"{json.dumps(line)}\n")
        pbar_success.update()


def preprocess_docs(prop: dict):
    ctx = mp.get_context("spawn")
    with gzip.open(prop["raw_data_path"], "r") as f_read, gzip.open(prop["processed_data_path"], "wt") as f_write:
        # this takes a few seconds but in return we get a progress bar for processing
        number_of_lines = sum(1 for _ in tqdm(f_read, desc="Reading file to get number of lines"))
        f_read.seek(0)
        with tqdm(desc=f"Processed {prop['label']}", total=number_of_lines) as pbar_total, tqdm(
            desc="Successfully written"
        ) as pbar_success:
            write_callback = partial(write_jsonl_gzip, file=f_write, pbar_total=pbar_total, pbar_success=pbar_success)
            with ctx.Pool() as pool:
                for line in f_read:
                    if "file_name" in prop["args"]:
                        prop["args"]["file_name"] = f_read.filename
                    pool.apply_async(
                        prop["processor"],
                        args=(line, *prop["args"].values()) if prop["args"] is not None else (line,),
                        callback=write_callback,
                    )

                pool.close()
                pool.join()
