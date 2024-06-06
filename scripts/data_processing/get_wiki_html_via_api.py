import argparse
import gzip
import json
import multiprocessing as mp
import os
from typing import IO, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

f: IO | None = None
num_articles: int | None = None
pbar: tqdm | None = None


@retry(wait=wait_exponential(multiplier=1, min=10, max=60), stop=stop_after_attempt(3))
def get_response(api_url: str, params: Dict) -> requests.Response:
    response = requests.get(url=api_url, params=params, timeout=10)
    if response.status_code == 200:
        return response
    else:
        raise requests.exceptions.HTTPError


def download_wiki_html(api_url: str, params: Dict, pageid: str) -> Dict:
    params["pageid"] = pageid

    response = get_response(api_url, params)
    page = response.json()["parse"]

    text_html = page["text"]["*"]

    # Exclude disambiguation pages
    if "Template:Dmbox" not in text_html:
        return {"title": page["title"], "pageid": pageid, "text": text_html}


def callback(result: Optional[Dict] = None):
    if result:
        pbar.update()
        res = json.dumps(result) + "\n"
        f.write(res.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # argument for input wikipedia dump
    parser.add_argument(
        "--article-names",
        type=str,
        help="Path to created json file, wiki_page_ids.json, containing all wikipedia page ids and titles.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="Path to output directory",
    )

    parser.add_argument(
        "--language",
        default="en",
        type=str,
        help="Language of dump",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    in_file = args.article_names
    out_dir = args.out

    api_url = f"https://{args.language}.wikipedia.org/w/api.php"
    params = {"action": "parse", "prop": "text", "format": "json"}
    articles = json.load(open(in_file, "r"))

    global num_articles
    num_articles = len(articles)

    global pbar
    pbar = tqdm(total=num_articles)

    global f
    f = gzip.open(os.path.join(out_dir, f"{args.language}_wiki_html_articles.jsonl.gz"), "wb")

    # call api and write html articles to file using multiprocessing
    with mp.Pool() as pool:
        for article in list(articles):
            pool.apply_async(download_wiki_html, args=(api_url, params, article), callback=callback)

        pool.close()
        pool.join()

    f.close()
    pbar.close()


if __name__ == "__main__":
    main()
