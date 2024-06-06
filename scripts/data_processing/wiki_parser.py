import argparse
import datetime
import gzip
import json
import multiprocessing as mp
import unicodedata
from functools import partial
from typing import IO, Dict, List, Optional

import regex as re
from lxml import html as lh
from lxml.html.clean import Cleaner
from tqdm import std, tqdm

from llm.scripts_helper import handle_table_data

allowed_tags = [
    "html",
    "p",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "a",
]

re_br = re.compile(r"<(wbr|br|hr)(\s*\/>|>)", re.UNICODE)
re_div = re.compile(r"(?<=<)(div)(?=.*?>)", re.UNICODE)
re_math = re.compile(r'{\\displaystyle (.*?)}(?=">)', re.UNICODE | re.MULTILINE)
re_caption = re.compile(r"(?<=<caption>)(.*?)(?=</caption>)", re.UNICODE | re.MULTILINE | re.DOTALL)
re_empty_cell = re.compile(r"(?<=<td([^>]*)>)(\s+)(?=<\/td>)", re.UNICODE | re.MULTILINE | re.DOTALL)


def clean_wiki_html(text_html: str) -> lh.HtmlElement:
    """Clean the raw Wikipedia html text and create the html tree."""

    # normalize html str
    text_html = unicodedata.normalize("NFKC", text_html)

    # remove style attributes and comments
    cleaner = Cleaner(style=True, page_structure=False, safe_attrs_only=False)
    text_html = cleaner.clean_html(text_html)

    # remove displaystyle in math str
    text_html = re.sub(re_math, "\\1", text_html).replace(r"{\displaystyle ", "")

    # remove br, hr tags and html &
    text_html = text_html.replace("&amp;", "&")
    text_html = re.sub(re_br, "\n", text_html)

    # replace div by span tag
    text_html = re.sub(re_div, "span", text_html)
    text_html = text_html.replace("</div>", "</span>")

    # build element tree
    tree = lh.fromstring(text_html)

    # remove tags and their content (text and children)
    for elem in tree.xpath(
        f"//style | "
        f"//img | "
        f"//map | "
        f"//audio | "
        f"//video | "
        f"//source | "
        f"//cite | "
        f"//link | "
        f"//base | "
        f"//s | "
        f'//p[@class="mw-empty-elt"] | '  # remove empty p tags
        # remove short descriptions
        f'//span[contains(@class, "shortdescription ")] | '
        f'//span[contains(@class, "thumb ")] |'  # remove thumbnails
        f'//span[@class="toc"] |'  # remove tocs
        f'//span[@role="note"] |'  # remove notes
        f'//table[not(@class="wikitable")] |'  # remove non-wikitables
        f'//ul[contains(@class, "gallery ")] |'  # remove galleries
        f'//span[@class="mw-editsection"] |'  # remove [edit]
        f'//sup[@class="reference"] |'  # remove references, e.g. [43]
        # remove general no prints
        f'//sup[@class="noprint Inline-Template"] |'
        # remove [update]
        f'//sup[@class="plainlinks noexcerpt noprint asof-tag update"] |'
        # remove [citation needed]
        f'//sup[@class="noprint Inline-Template Template-Fact"]'
    ):
        elem.drop_tree()

    # replace text of math tag by alttext
    for elem in tree.xpath(f"//math"):
        elem.text = elem.attrib["alttext"]

    # replace root span tag by html tag
    for elem in tree.xpath(f'//span[@class="mw-parser-output"]'):
        elem.tag = "html"

    # replace dt by p tag
    for elem in tree.xpath(f"//dt"):
        elem.tag = "p"

    # drop all tags (keep content) that are not allowed
    for elem in tree.iter():
        if elem.tag not in allowed_tags:
            elem.drop_tag()

    # remove See also, references, notes, etc. sections
    remove_remaining_tags = False
    for elem in tree:
        if remove_remaining_tags:
            elem.drop_tree()
        elif elem.tag in ["h1", "h2", "h3", "h4", "h5", "h6"] and elem.text_content().lower() in [
            "see also",
            "notes",
            "references",
            "footnotes",
            "citations",
            "sources",
            "external links",
            "further reading",
            "general references",
            "notes, references, sources",
            "notes and references",
            "works cited",
            "bibliography",
        ]:
            remove_remaining_tags = True
            elem.drop_tree()

    return tree


def find_all(text: str, substring: str) -> List[int]:
    indices = []
    index = text.find(substring)
    l = len(substring)
    while index != -1:
        indices.append(index)
        index = text.find(substring, index + l)
    return indices


def parse_html_to_segments(
    cleaned_tree: lh.HtmlElement,
    parse_tables: bool,
) -> list[dict]:
    """Parse the cleaned html tree into a json schema (list of segments)"""

    # Initialize Segments
    segments = []

    # Initialize Counters
    segment_id = 0

    parse_lists = False
    parse_paragraphs = True
    parse_other_tags = False

    for elem in cleaned_tree:
        if elem.tag == "table" and parse_tables == True:
            # get number of table columns
            table_str = lh.tostring(elem, encoding=str)
            segments.append({"segment_id": segment_id, "tag": "table", "value": handle_table_data(table_str)})
            segment_id += 1
        elif elem.tag in ["ul", "ol"] and parse_lists == True:
            for list_elem in elem.getchildren():
                elem_text = " ".join(list_elem.text_content().strip().split())

                # Add them to the
                segments.append(
                    {
                        "segment_id": segment_id,
                        "value": elem_text,
                        "tag": list_elem.tag,
                    }
                )
                segment_id += 1
        elif elem.tag == "p" and parse_paragraphs is True:
            elem_text = " ".join(elem.text_content().strip().split())

            segments.append(
                {
                    "segment_id": segment_id,
                    "value": elem_text,
                    "tag": elem.tag,
                }
            )
            segment_id += 1
        elif parse_other_tags:
            elem_text = " ".join(elem.text_content().strip().split())

            segments.append(
                {
                    "segment_id": segment_id,
                    "value": elem_text,
                    "tag": elem.tag,
                }
            )
            segment_id += 1

    # First value is False because this is not a redirect page
    return segments


def parse_wiki_html(
    line: str,
    parse_tables: bool,
) -> Optional[Dict]:
    """Parse a single wikipedia article."""

    article_raw = json.loads(line)

    # Fill in the Wikidata Item ID for this article
    article_wikidata_id = None
    for prop in article_raw["properties"]:
        if prop["name"] == "wikibase_item":
            article_wikidata_id = prop["*"]

    # Clean HTML Tree
    cleaned_tree = clean_wiki_html(article_raw["text"])

    # Get individually parsed segments
    segments = parse_html_to_segments(
        cleaned_tree,
        parse_tables,
    )

    # Build a parsed article object and return it
    article_parsed = {
        "title": article_raw["title"],
        "id_": article_raw["pageid"],
        "wikibase_item_id": article_wikidata_id,
        "segments": segments,
    }

    return article_parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # argument for input wikipedia dump
    parser.add_argument(
        "--html-dump",
        default="enwiki_html_articles_full.jsonl.gz",
        type=str,
        help="Path to HTML Wikipedia Dump",
    )

    # argument for output parsed wikipedia dump
    parser.add_argument(
        "--out",
        default="enwiki_parsed_full.jsonl.gz",
        type=str,
        help="Path to output Parsed Wikipedia Dump",
    )

    # argument for extracting tables.
    parser.add_argument("--parse_tables", default=True, type=bool, help="Extract tables from scanning all text")

    return parser.parse_args()


def write_jsonl_gzip(line: Optional[dict], file: IO, pbar_success: std.tqdm, pbar_total: Optional[std.tqdm] = None):
    """
    Writes the data to a zip file
    Args:
        line: dict containing the data content
        file: file object for target file
        pbar_success: update the progress bar
    """
    if pbar_total is not None:
        pbar_total.update()
    if line is not None:
        file.write(f"{json.dumps(line)}\n")
        pbar_success.update()


def main():
    args = parse_args()

    # Get arguments
    wikipedia_dump_path = args.html - dump
    parsed_dump_path = args.out
    parse_tables = args.parse_tables

    print()
    print(f"Parsing Wikipedia dump located at : {wikipedia_dump_path}")
    print(f"Saving parsed dump at : {parsed_dump_path}")

    # Start Parsing
    print()
    print(datetime.datetime.now(), "---", "Parsing Started")

    with gzip.open(wikipedia_dump_path, "rb") as data_raw, gzip.open(parsed_dump_path, "wt") as data_parsed:
        with tqdm(desc="Successfully written") as pbar_success:
            callback_fn = partial(write_jsonl_gzip, file=data_parsed, pbar_success=pbar_success)

            with mp.Pool() as pool:

                for idx, line in enumerate(data_raw):
                    pool.apply_async(
                        parse_wiki_html,
                        args=(line, parse_tables),
                        callback=callback_fn,
                    )

                pool.close()
                pool.join()

    print(datetime.datetime.now(), "---", "Parsing Complete")


if __name__ == "__main__":
    main()
