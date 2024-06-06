import argparse
import bz2
import json

from lxml import etree as et
from tqdm import tqdm


def extract_pages(wiki_dump_path: str):
    pbar_1 = tqdm(total=347057554)
    pbar_2 = tqdm(leave=False)
    pages = {}
    with bz2.BZ2File(wiki_dump_path) as xml_file:
        parser = et.iterparse(xml_file, events=("end",))
        elems = (elem for _, elem in parser)

        elem = next(elems)
        namespace = elem.tag.rsplit("}", maxsplit=1)[0][1:]
        page_tag = f"{{{namespace}}}page"
        text_path = f"./{{{namespace}}}revision/{{{namespace}}}text"
        title_path = f"./{{{namespace}}}title"
        ns_path = f"./{{{namespace}}}ns"
        pageid_path = f"./{{{namespace}}}id"
        for elem in elems:
            pbar_1.update()
            if elem.tag == page_tag:

                title = elem.find(title_path).text
                text = elem.find(text_path).text
                pageid = elem.find(pageid_path).text
                ns = elem.find(ns_path).text
                if (
                    ns != "0"  # exclude everything but main articles
                    or text[:9].lower() == "#REDIRECT".lower()  # exclude redirects
                ):
                    elem.clear()
                    continue

                pages[pageid] = title
                pbar_2.update()

                elem.clear()
    pbar_1.close()
    pbar_2.close()

    return pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # argument for input wikipedia dump
    parser.add_argument(
        "--dump",
        type=str,
        help="Path to Wikipedia Dump e.g. enwiki-latest-pages-articles.xml.bz2",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="Path to output directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    wiki_dump_path = args.dump
    wiki_page_ids_path = args.out

    pages = extract_pages(wiki_dump_path)
    json.dump(pages, open(wiki_page_ids_path, "w"))
    print(len(pages))
    print(list(pages.values())[:5])


if __name__ == "__main__":
    main()
