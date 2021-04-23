import argparse
import gzip
import json
from unicodedata import normalize

from bs4 import BeautifulSoup
from tqdm import tqdm


SECTIONS_TO_IGNORE = ["脚注", "出典", "参考文献", "関連項目", "外部リンク"]
TAGS_TO_REMOVE = ["table"]
TAGS_TO_EXTRACT = ["p"]
# TAGS_TO_EXTRACT = ["p", "li"]
INNER_TAGS_TO_REMOVE = ["sup"]


def normalize_text(text):
    text = normalize("NFKC", text)
    text = " ".join(text.split())
    text = "".join(char for char in text if char.isprintable())
    text = text.strip()
    return text


def extract_paragraphs_from_html(html):
    soup = BeautifulSoup(html, features="lxml")
    section_title = "__LEAD__"
    section = soup.find(["section"])
    while section:
        if section.h2 is not None:
            section_title = section.h2.text

        for tag in section.find_all(TAGS_TO_REMOVE):
            tag.clear()

        for tag in section.find_all(TAGS_TO_EXTRACT):
            for inner_tag in tag.find_all(INNER_TAGS_TO_REMOVE):
                inner_tag.clear()

            text = normalize_text(tag.text)
            yield (section_title, text)

        section = section.find_next_sibling(["section"])


def main(args):
    with open(args.input_file, "rt") as f, open(args.output_file, "wt") as fo:
        for line in tqdm(f):
            input_item = json.loads(line.rstrip("\n"))
            rev_id = input_item["revid"]
            title = input_item["title"]
            html = input_item["html"]

            paragraph_index = 0
            for (section_title, text) in extract_paragraphs_from_html(html):
                if section_title in SECTIONS_TO_IGNORE:
                    continue
                if len(text) < args.min_text_length:
                    continue
                if len(text) > args.max_text_length:
                    continue

                output_item = {
                    "id": "{}-{}".format(rev_id, paragraph_index),
                    "title": title,
                    "section": section_title,
                    "paragraph_index": paragraph_index,
                    "text": text
                }
                print(json.dumps(output_item, ensure_ascii=False), file=fo)
                paragraph_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract paragraph texts from fetched HTMLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input HTML dump file. (.json.gz)")
    parser.add_argument("--output_file", type=str, required=True, help="Output file (.json.gz)")
    parser.add_argument("--min_text_length", type=int, default=20, help="Minimum text length.")
    parser.add_argument("--max_text_length", type=int, default=1000, help="Maximum text length.")
    args = parser.parse_args()
    main(args)
