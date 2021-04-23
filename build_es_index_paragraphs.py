"""
Script for building an Elasticsearch index of Wikipedia paragraphs.
"""

import argparse
import gzip
import json

from elasticsearch import Elasticsearch
from logzero import logger
from tqdm import tqdm


def create_index(es: Elasticsearch, index_name: str):
    es.indices.create(index=index_name, body={
        "mappings": {
            "paragraph": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "paragraph_index": {"type": "integer"},
                    "text": {
                        "type": "text",
                        "analyzer": "kuromoji"
                    }
                }
            }
        }
    })


def index_paragraphs(es: Elasticsearch, input_file: str, index_name: str, append_title_to_text: bool = False):
    with gzip.open(input_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            title = item["title"]
            text = item["text"]
            if append_title_to_text:
                text = title + ": " + text

            es.index(index=index_name, doc_type="paragraph", body={
                "id": item["id"],
                "title": title,
                "section": item["section"],
                "paragraph_index": item["paragraph_index"],
                "text": text
            })


def main(args):
    es = Elasticsearch(hosts=[{"host": args.hostname, "port": args.port}], timeout=60)

    logger.info("Creating an Elasticsearch index")
    create_index(es, args.index_name)

    logger.info("inserting documents")
    index_paragraphs(es, args.input_file, args.index_name, args.append_title_to_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an Elasticsearch index of Wikipedia paragraphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input Wikipedia paragraphs file.")
    parser.add_argument("--index_name", type=str, required=True, help="Elasticsearch index name to create.")
    parser.add_argument("--hostname", type=str, default="localhost", help="Elasticsearch host name.")
    parser.add_argument("--port", type=int, default=9200, help="Elasticsearch port number.")
    parser.add_argument("--append_title_to_text", action="store_true", help="Append Wikipedia page title to text.")
    args = parser.parse_args()
    main(args)
