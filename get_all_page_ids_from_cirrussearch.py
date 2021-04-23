import argparse
import gzip
import json

from tqdm import tqdm


def main(args):
    with gzip.open(args.cirrus_file, "rt") as f, open(args.output_file, "w") as fo:
        title = None
        page_id = None
        rev_id = None
        for line in tqdm(f):
            item = json.loads(line)
            if "index" in item:
                page_id = item["index"]["_id"]
            else:
                assert page_id is not None
                title = item["title"]
                rev_id = item["version"]
                output_item = {
                    "title": title,
                    "pageid": page_id,
                    "revid": rev_id
                }
                print(json.dumps(output_item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get all Wikipedia article page IDs from a Cirrussearch dump file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cirrus_file", type=str, required=True, help="Wikipedia Cirrussearch dump file.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file.")
    args = parser.parse_args()
    main(args)
