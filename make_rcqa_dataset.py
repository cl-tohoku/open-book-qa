import argparse
import gzip
import json
import os
from typing import Dict, List, Optional
from unicodedata import normalize

from elasticsearch import Elasticsearch
from logzero import logger
from tqdm import tqdm


class Retriever(object):
    def retrieve_top_k(self, question: str, k: int):
        raise NotImplementedError


class ESRetriever(Retriever):
    def __init__(self, index_name: str, host: str = "localhost", port: int = 9200):
        self.es = Elasticsearch(host=host, port=port)
        self.index_name = index_name

    def retrieve_top_k(self, question: str, k: int):
        query = {"match": {"text": question}}
        result = self.es.search(index=self.index_name, body=dict(query=query), size=k)
        return [{"title": hit["_source"]["title"], "text": hit["_source"]["text"]} for hit in result["hits"]["hits"]]


def normalize_text(text: str):
    text = normalize("NFKC", text)
    text = " ".join(text.strip().split())
    text = text.strip()
    return text


def load_and_split_dataset(dataset_file: str):
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    with gzip.open(dataset_file, "rt") as f:
        for line in tqdm(f):
            item = json.loads(line)

            if item["timestamp"] < "2009":
                train_dataset.append(item)
            elif "2009" <= item["timestamp"] < "2010":
                dev_dataset.append(item)
            elif item["timestamp"] >= "2010":
                test_dataset.append(item)
            else:
                logger.warning("Skipping qid %s: timestamp is out of scope.", item["qid"])
                continue

    return train_dataset, dev_dataset, test_dataset


def align_answer_to_context(answer: str, context: str):
    """align the original answer to the context by ignoring whitespaces"""

    answer_nospace = "".join(c for c in normalize_text(answer) if not c.isspace())
    context_nospace, offsets = zip(*[(c, i) for i, c in enumerate(context) if not c.isspace()])
    answer_start_nospace = "".join(context_nospace).find(answer_nospace)

    if answer_start_nospace == -1:
        return -1, -1

    answer_start = offsets[answer_start_nospace]
    answer_end = offsets[answer_start_nospace + len(answer_nospace) - 1] + 1

    return answer_start, answer_end


def is_answer_in_context(answer: str, context: str):
    answer_start, _ = align_answer_to_context(answer, context)
    return answer_start != -1


def process_dataset(
    dataset: List[Dict],
    answerability_threshold: int = 2,
    answerable_only: bool = False,
    mode: str = "original",
    retriever: Optional[Retriever] = None,
    retriever_k: int = 100,
) -> List[Dict]:
    assert mode in ("original", "open_domain", "hard_unanswerable")
    assert mode == "original" or retriever is not None

    processed_dataset = []
    num_answerable = 0
    for item in tqdm(dataset):
        qid = item["qid"]
        question = normalize_text(item["question"])
        answer = normalize_text(item["answer"])
        documents = item["documents"]

        retrieved_documents = None
        if retriever is not None:
            retrieved_documents = retriever.retrieve_top_k(question, k=retriever_k)

        if mode == "open_domain":
            assert retrieved_documents is not None
            documents = retrieved_documents

        for i, document in enumerate(documents):
            qd_id = "{}-{:04d}".format(qid, i + 1)
            context = normalize_text(document["text"])
            if mode == "open_domain":
                # answerability is determined by whether the answer in the context
                is_answerable = is_answer_in_context(answer, context)
            else:
                # answerability is determined by the annotated answerability score
                is_answerable = document["score"] >= answerability_threshold

            if mode == "original" and answerable_only and not is_answerable:
                continue

            if mode == "hard_unanswerable" and not is_answerable:
                # replace an original unanswerable item with a retrieved unanswerable item

                retrieved_context = None
                is_retrieved_answerable = True

                # pop a retrieved item until the popped item is unanswerable
                while is_retrieved_answerable:
                    retrieved_document = retrieved_documents.pop(0)
                    retrieved_context = normalize_text(retrieved_document["text"])
                    answer_start, answer_end = align_answer_to_context(answer, retrieved_context)
                    is_retrieved_answerable = answer_start != -1

                assert not is_retrieved_answerable
                assert retrieved_context is not None
                context = retrieved_context

            if mode in ("original", "hard_unanswerable") and is_answerable:
                answer_start, answer_end = align_answer_to_context(answer, context)
                answer_item = [{
                    "text": context[answer_start:answer_end],  # aligned to the context
                    "answer_start": answer_start,
                    "answer_end": answer_end
                }]
            else:
                answer_item = []

            output_item = {
                "title": qd_id,
                "paragraphs": [{
                    "context": context,
                    "qas": [{
                        "id": qd_id,
                        "question": question,
                        "answers": answer_item,
                        "original_answer": answer,
                        "is_impossible": not is_answerable
                    }]
                }]
            }
            if "score" in document:
                output_item["paragraphs"][0]["qas"][0]["score"] = document["score"]
            if is_answerable:
                num_answerable += 1

            processed_dataset.append(output_item)

    num_total = len(processed_dataset)
    num_not_answerable = num_total - num_answerable
    logger.info("Answerable: %s, Not answerable: %s, Total: %s", num_answerable, num_not_answerable, num_total)

    return processed_dataset


def main(args):
    if args.es_index_name is not None:
        logger.info("Initializing a retriever")
        retriever = ESRetriever(args.es_index_name, host=args.es_host, port=args.es_port)
    else:
        retriever = None

    logger.info("Loading datasets")
    train_dataset, dev_dataset, test_dataset = load_and_split_dataset(args.dataset_file)
    logger.info("Train dataset: %s questions", len(train_dataset))
    logger.info("Dev dataset: %s questions", len(dev_dataset))
    logger.info("Test dataset: %s questions", len(test_dataset))

    if args.skip_train:
        logger.info("Skipping the training dataset")
    else:
        logger.info("Processing training datasets")
        train_dataset_processed = process_dataset(
            train_dataset,
            answerability_threshold=args.answerability_threshold,
            answerable_only=args.answerable_only,
            mode=args.mode,
            retriever=retriever,
            retriever_k=args.retriever_k
        )
        json.dump({"data": train_dataset_processed, "version": "v2.0"},
                  open(os.path.join(args.output_dir, "train-v2.0.json"), "w"),
                  ensure_ascii=False,
                  indent=4)

    logger.info("Processing the development dataset")
    dev_dataset_processed = process_dataset(
        dev_dataset,
        answerability_threshold=args.answerability_threshold,
        answerable_only=args.answerable_only,
        mode=args.mode,
        retriever=retriever,
        retriever_k=args.retriever_k
    )
    json.dump({"data": dev_dataset_processed, "version": "v2.0"},
              open(os.path.join(args.output_dir, "dev-v2.0.json"), "w"),
              ensure_ascii=False,
              indent=4)

    logger.info("Processing the test dataset")
    test_dataset_processed = process_dataset(
        test_dataset,
        answerability_threshold=args.answerability_threshold,
        answerable_only=args.answerable_only,
        mode=args.mode,
        retriever=retriever,
        retriever_k=args.retriever_k
    )
    json.dump({"data": test_dataset_processed, "version": "v2.0"},
              open(os.path.join(args.output_dir, "test-v2.0.json"), "w"),
              ensure_ascii=False,
              indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", choices=("original", "open_domain", "hard_unanswerable"), required=True)
    parser.add_argument("--es_index_name", type=str)
    parser.add_argument("--es_host", type=str, default="localhost")
    parser.add_argument("--es_port", type=int, default=9200)
    parser.add_argument("--retriever_k", type=int, default=100)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--answerability_threshold", type=int, default=2)
    parser.add_argument("--answerable_only", action="store_true")
    args = parser.parse_args()
    main(args)
