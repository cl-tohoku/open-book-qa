import argparse
import logging
from collections import Counter

from elasticsearch import Elasticsearch
from tqdm import tqdm

import allennlp_models.rc.models.transformer_qa
from allennlp.common.checks import check_for_gpu
from allennlp.predictors import Predictor


logger = logging.getLogger(__name__)


class Retriever(object):
    def retrieve_top_k(self, question: str, k: int):
        raise NotImplementedError


class ESRetriever(Retriever):
    def __init__(self, index_name: str, host: str = "localhost", port: int = 9200):
        self.es = Elasticsearch(host=host, port=port)
        self.index_name = index_name

    def retrieve_top_k(self, question: str, k: int):
        query = {"match": {"text": question}}
        result = self.es.search(index=args.es_index_name, body=dict(query=query), size=k)
        return [{"title": hit["_source"]["title"], "text": hit["_source"]["text"]} for hit in result["hits"]["hits"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-file", type=str, required=True,
                        help="Path to the AllenNLP model file (model.tar.gz)")
    parser.add_argument("--es-index-name", type=str, required=True,
                        help="Elasticsearch index name")
    parser.add_argument("--retriever-k", type=int, default=100,
                        help="Number of the documents to be retrieved")
    parser.add_argument("--cuda-device", type=int, default=-1)
    args = parser.parse_args()

    check_for_gpu(args.cuda_device)

    overwriting_dict = {
        "dataset_reader": {
            "transformer_model_name": "cl-tohoku/bert-base-japanese-v2"
        },
        "model": {
            "transformer_model_name": "cl-tohoku/bert-base-japanese-v2"
        }
    }

    retriever = ESRetriever(args.es_index_name)
    predictor = Predictor.from_path(args.model_file, predictor_name="transformer_qa", cuda_device=args.cuda_device,
                                    overrides=overwriting_dict)

    print("")
    print("==============================")
    print("Welcome to Open-Book QA Solver")
    print("==============================")
    print("")
    while(1):
        question = input("Question: ")
        if question == "exit":
            break

        documents = retriever.retrieve_top_k(question, k=args.retriever_k)
        counter = Counter()

        for document in tqdm(documents):
            output = predictor.predict_json({"context": document["text"], "question": question})
            pred_answer = output["best_span_str"]
            if pred_answer:
                counter.update([pred_answer])

        counter["(unanswerable)"] = 0

        print("Predicted Answer:", counter.most_common(1)[0][0])
        print("Scores:", counter.most_common())
        print("")
