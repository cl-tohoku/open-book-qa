import argparse
import json
import logging
from typing import Set

import allennlp_models.rc.models.bidaf
from allennlp.common.checks import check_for_gpu
from allennlp.predictors import Predictor
from tqdm import tqdm

import allennlp_modules


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prediction for SQuAD 2.0")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--cuda-device", type=int, default=-1)
    args = parser.parse_args()

    # Read inputs
    check_for_gpu(args.cuda_device)
    predictor = Predictor.from_path(
        args.model_file, predictor_name="reading_comprehension", cuda_device=args.cuda_device
    )
    instances = predictor._dataset_reader.read(args.input_file)
    logger.info("Running on %d instances", len(instances))

    # Run model and evaluate results
    ids_seen: Set[str] = set()
    answers = {}
    for instance in tqdm(instances, desc="Evaluating instances"):
        qid = instance["metadata"]["id"]
        result = predictor.predict_instance(instance)
        assert qid not in ids_seen
        ids_seen.add(qid)
        answers[qid] = result["best_span_str"]

    # Print results
    logger.info("Printing results to file")
    with open(args.output_file, "w") as fo:
        print(json.dumps(answers, ensure_ascii=False), file=fo)
