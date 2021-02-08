import argparse
import json
import logging
from typing import Iterable, List, Set

import allennlp_models.rc.models.transformer_qa
from allennlp.common.checks import check_for_gpu
from allennlp.data import Instance
from allennlp.predictors import Predictor

import allennlp_modules


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prediction for SQuAD 2.0")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cuda-device", type=int, default=-1)
    args = parser.parse_args()

    # Read inputs
    check_for_gpu(args.cuda_device)
    predictor = Predictor.from_path(
        args.model_file, predictor_name="transformer_qa", cuda_device=args.cuda_device
    )
    predictor._dataset_reader.lazy = True
    instances = predictor._dataset_reader.read(args.input_file)

    # We have to make sure we put instances with the same qid all into the same batch.
    def batch_instances_by_qid(instances: Iterable[Instance]) -> Iterable[List[Instance]]:
        current_qid = None
        current_batch = []
        for instance in instances:
            instance_qid = instance["metadata"]["id"]
            if current_qid is None:
                current_qid = instance_qid
            if instance_qid == current_qid:
                current_batch.append(instance)
            else:
                yield current_batch
                current_batch = [instance]
                current_qid = instance_qid
        if len(current_batch) > 0:
            yield current_batch

    def make_batches(
        instances: Iterable[Instance], batch_size: int = 256
    ) -> Iterable[List[Instance]]:
        current_batch: List[Instance] = []
        for qid_instances in batch_instances_by_qid(instances):
            if len(qid_instances) + len(current_batch) < batch_size:
                current_batch.extend(qid_instances)
            else:
                if len(current_batch) > 0:
                    yield current_batch
                current_batch = qid_instances
        if len(current_batch) > 0:
            yield current_batch

    # Run model and evaluate results
    ids_seen: Set[str] = set()
    answers = {}
    for batch in make_batches(instances, batch_size=args.batch_size):
        for result in predictor.predict_batch_instance(batch):
            assert result["id"] not in ids_seen
            ids_seen.add(result["id"])
            answers[result["id"]] = result["best_span_str"]

    # Print results
    logger.info("Printing results to file")
    with open(args.output_file, "w") as fo:
        print(json.dumps(answers, ensure_ascii=False), file=fo)
