import argparse
import json
import string

from logzero import logger


def normalize_answer(answer):
    def compress_spaces(text):
        return "".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return compress_spaces(remove_punc(lower(answer)))


def main(args):
    with open(args.dataset_file) as f:
        dataset = json.load(f)["data"]

    prediction_dicts = []
    for prediction_file in args.prediction_files:
        with open(prediction_file) as f:
            predcition = json.load(f)
            prediction_dicts.append(predcition)

    output_items = []
    for article in dataset:
        for p in article["paragraphs"]:
            context = p["context"]
            for qa in p["qas"]:
                qid = qa["id"]
                question = qa["question"]
                is_impossible = qa.get("is_impossible", False)
                score = qa.get("score")
                if is_impossible:
                    gold_answers = [""]
                else:
                    gold_answers = [a["text"] for a in qa["answers"]
                                    if normalize_answer(a["text"])]

                pred_answers = []
                pred_results = []
                for i, prediction in enumerate(prediction_dicts):
                    if qid not in prediction:
                        logger.warning(f"Missing prediction ({i}) for {qid}: this example will be skipped")
                        continue

                    pred_answer = prediction[qid]
                    pred_result = any([
                        normalize_answer(pred_answer) == normalize_answer(gold_answer)
                        for gold_answer in gold_answers
                    ])
                    pred_answers.append(pred_answer)
                    pred_results.append(pred_result)

                if len(pred_answers) != len(prediction_dicts):
                        continue

                output_item = {
                    "id": qid,
                    "question": question,
                    "context": context,
                    "score": score,
                    "gold_answers": gold_answers,
                    "pred_answers": pred_answers,
                    "pred_results": pred_results
                }
                output_items.append(output_item)

    with open(args.output_file, "w") as fo:
        for output_item in output_items:
            print(json.dumps(output_item, ensure_ascii=False), sep="\t", file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script for RCQA dataset.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--prediction_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
