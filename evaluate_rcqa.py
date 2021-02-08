import argparse
import collections
import json
import string
from collections import OrderedDict


def normalize_answer(answer):
    def compress_spaces(text):
        return "".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return compress_spaces(remove_punc(lower(answer)))


def compute_exact(a_gold, a_pred):
    a_gold = normalize_answer(a_gold)
    a_pred = normalize_answer(a_pred)
    return int(a_gold == a_pred)


def compute_f1(a_gold, a_pred):
    a_gold = normalize_answer(a_gold)
    a_pred = normalize_answer(a_pred)

    common = collections.Counter(a_gold) & collections.Counter(a_pred)
    num_same = sum(common.values())
    if len(a_gold) == 0 or len(a_pred) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(a_gold == a_pred)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(a_pred)
    recall = 1.0 * num_same / len(a_gold)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_reading_scores(dataset, prediction):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                is_impossible = qa.get("is_impossible", False)
                if is_impossible:
                    continue

                gold_answers = [a["text"] for a in qa["answers"]
                                if normalize_answer(a["text"])]
                assert gold_answers

                if qid not in prediction:
                    print("Missing prediction for %s" % qid)
                    continue

                pred_answer = prediction[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(gold_answer, pred_answer)
                                        for gold_answer in gold_answers)
                f1_scores[qid] = max(compute_f1(gold_answer, pred_answer)
                                     for gold_answer in gold_answers)

    assert len(exact_scores) == len(f1_scores)
    total = len(exact_scores)
    em = sum(exact_scores.values()) / total
    f1 = sum(f1_scores.values()) / total
    return total, em, f1


def get_answerability_scores(dataset, prediction):
    num_true_answerable = 0
    num_true_unanswerable = 0
    num_false_answerable = 0
    num_false_unanswerable = 0
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                is_impossible = qa.get("is_impossible", False)

                if qid not in prediction:
                    print("Missing prediction for %s" % qid)
                    continue

                pred_answer = prediction[qid]
                if pred_answer != "" and not is_impossible:
                    num_true_answerable += 1
                elif pred_answer != "" and is_impossible:
                    num_false_answerable += 1
                elif pred_answer == "" and is_impossible:
                    num_true_unanswerable += 1
                else:
                    assert pred_answer == "" and not is_impossible
                    num_false_unanswerable += 1

    answerable_p = num_true_answerable / (num_true_answerable + num_false_answerable)
    answerable_r = num_true_answerable / (num_true_answerable + num_false_unanswerable)
    answerable_f = 2 * answerable_p * answerable_r / (answerable_p + answerable_r)
    unanswerable_p = num_true_unanswerable / (num_true_unanswerable + num_false_unanswerable)
    unanswerable_r = num_true_unanswerable / (num_true_unanswerable + num_false_answerable)
    unanswerable_f = 2 * unanswerable_p * unanswerable_r / (unanswerable_p + unanswerable_r)

    return answerable_p, answerable_r, answerable_f, unanswerable_p, unanswerable_r, unanswerable_f


def get_accuracies_per_score(dataset, prediction):
    num_examples = [0] * 6
    num_correct = [0] * 6
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                score = qa.get("score")
                if score is None:
                    continue

                is_impossible = qa.get("is_impossible", False)
                gold_answers = [a["text"] for a in qa["answers"]
                                if normalize_answer(a["text"])]
                if is_impossible:
                    gold_answers = [""]

                if qid not in prediction:
                    print("Missing prediction for %s" % qid)
                    continue

                pred_answer = prediction[qid]
                is_correct = max(
                    compute_exact(gold_answer, pred_answer) for gold_answer in gold_answers
                )
                num_correct[score] += is_correct
                num_examples[score] += 1

    accuracies_per_score = [num_correct[i] / max(1, num_examples[i]) for i in range(6)]
    return accuracies_per_score


def main(args):
    with open(args.dataset_file) as f:
        dataset = json.load(f)["data"]

    with open(args.prediction_file) as f:
        predcition = json.load(f)

    output_dict = OrderedDict()

    # reading scores
    total, em, f1 = get_reading_scores(dataset, predcition)
    print(f"Total: {total}")
    print(f"Reading Exact Match: {em:.1%}")
    print(f"Reading Character-level F1: {f1:.1%}")
    output_dict.update({"total": total, "em": em, "f1": f1})

    # answerability scores
    if args.output_answerability_scores:
        answerable_p, answerable_r, answerable_f, unanswerable_p, unanswerable_r, unanswerable_f = \
            get_answerability_scores(dataset, predcition)
        print(f"Answerable Binary Precision: {answerable_p:.1%}")
        print(f"Answerable Binary Recall: {answerable_r:.1%}")
        print(f"Answerable Binary F1: {answerable_f:.1%}")
        print(f"Unanswerable Binary Precision: {unanswerable_p:.1%}")
        print(f"Unanswerable Binary Recall: {unanswerable_r:.1%}")
        print(f"Unanswerable Binary F1: {unanswerable_f:.1%}")
        output_dict.update({
            "answerable_p": answerable_p, "answerable_r": answerable_r, "answerable_f": answerable_f,
            "unanswerable_p": unanswerable_p, "unanswerable_r": unanswerable_r, "unanswerable_f": unanswerable_f}
        )

    # accuracies per score
    if args.output_accuracies_per_score:
        accuracies_per_score = get_accuracies_per_score(dataset, predcition)
        for i, acc in enumerate(accuracies_per_score):
            print(f"Accuracy for Score {i}: {acc:.1%}")
            output_dict.update({f"acc_{i}": acc})

    with open(args.output_file, "w") as fo:
        json.dump(output_dict, fo, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script for RCQA dataset.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--output_answerability_scores", action="store_true")
    parser.add_argument("--output_accuracies_per_score", action="store_true")
    args = parser.parse_args()
    main(args)
