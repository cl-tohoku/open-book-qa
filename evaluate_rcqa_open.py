import argparse
import collections
import json
import string
from collections import defaultdict, Counter

from logzero import logger
from tqdm import trange


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


def group_dataset_and_prediction_by_qid(dataset, prediction, top_k=10):
    qid_to_question = {}
    qid_to_gold_answer = {}
    qid_to_pred_answers = defaultdict(list)
    qid_to_is_impossible = defaultdict(list)

    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qaid = qa["id"]
                if qaid not in prediction:
                    print("Missing prediction for %s" % qaid)
                    continue

                qid, doc_rank = qaid.split("-")
                if int(doc_rank) > top_k:
                    continue

                question = qa["question"]
                assert qid_to_question.get(qid) in (None, question)
                qid_to_question[qid] = question

                gold_answer = qa["original_answer"]
                assert qid_to_gold_answer.get(qid) in (None, gold_answer)
                qid_to_gold_answer[qid] = gold_answer

                pred_answer = prediction[qaid]
                qid_to_pred_answers[qid].append(pred_answer)

                is_impossible = qa["is_impossible"]
                qid_to_is_impossible[qid].append(is_impossible)

    return qid_to_question, qid_to_gold_answer, qid_to_pred_answers, qid_to_is_impossible


def get_qa_scores(qid_to_gold_answer, qid_to_pred_answers, qid_to_is_impossible, top_k):
    exact_scores = {}
    f1_scores = {}
    not_impossible_qids = set()
    for qid, pred_answers in qid_to_pred_answers.items():
        pred_answers = pred_answers[:top_k]
        answer_counts = Counter(pred_answers)
        answer_counts[""] = 0
        most_common_answer = answer_counts.most_common()[0][0]

        gold_answer = qid_to_gold_answer[qid]

        exact_score = compute_exact(gold_answer, most_common_answer)
        f1_score = compute_f1(gold_answer, most_common_answer)

        is_impossible = all(qid_to_is_impossible[qid][:top_k])
        if not is_impossible:
            not_impossible_qids.add(qid)

        exact_scores[qid] = exact_score
        f1_scores[qid] = f1_score

    total = len(qid_to_pred_answers)
    assert len(exact_scores) == total, len(exact_scores)
    assert len(f1_scores) == total, len(f1_scores)

    averaged_em = sum(exact_scores.values()) / total
    averaged_f1 = sum(f1_scores.values()) / total
    upper_bound = len(not_impossible_qids) / total
    return averaged_em, averaged_f1, upper_bound


def main(args):
    logger.info("Loading the dataset file")
    with open(args.dataset_file) as f:
        dataset = json.load(f)["data"]

    logger.info("Loading the prediction file")
    with open(args.prediction_file) as f:
        predcition = json.load(f)

    logger.info("Processing the loaded files")
    qid_to_question, qid_to_gold_answer, qid_to_pred_answers, qid_to_is_impossible = \
        group_dataset_and_prediction_by_qid(dataset, predcition, top_k=args.top_k)

    logger.info("Computing EM and F1")
    with open(args.output_file, "w") as fo:
        print("k", "EM", "F1", "upper bound", sep="\t", file=fo)
        best_em = 0.0
        f1_for_best_em = 0.0
        k_for_best_em = None
        if args.fix_k:
            ks = [args.top_k]
        else:
            ks = trange(1, args.top_k + 1)

        for k in ks:
            em, f1, upper_bound = get_qa_scores(qid_to_gold_answer, qid_to_pred_answers, qid_to_is_impossible, top_k=k)
            print(f"{k}\t{em:.4f}\t{f1:.4f}\t{upper_bound:.4f}", sep="\t", file=fo)
            if em > best_em:
                best_em = em
                f1_for_best_em = f1
                k_for_best_em = k

        assert k_for_best_em is not None
        logger.info("Best EM: %.3f (F1 = %.3f, k = %d)", best_em, f1_for_best_em, k_for_best_em)

    if args.output_qa_prediction_file is not None:
        logger.info("Writing predictions to file")
        qa_predictions = []
        for qid in qid_to_question:
            qa_prediction = {
                "question_id": qid,
                "question": qid_to_question[qid],
                "gold_answer": qid_to_gold_answer[qid],
                "pred_answers": qid_to_pred_answers[qid]
            }
            qa_predictions.append(qa_prediction)
        json.dump(qa_predictions, open(args.output_qa_prediction_file, "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script for open-domain QA using RCQA dataset.")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_qa_prediction_file", type=str)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--fix_k", action="store_true")
    args = parser.parse_args()
    main(args)
