import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_evaluation_file(fpath):
    with open(fpath) as f:
        eval_dict = json.load(f)

    accuracies = [eval_dict[f"acc_{i}"] for i in range(0, 6)]
    return accuracies


def main(args):
    xs = np.array([2, 3, 4, 5])
    width = 0.2
    all_answerable_accs = load_evaluation_file(args.all_answerable_file)[2:]
    answerable_only_accs = load_evaluation_file(args.answerable_only_file)[2:]
    soft_answerability_accs = load_evaluation_file(args.soft_answerability_file)[2:]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, xlabel="$s$ (answerability score)", ylabel="EM")
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.8, linestyle="--", linewidth=1)
    ax.bar(xs - width - 0.05, answerable_only_accs, width, color="C0", label="AnswerableOnly")
    ax.bar(xs, all_answerable_accs, width, color="C2", label="AllAnswerable")
    ax.bar(xs + width + 0.05, soft_answerability_accs, width, color="C3", label="SoftAnswerability")
    ax.legend(loc="lower right")
    plt.savefig(args.output_file, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_answerable_file", type=str, required=True)
    parser.add_argument("--answerable_only_file", type=str, required=True)
    parser.add_argument("--soft_answerability_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
