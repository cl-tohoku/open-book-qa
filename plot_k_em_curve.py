import argparse

import matplotlib.pyplot as plt
import numpy as np

def load_metrics_dataset(dataset_path):
    ks = []
    em_values = []
    f1_values = []
    upper_bounds = []
    with open(dataset_path) as f:
        f.readline()
        for line in f:
            k, em, f1, upper_bound = line.rstrip("\n").split("\t")
            ks.append(int(k))
            em_values.append(float(em))
            f1_values.append(float(f1))
            upper_bounds.append(float(upper_bound))

    ks = np.array(ks)
    em_values = np.array(em_values) * 100
    f1_values = np.array(f1_values) * 100
    upper_bounds = np.array(upper_bounds) * 100

    return ks, em_values, f1_values, upper_bounds


def main(args):
    ks, em_values_all_answerable, _, upper_bounds = load_metrics_dataset(args.all_answerable_file)
    _, em_values_answerable_only, _, _ = load_metrics_dataset(args.answerable_only_file)
    _, em_values_soft_answerability, _, _ = load_metrics_dataset(args.soft_answerability_file)
    _, em_values_hard_answerability, _, _ = load_metrics_dataset(args.hard_answerability_file)

    fig = plt.figure(figsize=(5.5, 3.5))
    ax = fig.add_subplot(111, xlabel="$k$ (Number of retrieved documents)", ylabel="EM")
    ax.set_xscale("log")
    xticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.minorticks_off()
    ax.grid(axis="y", alpha=0.8, linestyle="--", linewidth=1)
    ax.plot(ks, em_values_answerable_only, color="C0", marker="^", markersize=4, markevery=xticks-1, label="AnswerableOnly")
    ax.plot(ks, em_values_soft_answerability, color="C3", marker="o", markersize=4, markevery=xticks-1, label="SoftAnswerability")
    ax.plot(ks, em_values_all_answerable, color="C2", marker="D", markersize=4, markevery=xticks-1, label="AllAnswerable")
    ax.plot(ks, em_values_hard_answerability, color="C1", marker="s", markersize=4, markevery=xticks-1, label="HardAnswerability")
    ax.plot(ks, upper_bounds, color="gray", linestyle="dotted", label="upper bound")
    ax.legend(loc="best", fontsize=10)
    plt.savefig(args.output_file, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_answerable_file", type=str, required=True)
    parser.add_argument("--answerable_only_file", type=str, required=True)
    parser.add_argument("--soft_answerability_file", type=str, required=True)
    parser.add_argument("--hard_answerability_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
