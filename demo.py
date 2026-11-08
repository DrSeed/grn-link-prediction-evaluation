# Self-contained demo: the same model, four evaluation protocols, and a model that never saw real expression.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grn_eval import simulate_grn, sample_pairs, pair_features, cross_validate, metrics


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    n_tfs = 24
    expr, true, observed = simulate_grn(n_tfs=n_tfs)
    rng = np.random.default_rng(1)
    shuffled = np.column_stack([rng.permutation(c) for c in expr.T])  # every gene shuffled: no biology left

    rows = []
    for neg in ["random", "tf_matched"]:
        tfs, targets, y = sample_pairs(observed, negatives=neg)
        X, corr = pair_features(expr, tfs, targets, n_tfs)
        Xs, _ = pair_features(shuffled, tfs, targets, n_tfs)
        for split in ["pair", "tf"]:
            for name, feats in [("model, real expression", X), ("model, shuffled expression", Xs)]:
                auroc, auprc = metrics(y, cross_validate(feats, y, tfs, split))
                rows.append({"negatives": neg, "split": split, "scorer": name, "AUROC": auroc, "AUPRC": auprc})
            auroc, auprc = metrics(y, corr)
            rows.append({"negatives": neg, "split": split, "scorer": "|correlation| (no training)",
                         "AUROC": auroc, "AUPRC": auprc})
    table = pd.DataFrame(rows).round(3)
    table.to_csv("results/summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    settings = [("random", "pair"), ("tf_matched", "pair"), ("random", "tf"), ("tf_matched", "tf")]
    labels = ["random pairs split\nrandom negatives", "random pairs split\nTF-matched negatives",
              "TF held out\nrandom negatives", "TF held out\nTF-matched negatives"]
    colours = {"model, real expression": "#4c72b0", "model, shuffled expression": "#c44e52",
               "|correlation| (no training)": "#8c8c8c"}
    x = np.arange(len(settings))
    for k, (scorer, c) in enumerate(colours.items()):
        vals = [table.loc[(table.negatives == n) & (table.split == s) & (table.scorer == scorer), "AUROC"].item()
                for n, s in settings]
        axes[0].bar(x + (k - 1) * 0.27, vals, 0.27, color=c, label=scorer)
        for xi, v in zip(x + (k - 1) * 0.27, vals):
            axes[0].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].axhline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_xticks(x, labels, fontsize=9)
    axes[0].set_ylim(0.3, 1.05)
    axes[0].set_ylabel("AUROC (out-of-fold)")
    axes[0].set_title("Red bars should sit at 0.5. When they don't, the benchmark leaks")
    axes[0].legend(fontsize=8, loc="upper right")

    tfs, targets, y = sample_pairs(observed, negatives="random")
    deg = observed.sum(1)
    axes[1].scatter(deg, [y[tfs == t].mean() for t in range(n_tfs)], color="#c44e52", label="random negatives")
    tfs2, _, y2 = sample_pairs(observed, negatives="tf_matched")
    axes[1].scatter(deg, [y2[tfs2 == t].mean() for t in range(n_tfs)], color="#4c72b0", label="TF-matched negatives")
    axes[1].set_xscale("log")
    axes[1].set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    axes[1].minorticks_off()
    axes[1].set_xlabel("TF's number of labelled targets")
    axes[1].set_ylabel("fraction of that TF's pairs labelled positive")
    axes[1].set_title("With random negatives, 'is this a hub TF?'\nalready predicts the label")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
