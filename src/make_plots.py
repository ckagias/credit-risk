import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baselines
import evaluate
from split import DB, load_part

OUT = Path("docs/img")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
MODEL = "#2a78d6"
REFERENCE = "#8a8985"


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def baseline_curves():
    con = sqlite3.connect(DB)
    train = load_part(con, "train", baselines.EXT_SOURCES)
    con.close()
    y = train["TARGET"].to_numpy()
    base_rate = y.mean()
    scores = evaluate.out_of_fold_predictions(baselines.ext_source_model(), train[baselines.EXT_SOURCES], y)
    fpr, tpr = evaluate.roc_curve(y, scores)
    recall, precision = evaluate.pr_curve(y, scores)

    fig, (roc, pr) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    roc.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1.5, label="Base rate for everyone")
    roc.plot(fpr, tpr, color=MODEL, linewidth=2, label="EXT_SOURCE mean, logistic")
    roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    roc.set_aspect("equal")
    roc.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    pr.plot([0, 1], [base_rate, base_rate], color=REFERENCE, linewidth=1.5, label=f"Base rate for everyone ({base_rate:.3f})")
    pr.plot(recall, precision, color=MODEL, linewidth=2, label="EXT_SOURCE mean, logistic")
    pr.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    pr.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    for ax in (roc, pr):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_curves()


if __name__ == "__main__":
    main()
