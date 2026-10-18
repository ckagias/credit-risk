import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PATH_LINES = 6


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


def best_c(curve, penalty):
    validation = curve[(curve["penalty"] == penalty) & (curve["split"] == "validation")]
    means = validation.groupby("C")["pr_auc"].mean()
    return means.idxmax(), means.max()


def validation_curve_plot():
    curve = pd.read_csv("results/regularization_curve.csv")
    floor = pd.read_csv("results/baselines.csv").query("metric == 'pr_auc' and model.str.startswith('EXT_SOURCE')")["mean"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, penalty in zip(axes, ("L2", "L1")):
        for split, color, label in (("train", REFERENCE, "Training folds"), ("validation", MODEL, "Validation fold")):
            part = curve[(curve["penalty"] == penalty) & (curve["split"] == split)]
            stats = part.groupby("C")["pr_auc"].agg(["mean", "std"])
            ax.plot(stats.index, stats["mean"], color=color, linewidth=2 if split == "validation" else 1.5, label=label)
            if split == "validation":
                ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=color, alpha=0.15, linewidth=0)
        c, value = best_c(curve, penalty)
        ax.plot([c], [value], marker="o", markersize=8, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"best C = {c:g}\nPR-AUC {value:.4f}", (c, value), textcoords="offset points", xytext=(-52 if penalty == "L1" else 0, -40), ha="center", fontsize=9, color=INK)
        ax.axhline(floor, color=REFERENCE, linewidth=1, alpha=0.6)
        ax.text(0.985, floor + 0.0007, "EXT_SOURCE baseline", transform=ax.get_yaxis_transform(), ha="right", fontsize=8.5, color=INK_SECONDARY)
        lowest = curve[(curve["penalty"] == penalty) & (curve["split"] == "validation")].groupby("C")["pr_auc"].mean().min()
        if lowest < 0.19:
            ax.text(0.985, 0.15, "C = 0.0001 is off the chart:\nevery score the same, PR-AUC 0.081", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_SECONDARY)
        ax.set_xscale("log")
        ax.set_ylim(0.185, 0.235)
        ax.set(xlabel="C (larger = weaker penalty)", title=f"{penalty} penalty")
        style_axis(ax)
    axes[0].set_ylabel("PR-AUC (5-fold mean, band = 1 std)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / "validation_curve.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def regularization_path_plot():
    path = pd.read_csv("results/coefficient_path.csv")
    top = (
        path[(path["penalty"] == "L2") & (path["C"] == path["C"].max())]
        .assign(size=lambda d: d["coefficient"].abs())
        .nlargest(PATH_LINES, "size")["feature"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    for ax, penalty in zip(axes, ("L2", "L1")):
        part = path[path["penalty"] == penalty]
        wide = part.pivot(index="C", columns="feature", values="coefficient")
        for feature in wide.columns.difference(top):
            ax.plot(wide.index, wide[feature], color=GRID, linewidth=0.8, zorder=1)
        for color, feature in zip(SERIES, top):
            ax.plot(wide.index, wide[feature], color=color, linewidth=2, label=feature.split("__", 1)[1], zorder=3)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set(xlabel="C (larger = weaker penalty)", title=f"{penalty} penalty")
        style_axis(ax)
    axes[0].set_ylabel("Coefficient (standardized columns)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT / "regularization_path.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def regression_plot():
    table = pd.read_csv("results/linear_regression.csv").query("clip")
    residuals = pd.read_csv("results/regression_residuals.csv")
    fig, (curve, resid) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    ols = table[table["model"] == "OLS"].iloc[0]
    curve.axhline(ols["rmse_mean"], color=REFERENCE, linewidth=1.5, label="OLS (no penalty)")
    for model, color in (("Ridge", SERIES[0]), ("Lasso", SERIES[1])):
        part = table[table["model"] == model].sort_values("alpha")
        curve.plot(part["alpha"], part["rmse_mean"], color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=model)
    curve.set_xscale("log")
    curve.set(xlabel="alpha (larger = stronger penalty)", ylabel="Validation RMSE of AMT_CREDIT (5-fold mean)", title="Penalty strength")
    curve.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    resid.scatter(residuals["fitted"] / 1000, residuals["residual"] / 1000, s=5, color=MODEL, alpha=0.18, linewidths=0, rasterized=True)
    bins = pd.qcut(residuals["fitted"], 12, duplicates="drop")
    grouped = residuals.groupby(bins, observed=True)
    centre = grouped["fitted"].mean() / 1000
    resid.plot(centre, grouped["residual"].quantile(0.9) / 1000, color=INK_SECONDARY, linewidth=1.2)
    resid.plot(centre, grouped["residual"].quantile(0.1) / 1000, color=INK_SECONDARY, linewidth=1.2)
    resid.plot(centre, grouped["residual"].mean() / 1000, color=SERIES[1], linewidth=2)
    resid.axhline(0, color=INK, linewidth=0.8)
    resid.text(0.98, 0.97, "orange: mean residual per bin\ngray: 10th and 90th percentile", transform=resid.transAxes, ha="right", va="top", fontsize=8.5, color=INK_SECONDARY)
    resid.set(xlabel="Predicted AMT_CREDIT (thousands)", ylabel="Residual: actual minus predicted (thousands)", title="Residuals, out-of-fold, 10,000 applicants")
    for ax in (curve, resid):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "regression.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_curves()
    validation_curve_plot()
    regularization_path_plot()
    regression_plot()


if __name__ == "__main__":
    main()
