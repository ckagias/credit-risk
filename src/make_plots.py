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


def loss_curve_plot():
    history = pd.read_csv("results/gradient_descent.csv")
    optimum = history["library_optimum"].iloc[0]
    shown = history[history["step"] >= 25]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.axhline(optimum, color=REFERENCE, linewidth=1.5, label=f"scikit-learn solution ({optimum:.4f})")
    ax.plot(shown["step"], shown["objective"], color=MODEL, linewidth=2, label="Gradient descent, log loss plus penalty")
    last = shown.iloc[-1]
    ax.plot([last["step"]], [last["objective"]], marker="o", markersize=8, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"{last['objective']:.4f} after {int(last['step']):,} steps", (last["step"], last["objective"]), textcoords="offset points", xytext=(-8, 14), ha="right", fontsize=9, color=INK)
    ax.text(0.02, 0.04, "step 0: 0.6931 (every probability 0.5), off the chart", transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_ylim(0.245, 0.285)
    ax.set(xlabel="Gradient steps", ylabel="Training objective (mean log loss + penalty)", title="Gradient descent on the logistic loss, C = 0.03")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "logistic_loss_curve.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def sigmoid_plot():
    sample = pd.read_csv("results/logit_sample.csv")
    base_rate = sample["default"].mean()
    z = np.linspace(-9, 4, 400)
    fig, (curve, hist) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}, facecolor=SURFACE)
    curve.plot(z, 1 / (1 + np.exp(-z)), color=INK_SECONDARY, linewidth=2)
    curve.axvline(0, color=GRID, linewidth=1)
    curve.axhline(0.5, color=GRID, linewidth=1)
    curve.text(0.15, 0.52, "probability 0.5 (score 0)", fontsize=8.5, color=INK_SECONDARY)
    curve.set(ylabel="Predicted probability of default", ylim=(0, 1), title="Sigmoid and the out-of-fold scores of 20,000 training applicants")
    bins = np.linspace(-9, 4, 66)
    for label, flag, color in (("Did not default", 0, SERIES[0]), ("Defaulted", 1, SERIES[1])):
        values = sample.loc[sample["default"] == flag, "logit"]
        hist.hist(values, bins=bins, density=True, color=color, alpha=0.55, label=f"{label} ({len(values):,})", linewidth=0)
        hist.axvline(values.median(), color=color, linewidth=2)
    hist.set(xlabel="Score z (log-odds of default)", ylabel="Share of applicants (density)")
    hist.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in (curve, hist):
        style_axis(ax)
    ticks = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9]
    top = curve.secondary_xaxis("top", functions=(lambda z: 1 / (1 + np.exp(-z)), lambda p: np.log(p / (1 - p))))
    top.set_xticks(ticks)
    top.set_xticklabels([f"{t:g}" for t in ticks])
    top.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    top.set_xlabel("Probability at that score", color=INK_SECONDARY, fontsize=9)
    top.spines["top"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT / "logistic_sigmoid.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tree_depth_plot():
    curve = pd.read_csv("results/tree_depth.csv")
    stats = curve.groupby(["split", curve["max_depth"].fillna(0)])[["pr_auc", "log_loss"]].agg(["mean", "std"])
    depths = sorted(curve["max_depth"].fillna(0).unique(), key=lambda d: np.inf if d == 0 else d)
    positions = np.arange(len(depths))
    labels = [f"{int(d)}" if d else "none" for d in depths]
    floor = pd.read_csv("results/baselines.csv").query("metric == 'pr_auc' and model.str.startswith('EXT_SOURCE')")["mean"].iloc[0]
    fig, (pr, loss) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, metric in ((pr, "pr_auc"), (loss, "log_loss")):
        for split, color, label in (("train", REFERENCE, "Training folds"), ("validation", MODEL, "Validation fold")):
            part = stats.loc[split].loc[depths, metric]
            ax.plot(positions, part["mean"], color=color, linewidth=2 if split == "validation" else 1.5, label=label)
            if split == "validation":
                ax.fill_between(positions, part["mean"] - part["std"], part["mean"] + part["std"], color=color, alpha=0.15, linewidth=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("max_depth (none = grown until every leaf is pure)")
        style_axis(ax)
    validation = stats.loc["validation"].loc[depths]
    best = validation["pr_auc"]["mean"].to_numpy().argmax()
    pr.plot([positions[best]], [validation["pr_auc"]["mean"].iloc[best]], marker="o", markersize=8, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    pr.annotate(f"best: depth {labels[best]}\nPR-AUC {validation['pr_auc']['mean'].iloc[best]:.4f}", (positions[best], validation["pr_auc"]["mean"].iloc[best]), textcoords="offset points", xytext=(-8, -34), ha="right", fontsize=9, color=INK)
    pr.axhline(floor, color=REFERENCE, linewidth=1, alpha=0.6)
    pr.text(0.015, floor + 0.012, "EXT_SOURCE baseline", transform=pr.get_yaxis_transform(), fontsize=8.5, color=INK_SECONDARY)
    pr.set(ylabel="PR-AUC (5-fold mean, band = 1 std)", ylim=(0, 1.03), title="PR-AUC against tree depth")
    pr.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    loss.set_yscale("log")
    loss.text(0.98, 0.04, "none: training log loss is 0, off the chart", transform=loss.transAxes, ha="right", fontsize=8.5, color=INK_SECONDARY)
    loss.set(ylabel="Log loss (log scale)", title="Log loss against tree depth")
    fig.tight_layout()
    fig.savefig(OUT / "tree_depth.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def forest_plot():
    results = pd.read_csv("results/forest.csv")
    logistic = pd.read_csv("results/logistic.csv").query("metric == 'pr_auc'")["mean"].iloc[0]
    panels = (
        ("min_samples_leaf", "min_samples_leaf (larger = smaller trees)", "Leaf size, 100 trees"),
        ("max_features", "max_features (columns tried per split, of 300)", "Feature subsampling, 100 trees"),
        ("trees", "Number of trees", "Number of trees"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (experiment, xlabel, title) in zip(axes, panels):
        part = results[(results["experiment"] == experiment) & (results["split"] == "validation")]
        stats = part.groupby("setting")["pr_auc"].agg(["mean", "std"])
        ax.axhline(logistic, color=REFERENCE, linewidth=1, alpha=0.7)
        ax.plot(stats.index, stats["mean"], color=MODEL, linewidth=2)
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=MODEL, alpha=0.15, linewidth=0)
        best = stats["mean"].idxmax()
        ax.plot([best], [stats.loc[best, "mean"]], marker="o", markersize=8, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"{best:g}: {stats.loc[best, 'mean']:.4f}", (best, stats.loc[best, "mean"]), textcoords="offset points", xytext=(0, -26), ha="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set(xlabel=xlabel, title=title, ylim=(0.12, 0.24))
        style_axis(ax)
    axes[0].text(0.02, logistic + 0.0025, "logistic regression", transform=axes[0].get_yaxis_transform(), fontsize=8.5, color=INK_SECONDARY)
    axes[0].set_ylabel("Validation PR-AUC (5-fold mean, band = 1 std)")
    fig.tight_layout()
    fig.savefig(OUT / "forest.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_curves()
    validation_curve_plot()
    regularization_path_plot()
    regression_plot()
    loss_curve_plot()
    sigmoid_plot()
    tree_depth_plot()
    forest_plot()


if __name__ == "__main__":
    main()
