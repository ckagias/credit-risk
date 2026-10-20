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


def boosting_plot():
    curve = pd.read_csv("results/lightgbm_curve.csv")
    reached = curve.groupby(["learning_rate", "rounds", "split"])["fold"].transform("nunique")
    curve = curve[reached == curve["fold"].nunique()]
    stopped = pd.read_csv("results/lightgbm_stopped.csv").groupby("learning_rate")[["best_round", "pr_auc"]].mean()
    rates = sorted(curve["learning_rate"].unique(), reverse=True)
    fig, (score, gap) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for color, rate in zip(SERIES, rates):
        part = curve[(curve["learning_rate"] == rate) & (curve["split"] == "validation")].groupby("rounds")["pr_auc"].mean()
        score.plot(part.index, part.values, color=color, linewidth=2, label=f"learning rate {rate:g}")
        score.plot([stopped.loc[rate, "best_round"]], [stopped.loc[rate, "pr_auc"]], marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    score.text(0.98, 0.04, "dots: early stopping, mean best round\nand its validation PR-AUC", transform=score.transAxes, ha="right", fontsize=8.5, color=INK_SECONDARY)
    score.set_xscale("log")
    score.set(xlabel="Boosting rounds (trees)", ylabel="Validation PR-AUC (5-fold mean)", title="Score by round", ylim=(0.17, 0.25))
    score.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    slow = curve[curve["learning_rate"] == rates[-1]]
    for split, color, label in (("train", REFERENCE, "Training sample"), ("validation", MODEL, "Validation fold")):
        part = slow[slow["split"] == split].groupby("rounds")["log_loss"].mean()
        gap.plot(part.index, part.values, color=color, linewidth=2 if split == "validation" else 1.5, label=label)
    gap.set_xscale("log")
    gap.set(xlabel="Boosting rounds (trees)", ylabel="Log loss (5-fold mean)", title=f"Training against validation, learning rate {rates[-1]:g}")
    gap.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in (score, gap):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "lightgbm_rounds.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tuning_plot():
    trials = pd.read_csv("results/optuna_trials.csv")
    importance = pd.read_csv("results/optuna_importance.csv").sort_values("importance")
    done = trials[trials["state"] == "COMPLETE"]
    pruned = trials[trials["state"] == "PRUNED"]
    fig, (history, bars) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}, facecolor=SURFACE)
    history.scatter(done["number"], done["value"], s=28, color=MODEL, edgecolors=SURFACE, linewidths=0.8, zorder=3, label=f"Completed trial ({len(done)})")
    history.scatter(pruned["number"], [0.2318] * len(pruned), marker="|", s=90, color=REFERENCE, zorder=3, label=f"Pruned early ({len(pruned)}, drawn at the bottom)")
    history.plot(done["number"], done["value"].cummax(), color=SERIES[1], linewidth=2, drawstyle="steps-post", label="Best so far")
    first = trials.loc[trials["number"] == 0, "value"].iloc[0]
    history.axhline(first, color=REFERENCE, linewidth=1, alpha=0.7)
    history.text(0.99, first - 0.0006, f"default settings, trial 0: {first:.4f}", transform=history.get_yaxis_transform(), ha="right", fontsize=8.5, color=INK_SECONDARY)
    history.set(xlabel="Trial", ylabel="Mean validation PR-AUC (5 folds)", title="Optuna history, LightGBM at learning rate 0.05", ylim=(0.2305, 0.2445))
    history.legend(loc="lower right", bbox_to_anchor=(1.0, 0.15), frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    bars.barh(importance["parameter"], importance["importance"], color=MODEL, height=0.6)
    for y, value in enumerate(importance["importance"]):
        bars.text(value + 0.004, y, f"{value:.2f}", va="center", fontsize=9, color=INK_SECONDARY)
    bars.set(xlabel="Share of the variance in PR-AUC explained (fANOVA)", title="Which settings mattered", xlim=(0, 0.33))
    for ax in (history, bars):
        style_axis(ax)
    bars.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "optuna_history.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tree_diagram_plot():
    nodes = pd.read_csv("results/tree_depth3_nodes.csv").set_index("node")
    position = {}
    leaves = [n for n in nodes.index if nodes.loc[n, "left"] == -1]

    def place(node, depth):
        if nodes.loc[node, "left"] == -1:
            position[node] = (leaves.index(node), -depth)
        else:
            left, right = place(nodes.loc[node, "left"], depth + 1), place(nodes.loc[node, "right"], depth + 1)
            position[node] = ((left + right) / 2, -depth)
        return position[node][0]

    place(0, 0)
    top = nodes["default_rate"].max()
    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=SURFACE)
    for node, row in nodes.iterrows():
        if row["left"] == -1:
            continue
        for child, label in ((int(row["left"]), "yes"), (int(row["right"]), "no")):
            (x0, y0), (x1, y1) = position[node], position[child]
            ax.plot([x0, x1], [y0 - 0.17, y1 + 0.17], color=INK_SECONDARY, linewidth=1, zorder=1)
            ax.text((x0 + x1) / 2 + (-0.06 if label == "yes" else 0.06), (y0 + y1) / 2, label, fontsize=8, color=INK_SECONDARY, ha="right" if label == "yes" else "left", va="center")
    for node, row in nodes.iterrows():
        x, y = position[node]
        shade = 0.12 + 0.88 * row["default_rate"] / top
        face = tuple(1 - shade * (1 - c) for c in matplotlib.colors.to_rgb(SERIES[1]))
        rate = f"{100 * row['default_rate']:.1f}% default"
        if row["left"] == -1:
            text = f"{int(row['samples']):,}\n{rate}"
        else:
            text = f"{row['feature']} <= {row['threshold']:.3f}\n{int(row['samples']):,} applicants\n{rate}"
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=INK, zorder=3, bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=GRID, linewidth=1))
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(-3.45, 0.4)
    ax.axis("off")
    ax.set_title("Depth-3 tree on the 246,008 training applicants (Gini). Box colour: share of defaults", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "tree_diagram.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def impurity_plot():
    split = pd.read_csv("results/tree_root_split.csv").set_index("criterion").loc["gini"]
    p = np.linspace(0, 0.4, 400)
    gini = 2 * p * (1 - p)
    entropy = -(p * np.log2(p, where=p > 0, out=np.zeros_like(p)) + (1 - p) * np.log2(1 - p)) / 2
    parent, left, right = split["defaults"] / split["samples"], split["defaults_left"] / split["n_left"], split["defaults_right"] / split["n_right"]
    share = split["n_left"] / split["samples"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.plot(p, gini, color=MODEL, linewidth=2, label="Gini, 2p(1-p)")
    ax.plot(p, entropy, color=SERIES[1], linewidth=2, linestyle="--", label="Entropy in bits, divided by 2")
    ax.plot([left, right], [split["impurity_left"], split["impurity_right"]], color=INK_SECONDARY, linewidth=1, zorder=2)
    for x, y, label, text_at in (
        (parent, split["impurity_parent"], f"parent {split['impurity_parent']:.4f}", (0.17, 0.155)),
        (left, split["impurity_left"], f"left {split['impurity_left']:.4f}", (0.19, 0.27)),
        (right, split["impurity_right"], f"right {split['impurity_right']:.4f}", (0.17, 0.105)),
    ):
        ax.plot([x], [y], marker="o", markersize=7, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=text_at, textcoords="data", ha="left", va="center", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.8, shrinkA=2, shrinkB=4))
    ax.set(xlabel="Share of defaults in the node, p", ylabel="Impurity", title="Impurity of the root node and its two children", xlim=(0, 0.4))
    ax.text(0.98, 0.05, f"root split EXT_SOURCE_3 <= {split['threshold']:.3f}\nweighted children {share * split['impurity_left'] + (1 - share) * split['impurity_right']:.4f}, decrease {split['decrease']:.4f}", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "impurity.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def bagging_variance_plot():
    table = pd.read_csv("results/forest_variance.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    for color, (features, part) in zip(SERIES, table.groupby("max_features")):
        rho, sigma2 = part["rho"].iloc[0], part["sigma2"].iloc[0]
        b = np.logspace(0, np.log10(part["trees"].max()), 100)
        ax.plot(b, rho * sigma2 + (1 - rho) * sigma2 / b, color=color, linewidth=1.5, zorder=2)
        ax.plot(part["trees"], part["variance"], marker="o", markersize=6, linestyle="none", color=color, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4, label=f"{features} columns per split: single tree {sigma2:.4f}, rho {rho:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 30])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "30"])
    ax.set(xlabel="Trees averaged", ylabel="Variance of the predicted probability", title="Averaging trees removes variance")
    ax.text(0.02, 0.04, "dots: measured over 8 training subsamples and 49,202 applicants\nlines: $\\rho\\sigma^2 + (1-\\rho)\\sigma^2 / B$, $\\sigma^2$ = variance of one tree", transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "bagging_variance.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def boosting_demo_plot():
    demo = pd.read_csv("results/boosting_demo.csv")
    demo = demo[demo["round"] >= 5]
    fig, (loss, pr) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for column, color, style, label in (
        ("train_log_loss", REFERENCE, "-", "Training, hand-written"),
        ("lightgbm_train_log_loss", REFERENCE, "--", "Training, LightGBM"),
        ("validation_log_loss", MODEL, "-", "Validation, hand-written"),
        ("lightgbm_validation_log_loss", SERIES[1], "--", "Validation, LightGBM"),
    ):
        loss.plot(demo["round"], demo[column], color=color, linestyle=style, linewidth=2 if "validation" in column else 1.5, label=label)
    for column, color, style, label in (
        ("validation_pr_auc", MODEL, "-", "Hand-written"),
        ("lightgbm_validation_pr_auc", SERIES[1], "--", "LightGBM"),
    ):
        pr.plot(demo["round"], demo[column], color=color, linestyle=style, linewidth=2, label=label)
    loss.set(xlabel="Boosting rounds", ylabel="Log loss (fold 0)", title="Log loss by round")
    pr.set(xlabel="Boosting rounds", ylabel="Validation PR-AUC (fold 0)", title="Validation PR-AUC by round")
    loss.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    pr.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in (loss, pr):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "boosting_demo.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pca_plot():
    variance = pd.read_csv("results/pca_variance.csv").groupby("component")[["explained_ratio", "cumulative"]].mean()
    models = pd.read_csv("results/pca_models.csv")
    names = {"logistic": "Logistic regression", "lightgbm": "LightGBM", "naive_bayes": "Naive Bayes"}
    fig, (scree, cumulative, scores) = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    scree.bar(variance.index, variance["explained_ratio"], color=MODEL, width=0.7)
    scree.annotate(f"component 1: {variance['explained_ratio'].iloc[0]:.0%}", (1, variance["explained_ratio"].iloc[0]), xytext=(6, -2), textcoords="offset points", fontsize=9, color=INK_SECONDARY, va="top")
    scree.set(xlabel="Component", ylabel="Share of the variance", title="Scree plot, 70 continuous columns", xlim=(0, 71))
    cumulative.plot(variance.index, variance["cumulative"], color=MODEL, linewidth=2)
    for target in (0.8, 0.9, 0.95):
        needed = int((variance["cumulative"] < target).sum() + 1)
        cumulative.plot([needed], [target], marker="o", markersize=7, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        cumulative.annotate(f"{needed} components: {target:.0%}", (needed, target), xytext=(8, -12), textcoords="offset points", fontsize=9, color=INK_SECONDARY)
    cumulative.set(xlabel="Number of components", ylabel="Cumulative share of the variance", title="Variance kept", xlim=(0, 71), ylim=(0, 1.02))
    for color, (model, label) in zip(SERIES, names.items()):
        part = models[(models["model"] == model)]
        curve = part[part["representation"] == "pca"].groupby("n_columns")["pr_auc"].mean()
        raw = part[part["representation"] == "raw"]["pr_auc"].mean()
        scores.plot(curve.index, curve.values, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        scores.axhline(raw, color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.8)
    scores.plot([], [], color=REFERENCE, linewidth=1.2, linestyle=(0, (4, 3)), label="Same model on the raw columns")
    scores.set(xlabel="Number of components used", ylabel="Mean validation PR-AUC (5 folds)", title="Does PCA help a model?", xlim=(0, 72))
    scores.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in (scree, cumulative, scores):
        style_axis(ax)
    scree.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "pca.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def kmeans_plot():
    summary = pd.read_csv("results/kmeans_summary.csv")
    clusters = pd.read_csv("results/kmeans_clusters.csv")
    mean = summary.groupby("k")[["inertia_per_row", "silhouette", "roc_auc"]].mean()
    base_rate = clusters[clusters["k"] == 2].groupby("fold").apply(lambda g: (g["n_val"] * g["rate_val"]).sum() / g["n_val"].sum(), include_groups=False).mean()
    fig, (elbow, rates, scores) = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    elbow.plot(mean.index, mean["inertia_per_row"], color=MODEL, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    elbow.set(xlabel="Number of clusters k", ylabel="Inertia per row (blue)", title="Inertia falls with every k", xticks=list(mean.index))
    silhouette = elbow.twinx()
    silhouette.plot(mean.index, mean["silhouette"], color=SERIES[1], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    silhouette.set_ylabel("Silhouette score (orange)", color=INK_SECONDARY)
    silhouette.spines["top"].set_visible(False)
    five = clusters[(clusters["k"] == 5) & (clusters["fold"] == 0)].sort_values("rate_val")
    positions = np.arange(len(five))
    rates.bar(positions, five["rate_val"], color=MODEL, width=0.7)
    rates.axhline(base_rate, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)), label=f"Overall rate ({base_rate:.3f})")
    for x, (rate, n) in enumerate(zip(five["rate_val"], five["n_val"])):
        rates.annotate(f"n={n:,}", (x, rate), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
    rates.set(xlabel="Cluster (k = 5, fold 0, sorted by rate)", ylabel="Default rate in the validation fold", title="Default rate per cluster", ylim=(0, five["rate_val"].max() * 1.15), xticks=positions, xticklabels=[str(c) for c in five["cluster"]])
    rates.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    scores.plot(mean.index, mean["roc_auc"], color=MODEL, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    scores.axhline(0.5, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)), label="No ranking (0.5)")
    scores.set(xlabel="Number of clusters k", ylabel="Validation ROC-AUC of the cluster default rate", title="Cluster as a risk score", xticks=list(mean.index))
    scores.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in (elbow, rates, scores):
        style_axis(ax)
    rates.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "kmeans.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def knn_distance_plot():
    distance = pd.read_csv("results/knn_distance.csv")
    curve = distance.groupby("columns")["ratio"].agg(["mean", "min", "max"])
    sweep = pd.read_csv("results/knn_svm.csv").query("experiment == 'knn_k'")
    sweep = sweep.assign(setting=sweep["setting"].astype(float)).groupby("setting")["pr_auc"].mean().sort_index()
    ext = pd.read_csv("results/knn_ext.csv")
    ext_knn = ext[ext["model"] == "knn"].groupby("setting")["pr_auc"].mean()
    ext_logistic = ext[ext["model"] == "logistic"]["pr_auc"].mean()
    fig, (contrast, neighbours) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    contrast.fill_between(curve.index, curve["min"], curve["max"], color=MODEL, alpha=0.15, linewidth=0)
    contrast.plot(curve.index, curve["mean"], color=MODEL, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    contrast.set_xscale("log")
    contrast.set_xticks(curve.index)
    contrast.set_xticklabels([str(c) for c in curve.index])
    contrast.set(xlabel="Number of continuous columns used", ylabel="Distance to nearest neighbour / median distance", title="The nearest neighbour is not much nearer")
    contrast.text(0.04, 0.9, "300 validation applicants against 20,000 training rows,\nrandom columns, band: min and max of 5 draws", transform=contrast.transAxes, fontsize=8.5, color=INK_SECONDARY, va="top")
    neighbours.plot(sweep.index, sweep.values, color=MODEL, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label="kNN, all columns")
    neighbours.plot(ext_knn.index, ext_knn.values, color=SERIES[1], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label="kNN, 3 EXT_SOURCE columns")
    neighbours.axhline(ext_logistic, color=SERIES[1], linewidth=1.2, linestyle=(0, (4, 3)), label="Logistic regression, 3 EXT_SOURCE columns")
    neighbours.axhline(0.2139, color=MODEL, linewidth=1.2, linestyle=(0, (4, 3)), label="Logistic regression, all columns")
    neighbours.set_xscale("log")
    neighbours.set_xticks(sweep.index)
    neighbours.set_xticklabels([str(int(k)) for k in sweep.index])
    neighbours.set(xlabel="Number of neighbours k", ylabel="Mean validation PR-AUC (5 folds)", title="kNN needs a large k, 20,000 training rows")
    neighbours.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    for ax in (contrast, neighbours):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "knn_distance.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def svm_margin_plot():
    support = pd.read_csv("results/svm_support.csv")
    margins = pd.read_csv("results/svm_margins.csv")
    m = (2 * margins["y"] - 1) * margins["f"]
    grid = np.linspace(-2.5, 2.5, 200)
    fig, (loss, bound) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    loss.plot(grid, (grid < 0).astype(float), color=REFERENCE, linewidth=1.5, label="0-1 loss")
    loss.plot(grid, np.maximum(0, 1 - grid), color=MODEL, linewidth=2.2, label="Hinge loss (SVM)")
    loss.plot(grid, np.log2(1 + np.exp(-grid)), color=SERIES[1], linewidth=2.2, label="Log loss (logistic regression), in bits")
    loss.set(xlabel="Margin $y_i f(x_i)$, $y_i = \\pm 1$", ylabel="Loss", title="Hinge loss against log loss", xlim=(-2.5, 2.5), ylim=(0, 3.6))
    histogram = loss.twinx()
    histogram.hist([m[margins["y"] == 0], m[margins["y"] == 1]], bins=np.linspace(-2.5, 2.5, 41), color=[SERIES[2], SERIES[3]], alpha=0.45, label=["Validation non-defaulters", "Validation defaulters"])
    histogram.set_ylim(0, margins.shape[0] * 0.5)
    histogram.set_yticks([])
    histogram.spines["top"].set_visible(False)
    handles = loss.get_legend_handles_labels()
    extra = histogram.get_legend_handles_labels()
    loss.legend(handles[0] + extra[0], handles[1] + extra[1], loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    bound.plot(support["C"], support["share_support"], color=MODEL, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label="Support vectors")
    bound.plot(support["C"], support["share_at_upper_bound"], color=SERIES[1], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label="Support vectors at the upper bound ($\\alpha_i = C$)")
    for c, width in zip(support["C"], support["margin_width"]):
        bound.annotate(f"margin width {width:.3f}", (c, 0.45), ha="center", fontsize=8.5, color=INK_SECONDARY)
    bound.set_xscale("log")
    bound.set_xticks(support["C"])
    bound.set_xticklabels([f"{c:g}" for c in support["C"]])
    bound.set(xlabel="C", ylabel="Share of the 5,000 training rows", title="Larger C, fewer points inside the margin", ylim=(0.4, 1.02), xlim=(0.04, 25))
    bound.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    for ax in (loss, bound):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "svm_margin.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pca_math_plot():
    rec = pd.read_csv("results/pca_reconstruction.csv")
    deciles = pd.read_csv("results/pca_deciles.csv")
    fig, (error, rates) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    error.plot(rec["components"], rec["val_error"], color=MODEL, linewidth=2, label="Measured on the validation fold")
    error.plot(rec["components"], rec["tail_eigenvalues"], color=SERIES[1], linewidth=0, marker="o", markersize=4, label="Sum of the discarded eigenvalues, from the training fold")
    error.set(xlabel="Components kept", ylabel="Mean squared reconstruction error", title="Reconstruction error is the discarded variance", xlim=(0, 70))
    error.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    overall = np.average(deciles["default_rate"], weights=deciles["n"])
    for color, (component, part) in zip(SERIES, deciles.groupby("component")):
        rates.plot(part["decile"], part["default_rate"], color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"Component {component}")
    rates.axhline(overall, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)), label=f"Overall rate ({overall:.3f})")
    rates.set(xlabel="Decile of the component score", ylabel="Default rate in the validation fold", title="Default rate along the first three components", xticks=range(1, 11))
    rates.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    for ax in (error, rates):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "pca_math.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def kmeans_lloyd_plot():
    lloyd = pd.read_csv("results/kmeans_lloyd.csv")
    clusters = pd.read_csv("results/kmeans_clusters.csv").query("fold == 0 and k == 5")
    summary = pd.read_csv("results/kmeans_summary.csv").query("fold == 0 and k == 5")
    plus_plus = summary["inertia_per_row"].iloc[0] * clusters["n_fit"].sum()
    fig, (path, gap) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for color, (start, part) in zip(SERIES, lloyd.groupby("start")):
        path.plot(part["iteration"], part["after_assign"] / 1e6, color=color, linewidth=1.8, label=f"Random start {start}: {part['final'].iloc[0] / 1e6:.2f}")
        gap.plot(part["iteration"], (part["after_assign"] - part["final"]) / part["final"], color=color, linewidth=1.8)
    path.axhline(plus_plus / 1e6, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)), label=f"k-means++ start: {plus_plus / 1e6:.2f}")
    path.set(xlabel="Iteration", ylabel="Objective $J$ (millions)", title="Each iteration lowers the objective", xlim=(1, 40), ylim=(10, 16))
    path.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    gap.set_yscale("symlog", linthresh=1e-5)
    gap.set(xlabel="Iteration", ylabel="Relative gap to the final objective", title="Convergence to different local minima", xlim=(1, 120))
    for ax in (path, gap):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "kmeans_lloyd.png", dpi=160, facecolor=SURFACE)
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
    boosting_plot()
    tuning_plot()
    tree_diagram_plot()
    impurity_plot()
    bagging_variance_plot()
    boosting_demo_plot()
    pca_plot()
    kmeans_plot()
    knn_distance_plot()
    svm_margin_plot()
    pca_math_plot()
    kmeans_lloyd_plot()


if __name__ == "__main__":
    main()
