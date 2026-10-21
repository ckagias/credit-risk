import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import evaluate
import explain
import features
import threshold
from forest import prepare_folds
from preprocess import load_features
from split import DB

OOF = Path("data/oof_best.parquet")
RESULTS = Path("results/segments.csv")
SEED = 42
BOOTSTRAPS = 200
CONFIDENCE = 0.95
AGE_EDGES = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["under 30", "30 to 39", "40 to 49", "50 to 59", "60 and over"]
INCOME_BANDS = 5
SCHEME = threshold.AMOUNT
CUT = threshold.bayes_threshold(SCHEME)


def out_of_fold(X, y):
    predictions = np.empty(len(y))
    fold = np.empty(len(y), dtype=int)
    for part in prepare_folds(X, y, seed=SEED):
        model = explain.fit_best(part)
        predictions[part["val_idx"]] = model.predict_proba(part["Z_val"])[:, 1]
        fold[part["val_idx"]] = part["fold"]
        print(f"fold {part['fold']} done", flush=True)
    return pd.DataFrame({"fold": fold, "y": y, "p": predictions})


def income_band(income, bands=INCOME_BANDS):
    codes = pd.qcut(income, bands, labels=False, duplicates="drop")
    edges = income.groupby(codes).agg(["min", "max"])
    names = {}
    for code, row in edges.iterrows():
        last = code == edges.index.max()
        span = f"{int(row['min']) // 1000}k and over" if last else f"{int(row['min']) // 1000}k to {int(row['max']) // 1000}k"
        names[code] = f"income {span}"
    ordered = [names[code] for code in sorted(names)]
    return pd.Series(pd.Categorical(codes.map(names), categories=ordered, ordered=True), index=income.index)


def segment_labels(X):
    years = -X["DAYS_BIRTH"] / 365.25
    return {
        "income": income_band(X["AMT_INCOME_TOTAL"]),
        "age": pd.cut(years, AGE_EDGES, labels=AGE_LABELS, right=False),
        "contract type": X["NAME_CONTRACT_TYPE"].astype(str),
        "bureau record": np.where(X["bur_n_credits"] > 0, "has bureau record", "no bureau record"),
        "EXT_SOURCE scores known": X["app_ext_n_known"].astype(int).map(lambda n: f"{n} of 3 known"),
    }


def interval(successes, trials):
    if trials == 0:
        return float("nan"), float("nan")
    ci = binomtest(int(successes), int(trials)).proportion_ci(confidence_level=CONFIDENCE, method="wilson")
    return float(ci.low), float(ci.high)


def segment_row(y, p, size, cut=CUT, scheme=SCHEME, boots=BOOTSTRAPS):
    y, p = np.asarray(y), np.asarray(p)
    refused = p >= cut
    positives, negatives = int(y.sum()), int((1 - y).sum())
    missed, refused_good = int(((y == 1) & ~refused).sum()), int(((y == 0) & refused).sum())
    fnr_low, fnr_high = interval(missed, positives)
    fpr_low, fpr_high = interval(refused_good, negatives)
    rate_low, rate_high = interval(positives, len(y))
    cost = 1000 * threshold.cost_per_applicant(y, p, np.array([cut]), scheme, size)[0]
    nobody = 1000 * threshold.cost_per_applicant(y, p, np.array([1.0 + 1e-9]), scheme, size)[0]
    row = {
        "n": len(y),
        "default_rate": positives / len(y),
        "default_rate_low": rate_low,
        "default_rate_high": rate_high,
        "mean_probability": float(p.mean()),
        "share_refused": float(refused.mean()),
        "fnr": missed / positives if positives else float("nan"),
        "fnr_low": fnr_low,
        "fnr_high": fnr_high,
        "fpr": refused_good / negatives if negatives else float("nan"),
        "fpr_low": fpr_low,
        "fpr_high": fpr_high,
        "cost_per_1000": cost,
        "cost_refuse_nobody": nobody,
    }
    if positives and negatives and boots:
        scored = evaluate.bootstrap(y, p, n_boot=boots, seed=SEED, threshold=cut)
        for metric in ("roc_auc", "pr_auc"):
            row[metric] = float(scored.loc[metric, "estimate"])
            row[f"{metric}_low"] = float(scored.loc[metric, "low"])
            row[f"{metric}_high"] = float(scored.loc[metric, "high"])
    return row


def segment_names(label):
    if isinstance(label.dtype, pd.CategoricalDtype):
        return [name for name in label.cat.categories if (label == name).any()]
    return sorted(label.unique())


def segment_table(frame, labels, size):
    y, p = frame["y"].to_numpy(), frame["p"].to_numpy()
    rows = []
    for family, label in labels.items():
        label = pd.Series(label).reset_index(drop=True)
        for name in segment_names(label):
            keep = (label == name).to_numpy()
            rows.append({"family": family, "segment": name, **segment_row(y[keep], p[keep], size[keep])})
    rows.append({"family": "all", "segment": "all applicants", **segment_row(y, p, size)})
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train", features.GROUPS)
    con.close()
    if OOF.exists():
        frame = pd.read_parquet(OOF)
    else:
        frame = out_of_fold(X, y)
        OOF.parent.mkdir(exist_ok=True)
        frame.to_parquet(OOF)
    print("out-of-fold", evaluate.score(frame["y"], frame["p"], CUT))
    size = (X["AMT_CREDIT"] / X["AMT_CREDIT"].mean()).to_numpy()
    table = segment_table(frame, segment_labels(X), size)
    RESULTS.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    columns = ["family", "segment", "n", "default_rate", "mean_probability", "roc_auc", "fnr", "fpr", "share_refused", "cost_per_1000", "cost_refuse_nobody"]
    print(table[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
