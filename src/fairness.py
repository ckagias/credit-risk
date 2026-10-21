import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate
import features
import segments
from preprocess import load_features
from split import DB

OOF_WITH = segments.OOF
OOF_WITHOUT = Path("data/oof_no_gender.parquet")
GROUPS = Path("results/fairness.csv")
GAPS = Path("results/fairness_gaps.csv")
SEED = 42
BOOTSTRAPS = 1000
CUT = segments.CUT
MODELS = {"with gender": OOF_WITH, "without gender": OOF_WITHOUT}


def group_row(y, p, cut=CUT):
    y, p = np.asarray(y), np.asarray(p)
    refused = p >= cut
    row = segments.segment_row(y, p, np.ones(len(y)), cut=cut, boots=0)
    refused_low, refused_high = segments.interval(int(refused.sum()), len(y))
    row["approval_rate"] = 1 - row["share_refused"]
    row["approval_low"], row["approval_high"] = 1 - refused_high, 1 - refused_low
    row["tpr"] = 1 - row["fnr"]
    row["precision"] = float(y[refused].mean()) if refused.any() else float("nan")
    row["roc_auc"] = evaluate.roc_auc(y, p) if 0 < y.sum() < len(y) else float("nan")
    return row


def labels(X):
    sex = X["CODE_GENDER"].map({"F": "women", "M": "men"})
    age = segments.segment_labels(X)["age"]
    return {"gender": sex, "age": age}


def rates(y, p, cut):
    refused = p >= cut
    return np.array([1 - refused.mean(), (refused & (y == 1)).sum() / (y == 1).sum(), (refused & (y == 0)).sum() / (y == 0).sum()])


def gap_row(y_a, p_a, y_b, p_b, cut=CUT, boots=BOOTSTRAPS, seed=SEED):
    y_a, p_a, y_b, p_b = (np.asarray(v) for v in (y_a, p_a, y_b, p_b))
    rng = np.random.default_rng(seed)
    point = rates(y_a, p_a, cut) - rates(y_b, p_b, cut)
    ratio_point = rates(y_a, p_a, cut)[0] / rates(y_b, p_b, cut)[0]
    draws, ratios = [], []
    for _ in range(boots):
        ia, ib = rng.integers(0, len(y_a), len(y_a)), rng.integers(0, len(y_b), len(y_b))
        a, b = rates(y_a[ia], p_a[ia], cut), rates(y_b[ib], p_b[ib], cut)
        draws.append(a - b)
        ratios.append(a[0] / b[0])
    gap_bounds = np.quantile(draws, [0.025, 0.975], axis=0)
    ratio_low, ratio_high = np.quantile(ratios, [0.025, 0.975])
    names = ["approval_rate", "tpr", "fpr"]
    row = {f"{name}_gap": point[k] for k, name in enumerate(names)}
    row.update({f"{name}_gap_low": gap_bounds[0, k] for k, name in enumerate(names)})
    row.update({f"{name}_gap_high": gap_bounds[1, k] for k, name in enumerate(names)})
    row.update({"approval_ratio": ratio_point, "approval_ratio_low": ratio_low, "approval_ratio_high": ratio_high})
    return row


def group_table(frame, label_map, model):
    y, p = frame["y"].to_numpy(), frame["p"].to_numpy()
    rows = []
    for family, label in label_map.items():
        label = pd.Series(label).reset_index(drop=True)
        for name in segments.segment_names(label.dropna()):
            keep = (label == name).to_numpy()
            rows.append({"model": model, "family": family, "group": name, **group_row(y[keep], p[keep])})
    return pd.DataFrame(rows)


def gap_table(frame, label_map, model, reference):
    y, p = frame["y"].to_numpy(), frame["p"].to_numpy()
    rows = []
    for family, label in label_map.items():
        label = pd.Series(label).reset_index(drop=True)
        names = segments.segment_names(label.dropna())
        for name in names:
            if name == reference[family]:
                continue
            a, b = (label == name).to_numpy(), (label == reference[family]).to_numpy()
            rows.append({"model": model, "family": family, "group": name, "reference": reference[family], **gap_row(y[a], p[a], y[b], p[b])})
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train", features.GROUPS)
    con.close()
    label_map = labels(X)
    frames = {}
    for model, path in MODELS.items():
        if path.exists():
            frames[model] = pd.read_parquet(path)
        else:
            frames[model] = segments.out_of_fold(X.drop(columns=["CODE_GENDER"]), y)
            frames[model].to_parquet(path)
        print(model, evaluate.score(frames[model]["y"], frames[model]["p"], CUT), flush=True)
    groups = pd.concat([group_table(frame, label_map, model) for model, frame in frames.items()], ignore_index=True)
    reference = {"gender": "women", "age": "60 and over"}
    gaps = pd.concat([gap_table(frame, label_map, model, reference) for model, frame in frames.items()], ignore_index=True)
    GROUPS.parent.mkdir(exist_ok=True)
    groups.to_csv(GROUPS, index=False, float_format="%.6g")
    gaps.to_csv(GAPS, index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(groups[["model", "group", "n", "default_rate", "mean_probability", "approval_rate", "tpr", "fpr", "precision", "roc_auc"]].round(3).to_string(index=False))
    print(gaps[["model", "group", "reference", "approval_ratio", "approval_ratio_low", "approval_ratio_high", "tpr_gap", "tpr_gap_low", "tpr_gap_high", "fpr_gap", "fpr_gap_low", "fpr_gap_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
