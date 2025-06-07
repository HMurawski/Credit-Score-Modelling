"""
Evaluate trained artefacts and choose decision threshold.
Changes:
* find_threshold  excludes trivial 0/1 cut- offs and uses
a dense grid if precision- recall curve fails.
* CLI flag --grid-step to adjust granularity of fallback grid.
script now auto-picks **latest** subfolder that starts with
`<model>_` (so e.g. `logistic_opt30_2025-06-07_18-56-06`)
"""
import argparse
from pathlib import Path
import json
from typing import List

import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)

from credit_score.features.load_data import load_split
from credit_score.features.cleaning  import clean_and_prepare
import warnings
warnings.filterwarnings("ignore")  #silence sklearn / pandas noise

# ------------------------------------------------------------------
# Utility – decile & KS
# ------------------------------------------------------------------
def _resolve_art_dir(root: Path, model_prefix: str) -> Path:
    """Return artefact directory.  
    If *root* already contains model.pkl, treat it as the artefact folder;
    otherwise pick the **latest** sub-dir whose name starts with
    `<model_prefix>_`."""
    if (root / "model.pkl").exists():
        return root

    cands: List[Path] = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{model_prefix}_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No artefacts for model '{model_prefix}' under {root}")
    return cands[0]

def decile_table(y_true: pd.Series, proba: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Return decile summary and KS statistic."""
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["decile"] = pd.qcut(df["p"], n, labels=False, duplicates="drop")
    g = (
        df.groupby("decile")
        .apply(
            lambda x: pd.Series({
                "min_p": x["p"].min(),
                "max_p": x["p"].max(),
                "events": x["y"].sum(),
                "non_events": len(x) - x["y"].sum(),
            })
        )
        .sort_index(ascending=False)
        .reset_index(drop=True)
    )
    g["event_rate"] = 100 * g["events"] / (g["events"] + g["non_events"])
    g["cum_events"] = g["events"].cumsum()
    g["cum_non_events"] = g["non_events"].cumsum()
    g["cum_event_rate"] = 100 * g["cum_events"] / g["events"].sum()
    g["cum_non_event_rate"] = 100 * g["cum_non_events"] / g["non_events"].sum()
    g["ks"] = (g["cum_event_rate"] - g["cum_non_event_rate"]).abs()
    return g

# ------------------------------------------------------------------
# Threshold search helpers
# ------------------------------------------------------------------

def _best_threshold_curve(y: np.ndarray, proba: np.ndarray, min_prec: float) -> float:
    """Return threshold from PR-curve with maximal recall >= min_prec.
    changed >> ignores thresholds < 1e-4 to avoid trivial 0 cut-off."""
    prec, rec, thr = precision_recall_curve(y, proba)
    mask = (prec >= min_prec) & (thr > 1e-4)  # changed >> filter tiny thr
    if mask.any():
        idx = np.argmax(rec * mask)
        return thr[idx]
    return np.nan  # curve failed constraint


def find_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_precision: float,
    grid_step: float = 0.01,
) -> float:
    """Return threshold with **max recall** s.t. precision ≥ min_precision."""
    best_thr, best_rec = 0.5, -1.0

    prec, rec, thr = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns thr len = len(rec)-1
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if 0.0 < t < 1.0 and p >= min_precision and r > best_rec:
            best_rec, best_thr = r, t

    # fallback grid if constraint never satisfied
    if best_rec < 0:
        for t in np.arange(grid_step, 1.0, grid_step):
            pred = proba >= t
            tp = np.logical_and(pred, y_true == 1).sum()
            fp = np.logical_and(pred, y_true == 0).sum()
            if tp + fp == 0:
                continue
            p = tp / (tp + fp)
            r = tp / (y_true == 1).sum()
            if p >= min_precision and r > best_rec:
                best_rec, best_thr = r, t
    return float(best_thr)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=["logistic", "rf", "xgb"], default="logistic",
                    help="Model prefix to load (default: logistic)")
    pa.add_argument("--split", choices=["train", "val", "test"], default="val")
    pa.add_argument("--artefacts", default="artifacts",
                    help="Artefact dir or root containing multiple runs")
    pa.add_argument("--threshold", default="0.5",
                    help="'auto' or numeric cut-off (default 0.5)")
    pa.add_argument("--val-split", default="val",
                    help="Split used to tune threshold when --threshold auto")
    pa.add_argument("--min-precision", type=float, default=0.0,
                    help="Precision constraint for auto-threshold search")
    pa.add_argument("--grid-step", type=float, default=0.01,
                    help="Grid step for fallback threshold search")
    args = pa.parse_args()

    # pick correct aftifact folder automatically
    art_root = Path(args.artefacts)
    art_dir = _resolve_art_dir(art_root, args.model)
    print(f"[artefacts] Using → {art_dir}")
    
    # ----- load artifacts ------
    preproc = joblib.load(art_dir / "preprocessor.pkl")
    clf     = joblib.load(art_dir / "model.pkl")
    meta    = pickle.load(open(art_dir / "cleaning_meta.pkl", "rb"))

    # load evaluation split
    df_eval = clean_and_prepare(load_split(args.split), meta)
    X_eval, y_eval = df_eval.drop("default", axis=1), df_eval["default"]
    proba_eval = clf.predict_proba(preproc.transform(X_eval))[:, 1]

    # auto threshold (if requested)
    if args.threshold == "auto":
        df_val = clean_and_prepare(load_split(args.val_split), meta)
        proba_val = clf.predict_proba(preproc.transform(df_val.drop("default", axis=1)))[:, 1]
        thr = find_threshold(
            df_val["default"].to_numpy(),
            proba_val,
            min_precision=args.min_precision,
            grid_step=args.grid_step,
        )
        p = ((proba_val >= thr) & (df_val["default"] == 1)).sum() / max(1, (proba_val >= thr).sum())
        r = ((proba_val >= thr) & (df_val["default"] == 1)).sum() / (df_val["default"] == 1).sum()
        print(f"[auto-thr] threshold={thr:.3f} (precision={p:.3f}, recall={r:.3f}) on {args.val_split}")
    else:
        thr = float(args.threshold)

    # final metrics
    y_pred = (proba_eval >= thr).astype(int)
    auc = roc_auc_score(y_eval, proba_eval)
    ks  = decile_table(y_eval, proba_eval)["ks"].max().round(4)
    print(f"{args.split.upper()}  AUC={auc:.4f} | KS={ks:.4f} | thr={thr:.3f}")
    print(classification_report(y_eval, y_pred, digits=4))

    # save deciles
    out_csv = art_dir / f"deciles_{args.split}.csv"
    decile_table(y_eval, proba_eval).to_csv(out_csv, index=False)
    print("Decile table saved →", out_csv.resolve())


if __name__ == "__main__":
    main()