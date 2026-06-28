"""Ridge baseline: how much cross-omic signal is linearly recoverable?

WHY
---
The transformer collapses to the column mean (per-feature Pearson r ~0.04 for
Model A, ~0.01 for Model B; MSE never beats mean-fill). That can mean either
(a) the pipeline is throwing away signal, or (b) there is almost no signal to
begin with. A plain ridge regression is the cleanest way to find the *ceiling*:

  * If ridge ALSO lands near r=0, the cross-omic signal basically isn't there
    and no architecture change will rescue it -> reframe the problem
    (predict only the subset of features that are associated, accept a low
    ceiling, etc.).
  * If ridge gets meaningfully above 0, the signal exists and the transformer
    is underfitting -> the tokenizer / loss / capacity fixes are worth doing.

This script reuses the pipeline's EXACT split + preprocessing + Mantel
evaluator, so every number is directly comparable to the notebook tables.

Reference (transformer, from your runs):
    Model A  Mantel r ~0.45-0.51 | per-feat r ~0.04 | MSE ~0.887 (mean-fill 0.888)
    Model B  Mantel r ~0.60      | per-feat r ~0.01 | MSE ~2.7e-4 (mean-fill 2.6e-4)

RUN
---
    python ridge_baseline.py
(no torch needed; ridge is fast.)
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from pipeline.config import Config
from pipeline.data import MetabolomePreprocessor, OmicsData, StratifiedSplitter
from pipeline.evaluation import Evaluator

ALPHAS = np.logspace(-2, 5, 15)   # ridge strengths searched per target


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def per_feature_stats(true: np.ndarray, pred: np.ndarray) -> dict:
    """Per-output-column Pearson r and R^2 vs the column-mean baseline."""
    rs, r2s = [], []
    col_mean = true.mean(axis=0, keepdims=True)
    ss_tot = ((true - col_mean) ** 2).sum(axis=0)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    for j in range(true.shape[1]):
        t, p = true[:, j], pred[:, j]
        if np.std(t) > 1e-12 and np.std(p) > 1e-12:
            rs.append(np.corrcoef(t, p)[0, 1])
        if ss_tot[j] > 1e-12:
            r2s.append(1.0 - ss_res[j] / ss_tot[j])
    rs = np.array(rs)
    r2s = np.array(r2s)
    return {
        "mean_r": float(np.nanmean(rs)),
        "median_r": float(np.nanmedian(rs)),
        "frac_r>0.1": float(np.mean(rs > 0.1)),
        "frac_r>0.3": float(np.mean(rs > 0.3)),
        "n_features": int(true.shape[1]),
        "mean_R2": float(np.nanmean(r2s)),
        "frac_R2>0": float(np.mean(r2s > 0)),
    }


def fit_ridge(Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray) -> np.ndarray:
    """Standardize X (fit on train), then multi-output RidgeCV (alpha per target)."""
    sx = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
    model.fit(sx.transform(Xtr), Ytr)
    return model.predict(sx.transform(Xva))


def clr(frac: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform of compositional rows (adds a pseudocount)."""
    x = np.asarray(frac, dtype=np.float64) + pseudocount
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def report(title: str, true: np.ndarray, pred: np.ndarray, mean_fill: np.ndarray,
           mantel_model=None, mantel_mean=None) -> dict:
    s_model = per_feature_stats(true, pred)
    mse_model = float(np.mean((pred - true) ** 2))
    mse_mean = float(np.mean((mean_fill - true) ** 2))
    print(f"\n--- {title} ---")
    print(f"  per-feature mean r   : {s_model['mean_r']:+.4f}  "
          f"(median {s_model['median_r']:+.4f})")
    print(f"  features with r>0.1  : {s_model['frac_r>0.1']*100:5.1f}%   "
          f"r>0.3: {s_model['frac_r>0.3']*100:4.1f}%   (of {s_model['n_features']})")
    print(f"  per-feature mean R^2 : {s_model['mean_R2']:+.4f}  "
          f"(features with R^2>0: {s_model['frac_R2>0']*100:.1f}%)")
    print(f"  MSE  ridge={mse_model:.6f}   mean-fill={mse_mean:.6f}   "
          f"({'BEATS' if mse_model < mse_mean else 'no better than'} mean-fill)")
    if mantel_model is not None:
        print(f"  Mantel ridge : {mantel_model}")
        print(f"  Mantel mean  : {mantel_mean}")
    row = {"direction": title, **s_model,
           "mse_ridge": round(mse_model, 6), "mse_mean_fill": round(mse_mean, 6)}
    if mantel_model is not None:
        row["mantel_r"] = round(mantel_model.statistic, 4)
        row["mantel_mean_fill"] = round(mantel_mean.statistic, 4)
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = Config()

    # Exact same data prep as the pipeline.
    data = OmicsData.load(cfg.data)
    split = StratifiedSplitter(cfg.data).split(data)
    pre = MetabolomePreprocessor(cfg.data)
    pre.fit(data.metabolome.loc[split.train])

    micro_tr = data.microbiome.loc[split.train].values
    micro_va = data.microbiome.loc[split.val].values
    metab_tr = pre.transform(data.metabolome.loc[split.train]).values  # standardized
    metab_va = pre.transform(data.metabolome.loc[split.val]).values

    print(f"[data] train={len(split.train)} val={len(split.val)} "
          f"| micro={micro_tr.shape[1]} feats  metab={metab_tr.shape[1]} feats")

    ev = Evaluator(cfg.eval)
    rows = []

    # ---- Direction A: microbiome -> standardized metabolome ---------------- #
    predA = fit_ridge(micro_tr, metab_tr, micro_va)
    meanA = np.repeat(metab_tr.mean(0, keepdims=True), len(metab_va), axis=0)
    rows.append(report(
        "A  microbiome -> metabolome", metab_va, predA, meanA,
        ev.evaluate_model_a(micro_va, metab_va, predA),
        ev.evaluate_model_a(micro_va, metab_va, meanA),
    ))

    # ---- Direction A' : microbiome (CLR) -> standardized metabolome --------- #
    # FAIR baseline for the transformer, which now sees CLR microbiome INPUT.
    # The plain-A row above uses raw (standardized) abundance, so comparing it to
    # the CLR transformer is apples-to-oranges. CLR lifted ridge B from 0.14 to
    # 0.21, so it may lift ridge A too -- if A' approaches the transformer's
    # ~0.17, the transformer is NOT adding value for direction A.
    micro_tr_clr_in = clr(micro_tr)
    micro_va_clr_in = clr(micro_va)
    predA_clr = fit_ridge(micro_tr_clr_in, metab_tr, micro_va_clr_in)
    rows.append(report(
        "A' microbiome (CLR) -> metabolome", metab_va, predA_clr, meanA,
        ev.evaluate_model_a(micro_va, metab_va, predA_clr),
        ev.evaluate_model_a(micro_va, metab_va, meanA),
    ))

    # ---- Direction B: standardized metabolome -> microbiome (raw frac) ------ #
    predB = fit_ridge(metab_tr, micro_tr, metab_va)
    meanB = np.repeat(micro_tr.mean(0, keepdims=True), len(micro_va), axis=0)
    rows.append(report(
        "B  metabolome -> microbiome (raw fraction)", micro_va, predB, meanB,
        ev.evaluate_model_b(micro_va, metab_va, predB),
        ev.evaluate_model_b(micro_va, metab_va, meanB),
    ))

    # ---- Direction B in CLR space (tests the compositional-loss idea) ------- #
    # Predict the microbiome in centered-log-ratio space, where a linear model
    # is far better matched to compositional data. This is the per-feature
    # ceiling that a CLR-loss transformer could aim for. (No Mantel here: the
    # space differs from the raw-fraction evaluator.)
    micro_tr_clr = clr(micro_tr)
    micro_va_clr = clr(micro_va)
    predB_clr = fit_ridge(metab_tr, micro_tr_clr, metab_va)
    meanB_clr = np.repeat(micro_tr_clr.mean(0, keepdims=True), len(micro_va_clr), axis=0)
    rows.append(report(
        "B' metabolome -> microbiome (CLR space)", micro_va_clr, predB_clr, meanB_clr,
    ))

    print("\n================ SUMMARY ================")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nRead: if 'per-feature mean r' is near 0 AND 'features with r>0.1' is")
    print("tiny in every direction, the linear ceiling is ~0 -> almost no signal.")
    print("If a chunk of features clear r>0.1/0.3, that signal is real and the")
    print("transformer is leaving it on the table.")


if __name__ == "__main__":
    main()
