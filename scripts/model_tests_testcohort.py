# -*- coding: utf-8 -*-
"""model_tests_testcohort — test OvCast against the nine other models at the level of test participants.

Same method as the input-configuration comparison (vis_fig2_testcohort.py).
  Predictions  per-night predictions of the original fit (b=0). CPU models from pred_models/ (bootstrap dump),
            deep models from PRED_DEEP (default pred_models_gpu/, the original fit made on GPU 3·4 — metrics
            exactly match b=0 of the reported boot_sota·boot_sota2. pred_models_cpu/ is a CPU refit).
  P         participant-level exact paired permutation (2^18)
  CI        participant-cluster bootstrap, 10,000 resamples
  Multiplicity  BH-FDR per comparison pair (OvCast vs model X, one input configuration), family = 4 granularities x 2 metrics

  python3 scripts/model_tests_testcohort.py
  python3 scripts/model_tests_testcohort.py --selfcheck
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))
import vis_fig2_testcohort as V                                        # noqa: E402

PRED_CPU = PAPER / "pred_models"
PRED_DEEP = Path(os.environ.get("OVUL_PRED_DEEP", "").strip() or PAPER / "pred_models_gpu")
CPU_MODELS = ("Yu2022", "Goodale2019", "Kilungeja2025", "LightGBM")
DEEP_MODELS = ("TabM", "Mamba", "KAN", "RealMLP", "ExcelFormer")
OUT = V.FIG / ("model_tests_testcohort" + os.environ.get("OVUL_OUT_SUFFIX", "") + ".csv")


def preds(lab, arm, nm):
    d = (PRED_DEEP if nm in DEEP_MODELS else PRED_CPU) / f"{lab}_{arm}_{nm}.csv"
    return pd.read_csv(d)


def confusions(lab, arm, a, b):
    """Per-participant confusion matrices for two models. First checks that both point to the same test rows."""
    k = int(lab[1])
    A, B = preds(lab, arm, a), preds(lab, arm, b)
    assert A[["idx", "g", "y_true"]].equals(B[["idx", "g", "y_true"]]), (lab, arm, a, b)
    pids = np.unique(A.g)
    ix = np.searchsorted(pids, A.g.to_numpy())
    out = []
    for d in (A, B):
        c = np.zeros((len(pids), k, k))
        np.add.at(c, (ix, d.y_true.to_numpy(), d.y_pred.to_numpy()), 1)
        out.append(c)
    return out


def analyse():
    rng = np.random.default_rng(V.SEED)
    rows = []
    for nm in CPU_MODELS + DEEP_MODELS:
        for arm in ("physio", "combined"):
            for k in V.CLASSES:
                lab = f"y{k}_std"
                A, B = confusions(lab, arm, "OvCast", nm)
                P = len(A)
                sw, exact = V.swap_matrix(P, rng)
                sw = sw.astype(float)
                W = rng.multinomial(P, np.full(P, 1 / P), size=V.B).astype(float)
                Af, Bf = A.reshape(P, -1), B.reshape(P, -1)
                for met in ("macro", "ov"):
                    f = V.metric(met, k)
                    d0 = f(A.sum(0)) - f(B.sum(0))
                    dd = (f((sw @ Bf + (1 - sw) @ Af).reshape(-1, k, k))
                          - f((sw @ Af + (1 - sw) @ Bf).reshape(-1, k, k)))
                    lo, hi = np.quantile(f(np.einsum("bp,pij->bij", W, A))
                                         - f(np.einsum("bp,pij->bij", W, B)), [.025, .975])
                    rows.append(dict(model=nm, arm=arm, classes=k, metric=met,
                                     ovcast=f(A.sum(0)), other=f(B.sum(0)), diff=d0,
                                     lo=lo, hi=hi, p=V.perm_p(dd, d0, exact),
                                     source="cpu" if nm in CPU_MODELS else PRED_DEEP.name))
    t = pd.DataFrame(rows)
    for _, g in t.groupby(["model", "arm"]):
        o = np.argsort(g.p.to_numpy(), kind="stable")[::-1]
        q = g.p.to_numpy()[o] * len(g) / np.arange(len(g), 0, -1)
        t.loc[g.index[o], "p_fdr"] = np.minimum(1, np.minimum.accumulate(q))
    return t


def selfcheck():
    """Does the OvCast prediction in the dump match pred_nights (the original fit used for figures and tests)?"""
    for k in V.CLASSES:
        for arm, f in (("physio", "physio"), ("combined", "combined")):
            a = preds(f"y{k}_std", arm, "OvCast")
            b = pd.read_csv(V.PRED / f"y{k}_std_{f}.csv")
            assert len(a) == len(b) and (a.y_true.values == b.y_true.values).all()
            assert (a.y_pred.values == b.y_pred.values).all(), (k, arm)
    print("selfcheck OK: OvCast dump == pred_nights")


def main():
    if "--selfcheck" in sys.argv:
        return selfcheck()
    t = analyse()
    t.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    s = t[t.classes == 4].round(4)
    print(s.to_string(index=False))
    print(f"\nq<0.05: {(t.p_fdr < .05).sum()} / {len(t)}  "
          f"(OvCast better {((t.p_fdr < .05) & (t['diff'] > 0)).sum()}, "
          f"worse {((t.p_fdr < .05) & (t['diff'] < 0)).sum()})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
