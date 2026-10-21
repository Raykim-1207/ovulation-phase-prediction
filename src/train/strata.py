# -*- coding: utf-8 -*-
"""
strata — performance stratified by cycle regularity, for the final model.

Test nights are split into three strata by measured cycle length (same boundaries as Methods).

    regular             27--32 days
    mildly irregular    24--26 or 33--35 days
    highly irregular    <=23 or >=36 days

**Strata only partition test nights; they play no part in training.** The model is fit once and
predictions are re-aggregated per stratum — training per stratum would let differences in sample
size leak into performance differences. Uncertainty uses the same patient-level bootstrap as other experiments.

Strata with fewer than 30 nights get no point estimate and are reported as not estimable.

  python3 src/train/strata.py --n-boot 1000 --workers 24
Output -> results/strata/
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovcast_config import build, fit_predict  # noqa: E402
from cohort import load, audit, macro, ovf1, OV_CLS, LABELS, FINAL  # noqa: E402
from features import SEED  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402

_rd = os.environ.get("OVUL_RESULT_DIR", "").strip()
OUT = Path(_rd) if _rd else ROOT / "results" / "strata"
MIN_NIGHTS = 30
_S = {}


def strata_of(cyc_len):
    """Same boundaries as Methods. Split by measured cycle length."""
    s = np.full(len(cyc_len), "highly_irregular", dtype=object)
    s[(cyc_len >= 27) & (cyc_len <= 32)] = "regular"
    s[((cyc_len >= 24) & (cyc_len <= 26)) |
      ((cyc_len >= 33) & (cyc_len <= 35))] = "mildly_irregular"
    return s


def init():
    d, D = load()
    cl = (pd.to_datetime(d["next_menses_start"]) -
          pd.to_datetime(d["menses_start"])).dt.days.to_numpy(float)
    D["stratum"] = strata_of(cl)
    _S["D"] = D


def one(job):
    b, lab, arm = job
    D = _S["D"]
    g, cd, cyc = D["g"], D["cd"], D["cyc"]
    nc, ovc = int(lab[1]), OV_CLS[lab]
    y = collapse(phase5(cd, D["ovt"]), lab)
    X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
    ite = np.where(~D["is_A"])[0]
    pats = np.unique(g[D["is_A"]])
    rng = np.random.RandomState(SEED + b)
    draw = pats if b == 0 else rng.choice(pats, len(pats), replace=True)
    itr = np.concatenate([np.where(g == p)[0] for p in draw])
    if len(np.unique(y[itr])) < nc:
        return None
    m, rule = build(arm, lab, cd, SEED + b)
    pred = fit_predict(m, rule, X, cyc, cd, y, itr, ite)
    cal = collapse(phase5(cd[ite], np.full(len(ite), 14.0)), lab)
    st = D["stratum"][ite]
    out = dict(boot=b, label=f"{nc}-class", arm=arm)
    for s_ in ("regular", "mildly_irregular", "highly_irregular"):
        k = st == s_
        if k.sum() < MIN_NIGHTS:
            continue                       # not estimable — no point estimate
        out[f"{s_}_n"] = int(k.sum())
        out[f"{s_}_model_macro"] = macro(y[ite][k], pred[k], nc)
        out[f"{s_}_cal_macro"] = macro(y[ite][k], cal[k], nc)
        out[f"{s_}_model_ov"] = ovf1(y[ite][k], pred[k], ovc)
        out[f"{s_}_cal_ov"] = ovf1(y[ite][k], cal[k], ovc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, D = load()
    audit(d, D)
    cl = (pd.to_datetime(d["next_menses_start"]) -
          pd.to_datetime(d["menses_start"])).dt.days.to_numpy(float)
    st = strata_of(cl)[~D["is_A"]]
    print("test-night strata:", {k: int((st == k).sum())
                          for k in np.unique(st)}, flush=True)
    cells = [(lab, arm) for lab in LABELS for arm in ("physio", "combined")]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init) as ex:
        for i, (lab, arm) in enumerate(cells, 1):
            got = [r for r in ex.map(one, [(b, lab, arm) for b in range(a.n_boot)],
                                     chunksize=2) if r]
            rows += got
            pd.DataFrame(rows).to_csv(OUT / "strata.csv", index=False)
            print(f"[{time.time()-t0:6.0f}s] {i}/{len(cells)} {lab} {arm}", flush=True)
    r = pd.DataFrame(rows)
    print("\n=== macro-F1 by stratum (combined set, model vs calendar rule) ===")
    for lab in ("2-class", "3-class", "4-class", "5-class"):
        s = r[(r.arm == "combined") & (r.label == lab)]
        line = [f"  {lab}"]
        for st_ in ("regular", "mildly_irregular"):
            if f"{st_}_model_macro" not in s:
                continue
            dd = s[f"{st_}_model_macro"] - s[f"{st_}_cal_macro"]
            line.append(f"{st_}: cal {s[f'{st_}_cal_macro'].mean():.3f} -> "
                        f"model {s[f'{st_}_model_macro'].mean():.3f} "
                        f"({dd.mean():+.3f} [{dd.quantile(.025):+.3f},"
                        f"{dd.quantile(.975):+.3f}])")
        print("   ".join(line))
    print(f"\n-> {OUT}/strata.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
