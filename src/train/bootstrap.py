# -*- coding: utf-8 -*-
"""
bootstrap — final overall comparison. Four prior studies + LightGBM + OvCast + calendar rule.

Protocol
  · Validation   external only. Train on Hospital A (25 participants) -> test on Hospital B (18), fixed.
  · Input        every model uses the same base matrix. physio = 18 physiological-signal columns,
            combined = those 18 columns + 8 calendar columns.
  · Config       OvCast uses ovcast_config.FINAL (module set fixed by ablation).
            LightGBM and the four prior models use settings chosen on Hospital A validation.
  · Uncertainty  refit 1,000 times on resampled training participants. The test partition is fixed.
  · Metrics      macro-F1 primary, ov-F1 secondary. Predictions are argmax (rule models recover them from the estimated ovulation date).

  python3 src/train/bootstrap.py --n-boot 1000 --workers 24
Output -> results/bootstrap/
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # compute on CPU

import argparse, ast, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovcast_config import build as build_ovcast, fit_predict, LGBM  # noqa: E402
from model.prior import Goodale2019, Yu2022, Kilungeja2025, Maijala2019  # noqa: E402
from cohort import load, audit, macro, ovf1, PHYSIO, OV_CLS, LABELS  # noqa: E402
from features import SEED  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402

_rd = os.environ.get("OVUL_RESULT_DIR", "").strip()
OUT = Path(_rd) if _rd else ROOT / "results" / "bootstrap"
# Hyperparameters of the prior-study models used in the paper (frozen output of tune_prior_models.py)
TUNE_PR = Path(__file__).resolve().parent / "picked_cfgs.json"
BBT = PHYSIO.index("bbt")
PRIOR = ["Maijala2019", "Yu2022", "Goodale2019", "Kilungeja2025"]
MODELS = PRIOR + ["LightGBM", "OvCast"]
_S = {}


def init():
    _S["D"] = load()[1]
    _S["pr"] = json.loads(TUNE_PR.read_text()) if TUNE_PR.exists() else {}


def build_prior(name, lab, arm, seed):
    c = dict(_S["pr"].get(f"{name}|{int(lab[1])}-class|{arm}", {}))
    for k in ("cols", "wins"):
        if isinstance(c.get(k), str):
            c[k] = ast.literal_eval(c[k])
    if name == "Goodale2019":
        return Goodale2019(random_state=seed, **c)
    if name == "Yu2022":
        return Yu2022(random_state=seed, **c)
    if name == "Kilungeja2025":
        return Kilungeja2025(random_state=seed, **c)
    return Maijala2019(bbt_col=BBT, **c)


# Dump per-night predictions of the original fit (b=0). Read by the participant-level test (scripts/model_tests_testcohort.py).
# Leave empty to write nothing — default behaviour is unchanged.
_PD = os.environ.get("OVUL_DUMP_PRED_DIR", "").strip()


def dump_pred(b, lab, arm, nm, D, ite, y, pred):
    if not _PD or b != 0:
        return
    d = Path(_PD)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(idx=ite, g=D["g"][ite], y_true=y[ite], y_pred=pred)).to_csv(
        d / f"{lab}_{arm}_{nm}.csv", index=False)


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
    out = dict(boot=b, label=f"{nc}-class", arm=arm)
    for nm in MODELS:
        try:
            if nm == "OvCast":
                m, rule = build_ovcast(arm, lab, cd, SEED + b)
                pred = fit_predict(m, rule, X, cyc, cd, y, itr, ite)
            elif nm == "LightGBM":
                m = lgb.LGBMClassifier(random_state=SEED + b, n_jobs=1, verbose=-1,
                                       **LGBM[arm])
                m.fit(X[itr], y[itr])
                pred = np.asarray(m.predict(X[ite]))
            else:
                m = build_prior(nm, lab, arm, SEED + b)
                if nm == "Maijala2019":
                    m.set_structure(cyc[ite], cd[ite])
                    ov_hat = m.fit(X[itr], y[itr]).ovulation_day(X[ite])
                    pred = collapse(phase5(cd[ite], ov_hat), lab)
                else:
                    m.set_structure(cyc[itr], cd[itr])
                    m.fit(X[itr], y[itr])
                    m.set_structure(cyc[ite], cd[ite])
                    pred = np.asarray(m.predict(X[ite]))
            dump_pred(b, lab, arm, nm, D, ite, y, pred)
            out[f"{nm}_macro"] = macro(y[ite], pred, nc)
            out[f"{nm}_ov"] = ovf1(y[ite], pred, ovc)
        except Exception as e:
            out[f"{nm}_err"] = str(e)[:60]
    for rl, dv in (("cd14", 14.0), ("cd15", 15.0)):
        pr = collapse(phase5(cd[ite], np.full(len(ite), dv)), lab)
        out[f"{rl}_macro"] = macro(y[ite], pr, nc)
        out[f"{rl}_ov"] = ovf1(y[ite], pr, ovc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    # Allow selecting which chunks (label x config) to run. The sensitivity check only needs 4-class,
    # and running all eight chunks takes four times longer.
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--arms", default="physio,combined")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, D = load()
    audit(d, D)
    _labs = [x.strip() for x in a.labels.split(",") if x.strip()]
    _arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert set(_labs) <= set(LABELS), f"unknown labels: {set(_labs) - set(LABELS)}"
    cells = [(lab, arm) for lab in _labs for arm in _arms]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init) as ex:
        for i, (lab, arm) in enumerate(cells, 1):
            got = [r for r in ex.map(one, [(b, lab, arm) for b in range(a.n_boot)],
                                     chunksize=2) if r]
            rows += got
            pd.DataFrame(rows).to_csv(OUT / "boot.csv", index=False)
            s = pd.DataFrame(got)
            errs = {c: s[c].dropna().iloc[0] for c in s.columns
                    if c.endswith("_err") and s[c].notna().any()}
            if errs:
                print("   error:", errs, flush=True)
            print(f"[{time.time()-t0:6.0f}s] {i}/{len(cells)} {lab} {arm:9s}  " +
                  "  ".join(f"{m}={s[m+'_macro'].mean():.3f}"
                            for m in MODELS if m + "_macro" in s), flush=True)
    print(f"\n-> {OUT}/boot.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
