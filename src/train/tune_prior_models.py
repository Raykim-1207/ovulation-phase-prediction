# -*- coding: utf-8 -*-
"""
tune_prior_models — choose the **uncertain parameters** of the prior models within Hospital A.

All four papers leave some details that cannot be read from the text. Fixing those
arbitrarily would let our assumptions drive the prior work's scores — choosing them
unfavourably would make our model's advantage an artifact of implementation mistakes.
So uncertain spots get several candidates, chosen by **exactly the same procedure as
our model**: selected on participant-level validation splits of the 25 Hospital A
participants, with Hospital B never seen during tuning.

  [1] Goodale    max_features ratio (around the paper's 3/11)
  [2] Yu         input signal set (BBT·HR / + ln SDNN·ln LF-HF), lag window length
  [3] Kilungeja  moving-window size set, number of trees
  [4] Maijala    interpretation of 0.15 degrees (phase/baseline/cover), threshold, baseline, consecutive days

Values stated in the papers (e.g. Goodale's 1,000 trees, Maijala's 0.15 degrees) sit at the
**center** of the candidates, with neighbours tested too. If the stated value is best, it is picked.

  python3 src/train/tune_prior_models.py --workers 24
Output -> results/prior_tune/
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Keep xgboost's CUDA context from grabbing GPU memory (compute runs on CPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.prior import Goodale2019, Yu2022, Kilungeja2025, Maijala2019  # noqa: E402
from cohort import load, audit, macro, PHYSIO, OV_CLS, LABELS  # noqa: E402
from features import SEED  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402

OUT = ROOT / "results" / "prior_tune"
REPEATS = 6
BBT, HR = PHYSIO.index("bbt"), PHYSIO.index("hr")
LNS, LNL = PHYSIO.index("ln_sdnn"), PHYSIO.index("ln_lfhf")
_S = {}


def candidates():
    """Candidates for each model's uncertain parameters, centered on the paper's stated values."""
    return {
        "Goodale2019": [dict(ratio=r, n_estimators=1000)
                        for r in (3 / 11, 0.15, 0.4, 0.6, 1.0)],
        "Yu2022": [dict(cols=c, lag=l)
                   for c in ((BBT, HR), (BBT, HR, LNS, LNL))
                   for l in (2, 3, 5, 7)],
        "Kilungeja2025": [dict(wins=w, n_estimators=n)
                          for w in ((1,), (1, 3), (1, 3, 5), (1, 3, 5, 7), (3, 7))
                          for n in (200, 500)],
        "Maijala2019": [dict(mode=m, thr=t, base=b, consec=c)
                        for m in ("phase", "baseline", "cover")
                        for t in (0.10, 0.15, 0.20, 0.30)
                        for b in (3, 6)
                        for c in (2, 3)],
    }


def init():
    _S["D"] = load()[1]


def build(name, cfg, seed):
    if name == "Goodale2019":
        return Goodale2019(random_state=seed, **cfg)
    if name == "Yu2022":
        return Yu2022(random_state=seed, **cfg)
    if name == "Kilungeja2025":
        return Kilungeja2025(random_state=seed, **cfg)
    return Maijala2019(bbt_col=BBT, **cfg)


def one(job):
    """One candidate × one validation split. Hospital B is never touched."""
    name, ci, cfg, lab, arm, rep = job
    D = _S["D"]
    nc = int(lab[1])
    y = collapse(phase5(D["cd"], D["ovt"]), lab)
    X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
    g, itr_all = D["g"], np.where(D["is_A"])[0]
    rng = np.random.RandomState(SEED + rep)
    pats = np.unique(g[itr_all])
    val = set(rng.choice(pats, max(2, len(pats) // 4), replace=False))
    mv = np.array([p in val for p in g[itr_all]])
    ifit, ival = itr_all[~mv], itr_all[mv]
    if len(np.unique(y[ifit])) < nc:
        return None
    try:
        m = build(name, cfg, SEED + rep)
        if name == "Maijala2019":
            m.set_structure(D["cyc"][ival], D["cd"][ival])
            ov_hat = m.fit(X[ifit], y[ifit]).ovulation_day(X[ival])
            pred = collapse(phase5(D["cd"][ival], ov_hat), lab)
        else:
            m.set_structure(D["cyc"][ifit], D["cd"][ifit])
            m.fit(X[ifit], y[ifit])
            m.set_structure(D["cyc"][ival], D["cd"][ival])
            pred = np.asarray(m.predict(X[ival]))
        return dict(model=name, cfg_i=ci, label=f"{nc}-class", arm=arm, rep=rep,
                    val_macro=macro(y[ival], pred, nc))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, D = load()
    audit(d, D)
    CAND = candidates()
    jobs = [(nm, i, CAND[nm][i], lab, arm, rep)
            for nm in CAND for i in range(len(CAND[nm]))
            for lab in LABELS for arm in ("physio", "combined")
            for rep in range(REPEATS)]
    print("candidates:", {k: len(v) for k, v in CAND.items()}, flush=True)
    print(f"tuning jobs: {len(jobs)} (within Hospital A only)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=4) if r]
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "val_runs.csv", index=False)
    picked = {}
    for (nm, lab, arm), s in r.groupby(["model", "label", "arm"]):
        v = s.groupby("cfg_i").val_macro.mean()
        ci = int(v.idxmax())
        picked[f"{nm}|{lab}|{arm}"] = CAND[nm][ci]
        print(f"  {nm:14s} {lab} {arm:9s} -> {CAND[nm][ci]}  (val {v.max():.4f})",
              flush=True)
    (OUT / "picked_cfgs.json").write_text(json.dumps(picked, indent=1, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
