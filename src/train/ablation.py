# -*- coding: utf-8 -*-
"""
ablation — module ablation of the final OvCast. Bootstrap the contribution of each module.

Flip one switch at a time from the final configuration (ovcast_config.FINAL). Resample
training participants with replacement and refit 1,000 times; pair with the full configuration
**within the same replicate** to get differences. Testing is fixed to Hospital B.

  full          final configuration
  no_ring       ① sigma = 0            (one-hot targets)
  no_bal        ② beta = 0             (no sample weighting)
  argmax        ③ decision = argmax    (instead of ring Bayes decision)
  no_summary    ④ cycle_summary = False
  summary_full  ④ summary_mode = full  (include absolute levels too)
  no_twostage   ⑤ two_stage = False
  no_prior      ⑦ prior_strength = 0

⑤⑦ only act in configurations with the calendar, so under physio those two rows come out
equal to full — they are reported as is (evidence that they switched off automatically).

  python3 src/train/ablation.py --n-boot 1000 --workers 24
Output -> results/ablation/
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Compute on CPU. Keep xgboost from grabbing GPU memory just by being imported.
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
from cohort import load, audit, macro, ovf1, OV_CLS, LABELS  # noqa: E402
from features import SEED  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402

_rd = os.environ.get("OVUL_RESULT_DIR", "").strip()
OUT = Path(_rd) if _rd else ROOT / "results" / "ablation"
VARIANTS = {
    "full": {},
    "no_ring": dict(sigma=0.0),
    "no_bal": dict(beta=0.0),
    "argmax": dict(decision="argmax"),
    "no_summary": dict(cycle_summary=False),
    "summary_full": dict(summary_mode="full"),
    "no_twostage": dict(two_stage=False),
    "no_prior": dict(prior_strength=0.0),
}
_S = {}


def init():
    _S["D"] = load()[1]


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
    for v, over in VARIANTS.items():
        try:
            m, rule = build(arm, lab, cd, SEED + b, **over)
            pred = fit_predict(m, rule, X, cyc, cd, y, itr, ite)
            out[f"{v}_macro"] = macro(y[ite], pred, nc)
            out[f"{v}_ov"] = ovf1(y[ite], pred, ovc)
        except Exception as e:
            out[f"{v}_err"] = str(e)[:60]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, D = load()
    audit(d, D)
    cells = [(lab, arm) for lab in LABELS for arm in ("physio", "combined")]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init) as ex:
        for i, (lab, arm) in enumerate(cells, 1):
            got = [r for r in ex.map(one, [(b, lab, arm) for b in range(a.n_boot)],
                                     chunksize=2) if r]
            rows += got
            pd.DataFrame(rows).to_csv(OUT / "abl.csv", index=False)
            s = pd.DataFrame(got)
            print(f"[{time.time()-t0:6.0f}s] {i}/{len(cells)} {lab} {arm:9s} "
                  f"full={s.full_macro.mean():.4f}  " +
                  "  ".join(f"{v}={s[v+'_macro'].mean():.4f}"
                            for v in VARIANTS if v != "full" and v + "_macro" in s),
                  flush=True)
    r = pd.DataFrame(rows)
    print("\n=== Paired difference vs. full configuration (macro-F1) ===")
    for arm in ("combined", "physio"):
        for lab in ("2-class", "3-class", "4-class", "5-class"):
            s = r[(r.arm == arm) & (r.label == lab)]
            if not len(s):
                continue
            print(f"\n  {arm} {lab}  full {s.full_macro.mean():.4f}")
            for v in VARIANTS:
                if v == "full":
                    continue
                dd = s[f"{v}_macro"] - s["full_macro"]
                star = " *" if dd.quantile(.975) < 0 else ""
                print(f"    {v:13s} {s[v+'_macro'].mean():.4f}  "
                      f"d {dd.mean():+.4f} "
                      f"[{dd.quantile(.025):+.4f},{dd.quantile(.975):+.4f}]"
                      f"  worse {(dd < 0).mean()*100:5.1f}%{star}")
    print(f"\n* = significantly worse when off\n-> {OUT}/abl.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
