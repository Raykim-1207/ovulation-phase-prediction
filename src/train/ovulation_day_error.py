# -*- coding: utf-8 -*-
"""
ovulation_day_error — day-level ovulation_date error and fertile-window capture rate.

macro-F1 is not clinically readable. "How many days off" and "what % of fertile nights
were caught" are what users actually experience. The manuscript currently lists this in Limitations
as not evaluated.

Protocol matches bootstrap — fixed train on Hospital A -> test on Hospital B, same base
matrix, OvCast is ovcast_config.FINAL, B resamples of training participants with replacement.

The ovulation_date estimation rule is **owned solely by day_error.py**. Here its constants are
imported, and at b=0 the fast numpy path for the bootstrap is checked exhaustively against
day_error.estimate (--selfcheck). If the rule diverges between the two places, that
check breaks.

  python3 src/train/ovulation_day_error.py --n-boot 1000 --workers 32
Output -> results/ovulation_day_error/boot.csv, scripts/pred_nights/*.csv
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # compute on CPU

import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PAPER))
from ovcast_config import build as build_ovcast, fit_predict, RULE_DAY  # noqa: E402
from cohort import load, audit, OV_CLS, LABELS, FINAL  # noqa: E402
from features import SEED  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402
from day_error import BAND_MID, FERTILE, estimate  # noqa: E402  sole owner of the rule

_rd = os.environ.get("OVUL_RESULT_DIR", "").strip()
OUT = Path(_rd) if _rd else ROOT / "results" / "ovulation_day_error"
DUMP = PAPER / "pred_nights"
FAMILY, METHOD = FINAL.parts[-2], FINAL.parts[-1]
SPLIT = "A->B"
ARMS = ["cal", "physio", "combined"]
_S = {}


def init():
    d, D = load()
    # measurement date - estimated ovulation date = offset relative to ovulation. Ground truth comes only from here.
    D["off"] = ((pd.to_datetime(d["measure_date"]) - pd.to_datetime(d["ovulation_date"]))
                .dt.days.to_numpy(float))
    D["pid"] = d["participant_id"].astype(str).to_numpy()
    D["cycno"] = d["cycle"].astype(str).to_numpy()
    D["day"] = pd.to_datetime(d["measure_date"]).dt.strftime("%Y-%m-%d").to_numpy()
    _S["D"] = D


def day_metrics(off, pred, cyckey, ovc):
    """Per-cycle estimated ovulation_date error and fertile-window capture rate.

    Pre-specified rule: estimated ovulation_date = median(offsets of nights predicted as peri-ovulatory) - BAND_MID.
    The peri-ovulatory window -4..+1 has midpoint -1.5, so shift back by that amount.
    """
    errs, n_ne, cover = [], 0, []
    for key in np.unique(cyckey):
        m = cyckey == key
        o, p = off[m], pred[m]
        hit = o[p == ovc]
        if hit.size == 0:
            n_ne += 1
        else:
            errs.append(abs(float(np.median(hit)) - BAND_MID))
        fw = np.isin(o, list(FERTILE))
        if fw.any():
            cover.append(float((p[fw] == ovc).mean()))
    a = np.asarray(errs, float)
    return dict(
        n_cycles=int(np.unique(cyckey).size), n_not_estimable=n_ne,
        mae_days=float(a.mean()) if a.size else np.nan,
        median_ae=float(np.median(a)) if a.size else np.nan,
        within1=float((a <= 1).mean()) if a.size else np.nan,
        within2=float((a <= 2).mean()) if a.size else np.nan,
        fertile_capture=float(np.mean(cover)) if cover else np.nan,
    )


def predict(b, lab, arm, D):
    """Test-set predictions for one replicate. arm='cal' is a deterministic rule with no fitting."""
    g, cd, cyc = D["g"], D["cd"], D["cyc"]
    nc = int(lab[1])
    y = collapse(phase5(cd, D["ovt"]), lab)
    ite = np.where(~D["is_A"])[0]
    if arm == "cal":
        return ite, y, collapse(phase5(cd[ite], np.full(len(ite), RULE_DAY)), lab)
    X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
    pats = np.unique(g[D["is_A"]])
    rng = np.random.RandomState(SEED + b)
    draw = pats if b == 0 else rng.choice(pats, len(pats), replace=True)
    itr = np.concatenate([np.where(g == p)[0] for p in draw])
    if len(np.unique(y[itr])) < nc:
        return None, None, None
    m, rule = build_ovcast(arm, lab, cd, SEED + b)
    return ite, y, fit_predict(m, rule, X, cyc, cd, y, itr, ite)


def one(job):
    b, lab, arm = job
    D = _S["D"]
    ite, y, pred = predict(b, lab, arm, D)
    if pred is None:
        return None
    r = day_metrics(D["off"][ite], pred, D["cyc"][ite], OV_CLS[lab])
    r.update(boot=b, label=f"{int(lab[1])}-class", arm=arm,
             model="cd14 rule" if arm == "cal" else "OvCast")
    return r


def dump_nights(D):
    """Dump b=0 per-night predictions in the form day_error.py reads."""
    DUMP.mkdir(parents=True, exist_ok=True)
    for f in DUMP.glob("*.csv"):
        f.unlink()
    for lab in LABELS:
        for arm in ARMS:
            ite, y, pred = predict(0, lab, arm, D)
            pd.DataFrame(dict(
                family=FAMILY, method=METHOD, label=lab, arm=arm,
                model="cd14 rule" if arm == "cal" else "OvCast", split=SPLIT,
                pid=D["pid"][ite], cyc=D["cycno"][ite], day=D["day"][ite],
                y_true=y[ite], y_pred=pred,
            )).to_csv(DUMP / f"{lab}_{arm}.csv", index=False)
    print(f"-> {DUMP}  ({len(list(DUMP.glob('*.csv')))} files)", flush=True)


def selfcheck(D):
    """Does the fast path give the same values as day_error.estimate (sole owner of the rule)?"""
    bad = 0
    for lab in LABELS:
        ite, y, pred = predict(0, lab, "combined", D)
        off, cyckey, ovc = D["off"][ite], D["cyc"][ite], OV_CLS[lab]
        for key in np.unique(cyckey):
            m = cyckey == key
            hit = off[m][pred[m] == ovc]
            mine = float(np.median(hit)) - BAND_MID if hit.size else np.nan
            theirs = estimate(pd.DataFrame(dict(y_pred=pred[m], offset=off[m])),
                              ovc, "median")
            if not (np.isnan(mine) and np.isnan(theirs)) and \
               not np.isclose(mine, theirs):
                bad += 1
    assert bad == 0, f"fast path disagrees with day_error.estimate on {bad} cycles"
    print("selfcheck: fast path == day_error.estimate  (all cycles match)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, D = load()
    audit(d, D)
    init()
    D = _S["D"]
    selfcheck(D)
    dump_nights(D)

    cells = [(lab, arm) for lab in LABELS for arm in ("physio", "combined")]
    t0, rows = time.time(), [one((0, lab, "cal")) for lab in LABELS]
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init) as ex:
        for i, (lab, arm) in enumerate(cells, 1):
            got = [r for r in ex.map(one, [(b, lab, arm) for b in range(a.n_boot)],
                                     chunksize=2) if r]
            rows += got
            pd.DataFrame(rows).to_csv(OUT / "boot.csv", index=False)
            s = pd.DataFrame(got)
            print(f"[{time.time()-t0:6.0f}s] {i}/{len(cells)} {lab} {arm:9s} "
                  f" MAE={s.mae_days.mean():.2f}d  within1={s.within1.mean():.3f}"
                  f"  fertile={s.fertile_capture.mean():.3f}", flush=True)
    print(f"\n-> {OUT}/boot.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
