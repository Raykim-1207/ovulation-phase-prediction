# -*- coding: utf-8 -*-
"""
grid_bootstrap — rescore the fifteen feature cells with OvCast + patient-level bootstrap.

Table 1 previously came from the grid (median of four off-the-shelf learners, single split). Tables 2
and 3 beside it use OvCast and 1,000 bootstrap replicates. Mixing the two invites readers to compare
them, so the fifteen cells are re-measured on the same yardstick as Tables 2 and 3.

Feature column names differ by design (b 18 columns, d 16 columns, c 45 columns…), so columns are
read from the file. Meta columns and **calendar-derived columns** are dropped — the oura family carries
cycle_day (= day_of_cycle) as a feature, and leaving it in would break the physiology-only arm.

Protocol matches Tables 2 and 3: train on Hospital A -> test on Hospital B, 1,000 replicates, 4-class.

  python3 src/train/grid_bootstrap.py --family interp_huawei24mid \
      --method locf --n-boot 1000 --workers 24
Output -> results/grid_boot/<family>__<method>.csv
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
from ovcast_config import build as build_ovcast, fit_predict          # noqa: E402
from cohort import macro, ovf1, OV_CLS, cal_eng             # noqa: E402
from features import SEED                                        # noqa: E402
from phase_labels import phase5, collapse                       # noqa: E402

LAB = "y4_std"          # pre-specified primary granularity, used by Table 1
DATA = ROOT / "ovul_data" / "finalfinal"
OUT = ROOT / "results" / "grid_boot"

META = {"participant_id", "file_name_x", "cycle", "measure_date", "ecg_start", "used_from",
        "used_to", "used_min", "record_min", "label_date", "day_of_cycle", "menses_start",
        "next_menses_start", "ovulation_date", "bbt_raw", "temp_day_imputed",
        "temp_day_method", "n_hrv_win", "n_temp_pts", "m_cycle", "m_cycle_group",
        "window", "temp_mask", "temp_slot_imputed"}
_S = {}


def feature_cols(d: pd.DataFrame) -> list[str]:
    """Feature columns. Drop meta and label columns, and columns identical to the calendar.

    In the oura family, cycle_day is day_of_cycle itself. If it entered the physiology-only arm, that arm
    would be receiving the calendar, breaking the paper's central comparison. Masking is by value, not
    name — so a same-valued column under another name is caught too.
    """
    # OVUL_KEEP_CAL=1 disables masking. Control-only, to test whether design d topping the
    # earlier grid was due to the cycle-day column — not used for the main results.
    keep = os.environ.get("OVUL_KEEP_CAL") == "1"
    cd = pd.to_numeric(d["day_of_cycle"], errors="coerce")
    out, dropped = [], []
    for c in d.columns:
        if c in META or str(c).startswith("y"):
            continue
        v = pd.to_numeric(d[c], errors="coerce")
        m = v.notna() & cd.notna()
        if m.sum() >= 10 and np.allclose(v[m], cd[m]):
            dropped.append(c)
            if not keep:
                continue
        out.append(c)
    if dropped:
        print(f"  calendar-identical{' (kept)' if keep else ' (dropped)'}: {dropped}", flush=True)
    return out


def load(family: str, method: str):
    p = DATA / family / method / f"{LAB}.xlsx"
    assert p.is_file(), f"missing: {p}"
    d = pd.read_excel(p, engine="openpyxl")
    d.columns = [str(c).strip() for c in d.columns]
    d["day_of_cycle"] = pd.to_numeric(d["day_of_cycle"], errors="coerce")
    d = d.dropna(subset=["day_of_cycle", LAB]).sort_values(
        ["participant_id", "cycle", "day_of_cycle"], kind="mergesort").reset_index(drop=True)
    cols = feature_cols(d)
    g = d["participant_id"].astype(str).to_numpy()
    cd = d["day_of_cycle"].to_numpy(float)
    ovt = ((pd.to_datetime(d["ovulation_date"]) - pd.to_datetime(d["menses_start"])).dt.days
           + 1).to_numpy(float)
    Xp = np.nan_to_num(d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    cyc = (g + "|" + d["cycle"].astype(str)).to_numpy()
    print(f"  nights {len(d)} · features {len(cols)} columns · patients {len(set(g))}", flush=True)
    return dict(g=g, cd=cd, ovt=ovt, cyc=cyc, Xp=Xp, C=cal_eng(cd), ncol=len(cols),
                is_A=np.char.startswith(g.astype(str), "S"))


def audit(D):
    """Leakage checks this script can run. Derived columns differ by design, so the b-specific
    causal-formula comparison is not possible, but the two below must hold for any design."""
    itr, ite = np.where(D["is_A"])[0], np.where(~D["is_A"])[0]
    ov = set(D["g"][itr]) & set(D["g"][ite])
    assert not ov, f"patients span train and test: {ov}"
    assert not (set(itr) & set(ite)), "train and test indices overlap"


def init(args):
    _S["D"] = load(args[0], args[1])
    _S["n"] = args[2]


def one(job):
    b, arm = job
    D = _S["D"]
    g, cd, cyc = D["g"], D["cd"], D["cyc"]
    nc, ovc = int(LAB[1]), OV_CLS[LAB]
    y = collapse(phase5(cd, D["ovt"]), LAB)
    X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
    ite = np.where(~D["is_A"])[0]
    pats = np.unique(g[D["is_A"]])
    rng = np.random.RandomState(SEED + b)
    draw = pats if b == 0 else rng.choice(pats, len(pats), replace=True)
    itr = np.concatenate([np.where(g == p)[0] for p in draw])
    if len(np.unique(y[itr])) < nc:
        return None
    out = dict(boot=b, arm=arm)
    try:
        # summary_cols is hard-wired to 18 columns. Column counts differ by design, so adjust.
        m, rule = build_ovcast(arm, LAB, cd, SEED + b,
                               summary_cols=list(range(D["ncol"])))
        pred = fit_predict(m, rule, X, cyc, cd, y, itr, ite)
        out["OvCast_macro"] = macro(y[ite], pred, nc)
        out["OvCast_ov"] = ovf1(y[ite], pred, ovc)
    except Exception as e:
        out["OvCast_err"] = str(e)[:80]
    pr = collapse(phase5(cd[ite], np.full(len(ite), 14.0)), LAB)
    out["cd14_macro"] = macro(y[ite], pr, nc)
    out["cd14_ov"] = ovf1(y[ite], pr, ovc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--method", default="locf")
    ap.add_argument("--arms", default="combined")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    print(f"== {a.family}/{a.method} ==", flush=True)
    D = load(a.family, a.method)
    audit(D)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init,
                             initargs=((a.family, a.method, a.n_boot),)) as ex:
        for arm in arms:
            got = [r for r in ex.map(one, [(b, arm) for b in range(a.n_boot)],
                                     chunksize=4) if r]
            rows += got
            s = pd.DataFrame(got)
            if "OvCast_err" in s and s.OvCast_err.notna().any():
                print("   error:", s.OvCast_err.dropna().iloc[0], flush=True)
            print(f"  [{time.time()-t0:5.0f}s] {arm:9s} ov-F1={s.OvCast_ov.mean():.4f} "
                  f"macro={s.OvCast_macro.mean():.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "__keepcal" if os.environ.get("OVUL_KEEP_CAL") == "1" else ""
    f = OUT / f"{a.family}__{a.method}{tag}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"-> {f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
