# -*- coding: utf-8 -*-
"""
Path ④ : npj Women's Health 2025 (wrist multimodal) style.

Paper
----
https://www.nature.com/articles/s44294-025-00078-8
  - Empatica E4 / EmbracePlus: temp, HR, IBI, EDA (during sleep)
  - **fixed-size non-overlapping windows** → daily features
  - **rolling/sliding window** (deployment) → segment statistics
  - leave-last-cycle-out evaluation is in the training stage

Limitations of our data
----
  - no EDA
  - paper IBI → in our data, ECG avgNN (RR mean), same notation
  - sleep segment = ECG recording segment (30 min trimmed at sleep onset/wake)

Output
----
  ovul_data/pre/ovul_data_npj.xlsx  (one row per night, fixed-block + global statistics)
  Requires: ``python _raw_common.py`` QC keep only (run_mapping_loop)
  → 01 → 02(+enrich)

Pipeline: 00 (this) → 01 → 02 (+enrich).
"""
from __future__ import annotations

from datetime import timedelta
import os
from typing import Optional

import numpy as np
import pandas as pd

from _raw_common import (
    FS,
    OUT_DIR,
    PRE_DIR,
    SEG_MIN,
    SAMPLES_10MIN,
    bandpass_ecg,
    hrv_time,
    load_ecg,
    load_temp_json,
    resolve_used_span,
    rr_from_filtered,
    run_mapping_loop,
    temp_in_span,
)

# reduce sleep-edge noise (paper uses the whole sleep segment; we trim 30/30)
TRIM_HEAD_MIN = 30
TRIM_TAIL_MIN = 30
N_BLOCKS = 4  # fixed windows within night
FLATLINE_FRAC = 0.5

OUT_PATH = PRE_DIR / "ovul_data_npj.xlsx"
TAG = "npj"


def _nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")


def _nanstd(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanstd(a)) if np.sum(np.isfinite(a)) >= 2 else float("nan")


def _nanmin(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmin(a)) if np.any(np.isfinite(a)) else float("nan")


def _nanmax(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float("nan")


def _slot_series(
    vals: np.ndarray,
    ecg_start: pd.Timestamp,
    tstart: pd.Timestamp,
    tend: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 10-min slot HR / avgNN / (mask for length alignment)."""
    ecg_start = pd.Timestamp(ecg_start)
    hrs: list[float] = []
    avg_nns: list[float] = []
    if vals.size < FS * 5 or tend <= tstart:
        return np.array([]), np.array([]), np.array([])
    x = bandpass_ecg(vals)
    step = timedelta(minutes=SEG_MIN)
    w = pd.Timestamp(tstart)
    while w + step <= tend:
        off = int(round((w - ecg_start).total_seconds() * FS))
        w += step
        if off < 0 or off + SAMPLES_10MIN > vals.size:
            hrs.append(np.nan)
            avg_nns.append(np.nan)
            continue
        seg = vals[off : off + SAMPLES_10MIN]
        if np.mean(seg == 0.0) > FLATLINE_FRAC:
            hrs.append(np.nan)
            avg_nns.append(np.nan)
            continue
        rr = rr_from_filtered(x[off : off + SAMPLES_10MIN])
        if rr is None:
            hrs.append(np.nan)
            avg_nns.append(np.nan)
            continue
        h = hrv_time(rr)
        hrs.append(h.get("avgHR", np.nan))
        avg_nns.append(h.get("avgNN", np.nan))
    return np.asarray(hrs, float), np.asarray(avg_nns, float), np.ones(len(hrs))


def _temp_slot_means(
    tser: pd.Series,
    tstart: pd.Timestamp,
    n_slots: int,
) -> np.ndarray:
    """Mean temperature per 10-min slot."""
    out = np.full(n_slots, np.nan)
    if tser is None or len(tser) == 0 or n_slots == 0:
        return out
    for i in range(n_slots):
        a = tstart + timedelta(minutes=SEG_MIN * i)
        b = a + timedelta(minutes=SEG_MIN)
        sub = tser.loc[(tser.index >= a) & (tser.index < b)]
        if len(sub):
            out[i] = float(sub.mean())
    return out


def _block_stats(x: np.ndarray, n_blocks: int = N_BLOCKS) -> dict[str, float]:
    """fixed non-overlapping blocks → mean/std per block + global."""
    out: dict[str, float] = {
        "mean": _nanmean(x),
        "std": _nanstd(x),
        "min": _nanmin(x),
        "max": _nanmax(x),
    }
    n = len(x)
    if n == 0:
        for b in range(n_blocks):
            out[f"b{b}_mean"] = float("nan")
            out[f"b{b}_std"] = float("nan")
        return out
    # split into n_blocks as evenly as possible
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    for b in range(n_blocks):
        seg = x[edges[b] : edges[b + 1]]
        out[f"b{b}_mean"] = _nanmean(seg)
        out[f"b{b}_std"] = _nanstd(seg)
    return out


def build_night_row(
    code: str,
    patient: str,
    measure_date: Optional[pd.Timestamp],
    ecg_path: str | None,
    tmp_path: str | None,
    ecg_start: Optional[pd.Timestamp] = None,
    ecg_end: Optional[pd.Timestamp] = None,
) -> tuple[Optional[dict], Optional[str]]:
    if not ecg_path:
        return None, "no_ecg"
    if pd.isna(measure_date):
        return None, "no_date"
    if pd.isna(ecg_start):
        return None, "no_ecg_start"

    try:
        vals = load_ecg(str(ecg_path))
    except Exception:
        return None, "ecg_load_error"
    if vals.size < FS * 5:
        return None, "ecg_too_short"

    tstart, tend, span_err = resolve_used_span(
        pd.Timestamp(measure_date),
        pd.Timestamp(ecg_start),
        vals.size,
        os.environ.get("OVUL_WINDOW", "sleep"),
        TRIM_HEAD_MIN,
        TRIM_TAIL_MIN,
        ecg_end=ecg_end,
    )
    if span_err:
        return None, span_err

    hr, avg_nn, _ = _slot_series(vals, ecg_start, tstart, tend)
    if hr.size == 0 or not np.any(np.isfinite(hr)):
        return None, "no_hrv_windows"

    try:
        tdf = load_temp_json(tmp_path)
    except Exception:
        return None, "temp_load_error"
    tser = temp_in_span(tdf, tstart, tend)
    if len(tser) == 0:
        return None, "no_temp_in_span"
    temp = _temp_slot_means(tser, tstart, len(hr))

    # no EDA — no constant column created (eda_available removed)
    row: dict = {
        "participant_id": str(patient).strip(),
        "file_name_x": str(code).strip(),
        "cycle": np.nan,
        "measure_date": pd.Timestamp(measure_date).normalize(),
        "ecg_start": pd.Timestamp(ecg_start),
        "used_from": tstart,
        "used_to": tend,
        "used_min": float((tend - tstart).total_seconds() / 60.0),
        "n_slots": int(len(hr)),
        "ovulation_flag": -1,
    }

    for name, arr in (("temp", temp), ("hr", hr), ("avgNN", avg_nn)):
        st = _block_stats(arr)
        for k, v in st.items():
            row[f"{name}_{k}"] = v

    # short rolling (within-night): last-third vs first-third mean
    # short nights: thirds overlap or have very few samples → NaN
    n = len(hr)
    MIN_SLOTS_ROLL = 9
    if n < MIN_SLOTS_ROLL:
        row["hr_roll_late_minus_early"] = float("nan")
        row["temp_roll_late_minus_early"] = float("nan")
        row["avgNN_roll_late_minus_early"] = float("nan")
    else:
        third = n // 3
        row["hr_roll_late_minus_early"] = _nanmean(hr[-third:]) - _nanmean(hr[:third])
        row["temp_roll_late_minus_early"] = _nanmean(temp[-third:]) - _nanmean(temp[:third])
        row["avgNN_roll_late_minus_early"] = _nanmean(avg_nn[-third:]) - _nanmean(avg_nn[:third])

    return row, None


def enrich_npj_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """After 01: Δ over the last 1–2 nights. NaN if the day_of_cycle gap does not match the lag."""
    out = df.copy()
    for c in (
        "d_temp_mean_1", "d_hr_mean_1", "d_temp_mean_2", "d_hr_mean_2",
    ):
        out[c] = np.nan
    if "day_of_cycle" not in out.columns:
        return out

    def _d(vals: np.ndarray, days: np.ndarray, pos: int, lag: int) -> float:
        j = pos - lag
        if j < 0:
            return float("nan")
        if not (np.isfinite(vals[pos]) and np.isfinite(vals[j])):
            return float("nan")
        if not (np.isfinite(days[pos]) and np.isfinite(days[j])):
            return float("nan")
        gap = float(days[pos] - days[j])
        if gap < lag - 0.5 or gap > lag + 1.5:
            return float("nan")
        return float(vals[pos] - vals[j])

    for (_, _), g in out.groupby(["participant_id", "cycle"], dropna=False):
        if g["day_of_cycle"].isna().all():
            continue
        idx = g.sort_values("day_of_cycle", kind="mergesort").index.to_list()
        cd = pd.to_numeric(out.loc[idx, "day_of_cycle"], errors="coerce").to_numpy()
        t = pd.to_numeric(out.loc[idx, "temp_mean"], errors="coerce").to_numpy()
        h = pd.to_numeric(out.loc[idx, "hr_mean"], errors="coerce").to_numpy()
        for pos, i in enumerate(idx):
            out.loc[i, "d_temp_mean_1"] = _d(t, cd, pos, 1)
            out.loc[i, "d_hr_mean_1"] = _d(h, cd, pos, 1)
            out.loc[i, "d_temp_mean_2"] = _d(t, cd, pos, 2)
            out.loc[i, "d_hr_mean_2"] = _d(h, cd, pos, 2)
    return out


def main() -> None:
    import argparse

    global OUT_PATH, TAG

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    TAG = os.environ.get("OVUL_TAG", TAG)
    out_name = os.environ.get("OVUL_OUT")
    if out_name:
        OUT_PATH = PRE_DIR / out_name
    else:
        OUT_PATH = PRE_DIR / f"ovul_data_{TAG}.xlsx"
    window_mode = os.environ.get("OVUL_WINDOW", "sleep")
    os.environ["OVUL_WINDOW"] = window_mode
    if window_mode == "sleep":
        win_desc = "sleep ±30min trim"
    elif window_mode == "0206":
        win_desc = "resolved 02:00–06:00"
    else:
        win_desc = "recording center ±2h"
    print(f"npj WH 2025 ({TAG}): {win_desc}, fixed-block stats (temp/HR/avgNN); EDA=N/A")
    all_rows, skip_counts, _ = run_mapping_loop(
        build_night_row, desc=f"building master ({TAG})", workers=args.workers
    )
    if not all_rows:
        raise RuntimeError("No rows were generated.")
    df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_PATH, index=False, engine="openpyxl")
    print(f"saved: {OUT_PATH}  (nights {len(df)}, F≈{df.shape[1]})  skip={skip_counts}")
    print("next: 01_make_data.py → 02_temp_day_interp.py (enrich)")


if __name__ == "__main__":
    main()
