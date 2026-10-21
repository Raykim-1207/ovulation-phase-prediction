# -*- coding: utf-8 -*-
"""
Path 3: Huawei / Luo et al. 2022 (Reprod Biol Endocrinol) style.

Paper
----
https://doi.org/10.1186/s12958-022-00993-4
  - Exclude **the first 30 min after sleep onset and the last 30 min before waking** from the sleep record
  - Daily: ear BBT + wrist HR (+ SDNN, LF/HF)
  - SDNN and LF/HF are skewed → ln transform
  - Missing: imputed with the mean of the 2 days before and after (after cycle labelling; this step keeps NaN)
  - Model input is the **within-cycle waveform** (smoothing, switch) — cycle day exists only after 01.
    This 00 step outputs per-night scalars only. Cycle waveform enrichment is done in 02 via `enrich_huawei_cycle`.

Limitations of our data
----
  - No ear BBT → **mean nightly wrist temperature (temp) used as BBT proxy** (`bbt`)
  - **ECG-based HR/HRV** instead of Huawei Band PPG

Output
----
  ovul_data/pre/ovul_data_huawei.xlsx  (one row per night)
  Prerequisite: only QC-kept nights from ``python _raw_common.py`` (run_mapping_loop)
  → 01_make_data → 02_temp_day_interp (cycle waveform enrichment)

Pipeline: _raw_common QC → 00 (this) → 01 → 02 (+enrich).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from _raw_common import (
    BASE_DIR,
    FS,
    OUT_DIR,
    PRE_DIR,
    hrv_windows_over_span,
    load_ecg,
    load_temp_json,
    resolve_used_span,
    run_mapping_loop,
    temp_in_span,
)

# Luo 2022: first/last 30 min of sleep excluded
TRIM_HEAD_MIN = 30
TRIM_TAIL_MIN = 30

OUT_PATH = PRE_DIR / "ovul_data_huawei.xlsx"
TAG = "huawei"

# Per-night output columns (paper signals + proxy markers)
NIGHT_COLS = [
    "participant_id", "file_name_x", "cycle", "measure_date",
    "ecg_start", "used_from", "used_to", "used_min", "record_min",
    "bbt",              # wrist temp mean (BBT proxy)
    "hr",               # mean HR
    "sdnn", "lfhf",
    "ln_sdnn", "ln_lfhf",
    "hr_min", "rmssd",
    "n_hrv_win", "n_temp_pts",
    "ovulation_flag",
]


def _safe_ln(x: float, eps: float = 1e-6) -> float:
    if x is None or not np.isfinite(x) or x <= 0:
        return float("nan")
    return float(np.log(max(float(x), eps)))


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

    used_min = (tend - tstart).total_seconds() / 60.0
    record_min = vals.size / FS / 60.0

    hrv_rows = hrv_windows_over_span(vals, ecg_start, tstart, tend)
    if not hrv_rows:
        return None, "no_hrv_windows"

    dfh = pd.DataFrame(hrv_rows)
    hr = float(np.nanmean(dfh["avgHR"])) if "avgHR" in dfh else float("nan")
    hr_min = float(np.nanmin(dfh["avgHR"])) if "avgHR" in dfh else float("nan")
    sdnn = float(np.nanmean(dfh["sdnn"])) if "sdnn" in dfh else float("nan")
    lfhf = float(np.nanmean(dfh["lfhf"])) if "lfhf" in dfh else float("nan")
    rmssd = float(np.nanmean(dfh["rmssd"])) if "rmssd" in dfh else float("nan")

    try:
        tdf = load_temp_json(tmp_path)
    except Exception:
        return None, "temp_load_error"
    tser = temp_in_span(tdf, tstart, tend)
    if len(tser) == 0:
        return None, "no_temp_in_span"
    bbt = float(tser.mean())

    row = {
        "participant_id": str(patient).strip(),
        "file_name_x": str(code).strip(),
        "cycle": np.nan,
        "measure_date": pd.Timestamp(measure_date).normalize(),
        "ecg_start": pd.Timestamp(ecg_start),
        "used_from": tstart,
        "used_to": tend,
        "used_min": float(used_min),
        "record_min": float(record_min),
        "bbt": bbt,
        "hr": hr,
        "sdnn": sdnn,
        "lfhf": lfhf,
        "ln_sdnn": _safe_ln(sdnn),
        "ln_lfhf": _safe_ln(lfhf),
        "hr_min": hr_min,
        "rmssd": rmssd,
        "n_hrv_win": int(len(hrv_rows)),
        "n_temp_pts": int(len(tser)),
        "ovulation_flag": -1,
    }
    return row, None


def enrich_huawei_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    After 01 (cycle and day_of_cycle present): Luo-style within-cycle waveform features.
    - 3-day causal rolling mean (min_periods=3)
    - Δ1 / Δ3: NaN when the day_of_cycle gap is large (never treat missing days as adjacent)
    - slope5: polyfit over the day_of_cycle axis
    - first night days_since/rising_streak → NaN
    """
    out = df.copy()
    extra = [
        "bbt_s3", "hr_s3", "dbbt_1", "dhr_1", "dbbt_3", "dhr_3",
        "bbt_slope5", "hr_slope5", "days_since_bbt_nadir", "rising_streak",
    ]
    for c in extra:
        out[c] = np.nan

    if "day_of_cycle" not in out.columns or "cycle" not in out.columns:
        return out

    for (_, _), g in out.groupby(["participant_id", "cycle"], dropna=False):
        if g["day_of_cycle"].isna().all():
            continue
        idx = g.sort_values("day_of_cycle", kind="mergesort").index.to_list()
        # Fill only the remaining holes with the neighbour mean (when both sides exist). Keep values already interpolated in 02.
        for col in ("bbt", "hr", "sdnn", "lfhf", "ln_sdnn", "ln_lfhf"):
            if col not in out.columns:
                continue
            s = pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            src = s.copy()
            for i in range(len(s)):
                if np.isfinite(src[i]):
                    continue
                lo = src[i - 1] if i > 0 else np.nan
                hi = src[i + 1] if i + 1 < len(src) else np.nan
                if np.isfinite(lo) and np.isfinite(hi):
                    s[i] = 0.5 * (lo + hi)
            out.loc[idx, col] = s

        cd = pd.to_numeric(out.loc[idx, "day_of_cycle"], errors="coerce").to_numpy(dtype=float)
        bbt = pd.to_numeric(out.loc[idx, "bbt"], errors="coerce").to_numpy(dtype=float)
        hr = pd.to_numeric(out.loc[idx, "hr"], errors="coerce").to_numpy(dtype=float)
        bs = pd.Series(bbt).rolling(3, min_periods=3).mean().to_numpy()
        hs = pd.Series(hr).rolling(3, min_periods=3).mean().to_numpy()

        def slope(vals: np.ndarray, days: np.ndarray, w: int = 5) -> float:
            y = vals[-w:]
            d = days[-w:]
            m = np.isfinite(y) & np.isfinite(d)
            if m.sum() < 2:
                return float("nan")
            return float(np.polyfit(d[m], y[m], 1)[0])

        def _delta(vals: np.ndarray, days: np.ndarray, pos: int, lag: int) -> float:
            j = pos - lag
            if j < 0:
                return float("nan")
            if not (np.isfinite(vals[pos]) and np.isfinite(vals[j])):
                return float("nan")
            if not (np.isfinite(days[pos]) and np.isfinite(days[j])):
                return float("nan")
            gap = float(days[pos] - days[j])
            # Not adjacent if far from the expected gap (lag days)
            if gap < lag - 0.5 or gap > lag + 1.5:
                return float("nan")
            return float(vals[pos] - vals[j])

        for pos, i in enumerate(idx):
            out.loc[i, "bbt_s3"] = bs[pos]
            out.loc[i, "hr_s3"] = hs[pos]
            out.loc[i, "dbbt_1"] = _delta(bs, cd, pos, 1)
            out.loc[i, "dhr_1"] = _delta(hs, cd, pos, 1)
            out.loc[i, "dbbt_3"] = _delta(bs, cd, pos, 3)
            out.loc[i, "dhr_3"] = _delta(hs, cd, pos, 3)
            out.loc[i, "bbt_slope5"] = slope(bs[: pos + 1], cd[: pos + 1], 5)
            out.loc[i, "hr_slope5"] = slope(hs[: pos + 1], cd[: pos + 1], 5)

            hist_b = bs[: pos + 1]
            if pos >= 1 and np.any(np.isfinite(hist_b)):
                nad = int(np.nanargmin(hist_b))
                out.loc[i, "days_since_bbt_nadir"] = float(len(hist_b) - 1 - nad)
                streak = 0.0
                for a, b2 in zip(hist_b[:-1][::-1], hist_b[1:][::-1]):
                    if np.isfinite(a) and np.isfinite(b2) and b2 > a:
                        streak += 1.0
                    else:
                        break
                out.loc[i, "rising_streak"] = streak
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
        win_desc = "recording midpoint ±2h"
    print(f"Huawei/Luo 2022 ({TAG}): {win_desc}, nightly BBT(proxy)+HR+SDNN+LF/HF")
    print(f"  NOTE: ear BBT unavailable → wrist temp mean as `bbt`")
    all_rows, skip_counts, skipped = run_mapping_loop(
        build_night_row, desc=f"building master ({TAG})", workers=args.workers
    )
    if not all_rows:
        raise RuntimeError("No rows were generated.")
    df = pd.DataFrame(all_rows)
    df = df[[c for c in NIGHT_COLS if c in df.columns]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_PATH, index=False, engine="openpyxl")
    print(f"Saved: {OUT_PATH}  (nights {len(df)})  skip={skip_counts}")
    print("Next: 01_make_data.py → 02_temp_day_interp.py (enrich)")


if __name__ == "__main__":
    main()
