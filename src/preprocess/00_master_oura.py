# -*- coding: utf-8 -*-
"""
Path ⑤ : approximation of Oura Cycle Insights / Fertile Window.

References
----
- Oura blog / JMIR validation (ovulation detection vs calendar)
- The detailed feature list and model are not public → implement only the **published signal philosophy**:
    · nightly mean skin temperature (deviation from baseline)
    · HR / HRV (rMSSD)
    · temperature trend (over several days)
    · menstruation log ↔ day_of_cycle (assigned in 01)

Limitations of our data
----
  - no respiratory rate
  - Oura finger temp ≠ wrist temp (proxy)
  - commercial DL not reproduced — tabular summaries only

Output
----
  ovul_data/pre/ovul_data_oura.xlsx  (one row per night: temp/hr/rmssd means)
  Prerequisite: ``python _raw_common.py`` QC keep only (run_mapping_loop)
  → 01 → `enrich_oura_cycle` in 02 (early-follicular baseline, roll3, slope)

Pipeline: 00 (this file) → 01 → 02 (+enrich).
"""
from __future__ import annotations

from typing import Optional

import os
import numpy as np
import pandas as pd

from _raw_common import (
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

# Oura uses the "mean during sleep"; trim shorter edges than Ava's 90/30
TRIM_HEAD_MIN = 30
TRIM_TAIL_MIN = 30
# early follicular baseline: cycle days 1..BASE_CD_MAX (after 01)
BASE_CD_MAX = 8

OUT_PATH = PRE_DIR / "ovul_data_oura.xlsx"
TAG = "oura"

NIGHT_COLS = [
    "participant_id", "file_name_x", "cycle", "measure_date",
    "ecg_start", "used_from", "used_to", "used_min",
    "temp_mean", "temp_min", "temp_max",
    "hr_mean", "hr_min",
    "rmssd_mean", "sdnn_mean",
    "n_hrv_win", "n_temp_pts",
    "ovulation_flag",
]


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

    hrv_rows = hrv_windows_over_span(vals, ecg_start, tstart, tend)
    if not hrv_rows:
        return None, "no_hrv_windows"
    dfh = pd.DataFrame(hrv_rows)

    try:
        tdf = load_temp_json(tmp_path)
    except Exception:
        return None, "temp_load_error"
    tser = temp_in_span(tdf, tstart, tend)
    if len(tser) == 0:
        return None, "no_temp_in_span"

    row = {
        "participant_id": str(patient).strip(),
        "file_name_x": str(code).strip(),
        "cycle": np.nan,
        "measure_date": pd.Timestamp(measure_date).normalize(),
        "ecg_start": pd.Timestamp(ecg_start),
        "used_from": tstart,
        "used_to": tend,
        "used_min": float((tend - tstart).total_seconds() / 60.0),
        "temp_mean": float(tser.mean()) if len(tser) else float("nan"),
        "temp_min": float(tser.min()) if len(tser) else float("nan"),
        "temp_max": float(tser.max()) if len(tser) else float("nan"),
        "hr_mean": float(np.nanmean(dfh["avgHR"])) if "avgHR" in dfh else float("nan"),
        "hr_min": float(np.nanmin(dfh["avgHR"])) if "avgHR" in dfh else float("nan"),
        "rmssd_mean": float(np.nanmean(dfh["rmssd"])) if "rmssd" in dfh else float("nan"),
        "sdnn_mean": float(np.nanmean(dfh["sdnn"])) if "sdnn" in dfh else float("nan"),
        "n_hrv_win": int(len(hrv_rows)),
        "n_temp_pts": int(len(tser)),
        "ovulation_flag": -1,
    }
    return row, None


def enrich_oura_cycle(df: pd.DataFrame, base_cd_max: int = BASE_CD_MAX) -> pd.DataFrame:
    """
    After 01: Oura-style baseline deviation and 3-day trend.
    baseline = mean over past nights in the same cycle, **excluding the current one**, with day_of_cycle≤base_cd_max.
    If there is no past baseline, *_dev=NaN (do not fabricate 0 from the night itself).
    """
    out = df.copy()
    extra = [
        "temp_dev", "hr_dev", "rmssd_dev",
        "temp_roll3", "hr_roll3", "temp_slope3",
        "temp_rise_from_nadir", "cycle_day",
    ]
    for c in extra:
        out[c] = np.nan

    if "day_of_cycle" not in out.columns:
        return out

    def _slope(vals: np.ndarray, days: np.ndarray) -> float:
        m = np.isfinite(vals) & np.isfinite(days)
        if m.sum() < 2:
            return float("nan")
        return float(np.polyfit(days[m], vals[m], 1)[0])

    for (_, _), g in out.groupby(["participant_id", "cycle"], dropna=False):
        if g["day_of_cycle"].isna().all():
            continue
        idx = g.sort_values("day_of_cycle", kind="mergesort").index.to_list()
        cd = pd.to_numeric(out.loc[idx, "day_of_cycle"], errors="coerce").to_numpy()
        temp = pd.to_numeric(out.loc[idx, "temp_mean"], errors="coerce").to_numpy()
        hr = pd.to_numeric(out.loc[idx, "hr_mean"], errors="coerce").to_numpy()
        rmssd = pd.to_numeric(out.loc[idx, "rmssd_mean"], errors="coerce").to_numpy()

        for pos, i in enumerate(idx):
            out.loc[i, "cycle_day"] = float(cd[pos]) if np.isfinite(cd[pos]) else np.nan
            # exclude the current night
            past = list(range(pos))
            base_ix = [
                j for j in past
                if np.isfinite(cd[j]) and cd[j] <= base_cd_max and np.isfinite(temp[j])
            ]
            # HR/rmssd baseline uses the same indices (early follicular nights)

            def _dev(arr: np.ndarray, j: int, bi: list[int]) -> float:
                if not bi:
                    return float("nan")
                b = np.nanmean(arr[bi])
                if not np.isfinite(b) or not np.isfinite(arr[j]):
                    return float("nan")
                return float(arr[j] - b)

            out.loc[i, "temp_dev"] = _dev(temp, pos, base_ix)
            out.loc[i, "hr_dev"] = _dev(hr, pos, base_ix)
            out.loc[i, "rmssd_dev"] = _dev(rmssd, pos, base_ix)

            lo = max(0, pos - 2)
            t3 = temp[lo : pos + 1]
            h3 = hr[lo : pos + 1]
            d3 = cd[lo : pos + 1]
            out.loc[i, "temp_roll3"] = float(np.nanmean(t3)) if np.any(np.isfinite(t3)) else np.nan
            out.loc[i, "hr_roll3"] = float(np.nanmean(h3)) if np.any(np.isfinite(h3)) else np.nan
            out.loc[i, "temp_slope3"] = _slope(t3, d3)

            # rise: only when past history exists (do not fabricate 0 for the first night)
            if pos >= 1 and np.isfinite(temp[pos]) and np.any(np.isfinite(temp[:pos])):
                hist = temp[: pos + 1]
                out.loc[i, "temp_rise_from_nadir"] = float(temp[pos] - np.nanmin(hist))
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
        win_desc = "record center ±2h"
    print(f"Oura-approx ({TAG}): {win_desc}; baseline+trend after 01 (enrich_oura_cycle)")
    all_rows, skip_counts, _ = run_mapping_loop(
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
