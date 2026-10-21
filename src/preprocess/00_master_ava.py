# -*- coding: utf-8 -*-
"""
Path ①: Ava-style per-night aggregation — trim **+90/−30 min** (paper).

``loess_smooth`` output is clipped to the **observed input min/max** (blocks robust LOESS boundary overshoot).
The +30/−30 variant applies the same clip.
Output: ovul_data/pre/ovul_data_ava.xlsx
"""
from __future__ import annotations

import os
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from statsmodels.nonparametric.smoothers_lowess import lowess

from _raw_common import PRE_DIR, load_temp_json, resolve_used_span, run_mapping_loop

# ---------------------------------------------------------------------------
# Paths — this file is for 90/30 only
# ---------------------------------------------------------------------------
OUT_PATH = PRE_DIR / "ovul_data_ava.xlsx"
TAG = "ava"

FS = 256
SEG_MIN = 10
SAMPLES_10MIN = FS * 60 * SEG_MIN

# Ava-style trim (drop first 90 min after sleep onset / last 30 min before waking) — same as the paper
TRIM_HEAD_MIN = 90
TRIM_TAIL_MIN = 30
# LOESS as in the paper (locally estimated scatterplot smoothing)
LOESS_FRAC = 0.3
LOESS_MIN_PTS = 5

FLATLINE_FRAC = 0.5

# Base HRV metrics per window (hrv_time output)
HRV_BASE = [
    "avgHR", "maxNN", "avgNN", "sdnn", "nn50", "pnn50", "rmssd",
    "tp", "ulf", "vlf", "lf", "hf", "lfhf",
]
# Per-night aggregation: per-window HRV → LOESS → mean/min/max (avgHR_min = minHR)
AGG_SPEC: dict[str, tuple[str, ...]] = {
    "avgHR": ("mean", "min", "max"),
    "rmssd": ("mean", "min", "max"),
    "sdnn": ("mean", "min", "max"),
    "pnn50": ("mean", "min", "max"),
    "nn50": ("mean", "min", "max"),
    "avgNN": ("mean", "min", "max"),
    "maxNN": ("mean", "min", "max"),
    "lf": ("mean", "min", "max"),
    "hf": ("mean", "min", "max"),
    "lfhf": ("mean", "min", "max"),
    "tp": ("mean", "min", "max"),
    "vlf": ("mean", "min", "max"),
    "ulf": ("mean", "min", "max"),
}
HRV_AGG_COLS = [f"{b}_{a}" for b, aggs in AGG_SPEC.items() for a in aggs]
TEMP_AGG_COLS = ["temp_mean", "temp_min", "temp_max", "temp_range"]
QUALITY_COLS = ["n_hrv_win", "n_temp_pts", "used_min", "record_min"]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def loess_smooth(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    frac: float = LOESS_FRAC,
) -> np.ndarray:
    """
    1D LOESS (paper: each parameter LOESS-smoothed before analysis).
    Returns the input unchanged if there are too few observations. NaN positions are kept.
    Output is clipped to the observed range of y[ok] (prevents it=3 boundary overshoot).
    """
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n == 0:
        return y
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    if int(ok.sum()) < LOESS_MIN_PTS:
        return y
    # frac too large vs n → clamp
    f = float(np.clip(frac, 0.05, 1.0))
    if int(ok.sum()) * f < 3:
        f = min(1.0, 3.0 / max(int(ok.sum()), 1))
    y_ok = y[ok]
    sm = lowess(y_ok, x[ok], frac=f, return_sorted=True)
    out = y.copy()
    smoothed = np.interp(x[ok], sm[:, 0], sm[:, 1])
    lo, hi = float(np.min(y_ok)), float(np.max(y_ok))
    out[ok] = np.clip(smoothed, lo, hi)
    return out


@lru_cache(maxsize=64)
def _load_ecg_values_cached(path: str) -> np.ndarray:
    """Load a whole ECG csv at once (once per measurement; lru reuses the same path)."""
    try:
        return np.loadtxt(path, dtype=np.float64)
    except Exception:
        return np.genfromtxt(path, dtype=np.float64)


def bandpass_ecg(x: np.ndarray, fs: int = FS, low: float = 5.0, high: float = 25.0) -> np.ndarray:
    sos = butter(3, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def rr_from_filtered(x: np.ndarray, fs: int = FS) -> np.ndarray | None:
    """R-R (ms) from a band-passed 10-minute segment.

    The old criteria (prominence = 0.3*std, refractory 0.25 s) had a low threshold and counted
    T waves as beats. Mean overnight heart rate rose to 106 bpm and RMSSD exceeded SDNN on 90 % of
    nights — values impossible in a sleeping person. In a check against 34 real nights, the criteria below
    gave 60.6 bpm and RMSSD>SDNN on 28 %, restoring textbook ranges.

      · Refractory 0.40 s = 150 bpm ceiling. No sinus rhythm exceeds this during sleep.
      · Threshold is half the **R-wave amplitude itself** (98th percentile), not the signal std.
        std mixes in T waves and baseline wander, so it cannot anchor an R-wave threshold.

    After dropping out-of-range RR, the remaining array is no longer a contiguous beat sequence. Differencing
    across the gaps inflates RMSSD, so instead of dropping we split and return only the longest
    intact segment.
    """
    if x.size < fs * 5:
        return None
    dist = int(0.40 * fs)
    thr = float(np.percentile(np.abs(x), 98)) * 0.5
    peaks, _ = find_peaks(x, distance=dist, height=thr)
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / float(fs) * 1000.0
    ok = (rr > 300) & (rr < 2000)
    if not ok.any():
        return None
    # Longest intact run. Keeps differences from jumping across gaps.
    best, cur = (0, 0), 0
    for i, v in enumerate(np.r_[ok, False]):
        if v:
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (i - cur, i)
            cur = 0
    rr = rr[best[0]:best[1]]
    return rr if rr.size >= 2 else None


def hrv_time(rr_ms: np.ndarray) -> dict[str, float]:
    rr = np.asarray(rr_ms, dtype=float)
    med = float(np.median(rr))
    if med <= 0:
        return {k: np.nan for k in HRV_BASE}
    keep = (rr > 0.5 * med) & (rr < 1.8 * med)
    rr = rr[keep]
    if rr.size < 2:
        return {k: np.nan for k in HRV_BASE}
    d = np.diff(rr)
    nn50 = float(np.sum(np.abs(d) > 50.0))
    pnn50 = float(np.mean(np.abs(d) > 50.0) * 100.0) if d.size else np.nan
    out = {
        "avgHR": float(60000.0 / np.mean(rr)),
        "maxNN": float(np.max(rr)),
        "avgNN": float(np.mean(rr)),
        "sdnn": float(np.std(rr, ddof=1)) if rr.size > 1 else np.nan,
        "nn50": nn50,
        "pnn50": pnn50,
        "rmssd": float(np.sqrt(np.mean(d**2))) if d.size else np.nan,
    }
    out.update(hrv_freq(rr))
    return out


def hrv_freq(rr_ms: np.ndarray) -> dict[str, float]:
    """ULF/VLF/LF/HF/TP/lfhf via Welch (many NaNs for short segments)."""
    rr = np.asarray(rr_ms, dtype=float) / 1000.0
    if rr.size < 8:
        return {k: np.nan for k in ("tp", "ulf", "vlf", "lf", "hf", "lfhf")}
    t = np.concatenate([[0], np.cumsum(rr[:-1])])
    t_end = t[-1] + rr[-1]
    if t_end <= 0:
        return {k: np.nan for k in ("tp", "ulf", "vlf", "lf", "hf", "lfhf")}
    fs_r = 4.0
    t_u = np.arange(0.0, t_end, 1.0 / fs_r)
    if t_u.size < 32:
        return {k: np.nan for k in ("tp", "ulf", "vlf", "lf", "hf", "lfhf")}
    hr = 60.0 / rr
    j = 0
    sig = np.empty_like(t_u)
    for i, ti in enumerate(t_u):
        while j + 1 < len(t) and ti >= t[j + 1]:
            j += 1
        sig[i] = hr[min(j, len(hr) - 1)]
    sig = sig - np.nanmean(sig)
    nper = min(256, max(32, len(sig) // 4))
    f, pxx = welch(sig, fs=fs_r, nperseg=nper, noverlap=nper // 2)

    def _band(a: float, b: float) -> float:
        m = (f >= a) & (f < b)
        return float(_trapz(pxx[m], f[m])) if np.any(m) else np.nan

    ulf = _band(0.0, 0.0033)
    vlf = _band(0.0033, 0.04)
    lf = _band(0.04, 0.15)
    hf = _band(0.15, 0.4)
    tp = float(_trapz(pxx, f)) if pxx.size else np.nan
    lfhf = lf / hf if hf and hf == hf and hf > 0 else np.nan
    return {"tp": tp, "ulf": ulf, "vlf": vlf, "lf": lf, "hf": hf, "lfhf": lfhf}


def hrv_windows_over_span(
    vals: np.ndarray,
    ecg_start: pd.Timestamp,
    tstart: pd.Timestamp,
    tend: pd.Timestamp,
) -> list[dict[str, float]]:
    """Split the trimmed span into 10-minute windows and compute per-window HRV (valid windows only)."""
    ecg_start = pd.Timestamp(ecg_start)
    rows: list[dict[str, float]] = []
    if vals.size < FS * 5 or tend <= tstart:
        return rows
    x = bandpass_ecg(vals)
    step = timedelta(minutes=SEG_MIN)
    w = pd.Timestamp(tstart)
    while w + step <= tend:
        off = int(round((w - ecg_start).total_seconds() * FS))
        w += step
        if off < 0 or off + SAMPLES_10MIN > vals.size:
            continue
        seg_raw = vals[off : off + SAMPLES_10MIN]
        if np.mean(seg_raw == 0.0) > FLATLINE_FRAC:
            continue
        rr = rr_from_filtered(x[off : off + SAMPLES_10MIN])
        if rr is None:
            continue
        rows.append(hrv_time(rr))
    return rows


def aggregate_hrv(rows: list[dict[str, float]], use_loess: bool = True) -> dict[str, float]:
    """Per-window HRV list → (optional LOESS) → NaN-aware per-night aggregation."""
    out: dict[str, float] = {}
    df = pd.DataFrame(rows, columns=HRV_BASE) if rows else pd.DataFrame(columns=HRV_BASE)
    if use_loess and len(df) >= LOESS_MIN_PTS:
        x = np.arange(len(df), dtype=float)
        for base in AGG_SPEC:
            if base not in df.columns:
                continue
            df[base] = loess_smooth(df[base].to_numpy(dtype=float), x=x)
    for base, aggs in AGG_SPEC.items():
        col = df[base].to_numpy(dtype=float) if base in df else np.array([], dtype=float)
        valid = col[np.isfinite(col)] if col.size else col
        for a in aggs:
            name = f"{base}_{a}"
            if valid.size:
                out[name] = float(getattr(np, f"nan{a}")(valid))
            else:
                out[name] = np.nan
    out["n_hrv_win"] = int(len(rows))
    return out


def aggregate_temp(
    raw: Optional[pd.DataFrame],
    tstart: pd.Timestamp,
    tend: pd.Timestamp,
    use_loess: bool = True,
) -> dict[str, float]:
    """Observed temperature within the valid span → (optional LOESS) → mean/min/max/range."""
    empty = {
        "temp_mean": np.nan, "temp_min": np.nan, "temp_max": np.nan,
        "temp_range": np.nan, "n_temp_pts": 0,
    }
    if raw is None or raw.empty:
        return empty
    sub = raw.loc[(raw.index >= tstart) & (raw.index < tend), "valueC"].dropna()
    if sub.empty:
        return empty
    v = sub.to_numpy(dtype=float)
    if use_loess and v.size >= LOESS_MIN_PTS:
        x = (sub.index - pd.Timestamp(tstart)).total_seconds().to_numpy(dtype=float)
        v_s = loess_smooth(v, x=x)
    else:
        v_s = v
    return {
        "temp_mean": float(np.nanmean(v_s)),
        "temp_min": float(np.nanmin(v_s)),
        "temp_max": float(np.nanmax(v_s)),
        "temp_range": float(np.nanmax(v_s) - np.nanmin(v_s)),
        "n_temp_pts": int(sub.size),
    }


def recording_duration_min(
    ecg_start: pd.Timestamp,
    n_samples: int,
    ecg_end: Optional[pd.Timestamp] = None,
) -> float:
    """Raw recording length (min). If the mapping has an end time, the shorter of file length and mapping length."""
    ecg_start = pd.Timestamp(ecg_start)
    file_min = (n_samples / FS) / 60.0
    if ecg_end is not None and pd.notna(ecg_end):
        map_min = (pd.Timestamp(ecg_end) - ecg_start).total_seconds() / 60.0
        if map_min > 0:
            return float(min(file_min, map_min))
    return float(file_min)


def build_night_row(
    code: str,
    patient: str,
    measure_date: Optional[pd.Timestamp],
    ecg_path: str | None,
    tmp_path: str | None,
    ecg_start: Optional[pd.Timestamp] = None,
    ecg_end: Optional[pd.Timestamp] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """One-row per-night summary (a single dict instead of rows). Paper: 90/30 trim + LOESS only."""
    if not ecg_path:
        return None, "no_ecg"
    if pd.isna(measure_date):
        return None, "no_date"
    if pd.isna(ecg_start):
        return None, "no_ecg_start"

    try:
        vals = _load_ecg_values_cached(str(ecg_path))
    except Exception:
        return None, "ecg_load_error"
    if vals.size < FS * 5:
        return None, "ecg_too_short"

    dur_min = recording_duration_min(ecg_start, vals.size, ecg_end=ecg_end)

    head = int(os.environ.get("OVUL_AVA_TRIM_HEAD", TRIM_HEAD_MIN))
    tail = int(os.environ.get("OVUL_AVA_TRIM_TAIL", TRIM_TAIL_MIN))
    window_mode = os.environ.get("OVUL_WINDOW", "sleep")
    tstart, tend, span_err = resolve_used_span(
        pd.Timestamp(measure_date),
        pd.Timestamp(ecg_start),
        vals.size,
        window_mode,
        head,
        tail,
        ecg_end=ecg_end,
    )
    if span_err:
        return None, span_err
    used_min = (tend - tstart).total_seconds() / 60.0

    hrv_rows = hrv_windows_over_span(vals, ecg_start, tstart, tend)
    if not hrv_rows or not any(np.isfinite(r.get("avgHR", np.nan)) for r in hrv_rows):
        return None, "no_hrv_windows"
    hrv_agg = aggregate_hrv(hrv_rows)

    try:
        tdf = load_temp_json(tmp_path)
    except Exception:
        return None, "temp_load_error"
    temp_agg = aggregate_temp(tdf, tstart, tend)
    if int(temp_agg.get("n_temp_pts", 0) or 0) <= 0:
        return None, "no_temp_in_span"

    row = {
        "participant_id": str(patient).strip(),
        "file_name_x": str(code).strip(),
        "cycle": np.nan,
        "measure_date": pd.Timestamp(measure_date).normalize(),
        "ecg_start": pd.Timestamp(ecg_start),
        "used_from": tstart,
        "used_to": tend,
        "used_min": float(used_min),
        "record_min": float(dur_min),
        "ovulation_flag": -1,
    }
    row.update(hrv_agg)
    row.update(temp_agg)
    return row, None


def main(workers: int = 8):
    """Module constants + env (OVUL_AVA_*, OVUL_WINDOW) override for ava30/ava24/ava24mid."""
    global TRIM_HEAD_MIN, TRIM_TAIL_MIN, OUT_PATH, TAG

    TRIM_HEAD_MIN = int(os.environ.get("OVUL_AVA_TRIM_HEAD", str(TRIM_HEAD_MIN)))
    TRIM_TAIL_MIN = int(os.environ.get("OVUL_AVA_TRIM_TAIL", str(TRIM_TAIL_MIN)))
    TAG = os.environ.get("OVUL_AVA_TAG", TAG)
    window_mode = os.environ.get("OVUL_WINDOW", "sleep")
    out_name = os.environ.get("OVUL_AVA_OUT")
    if out_name:
        OUT_PATH = PRE_DIR / out_name
    elif TAG == "ava30":
        OUT_PATH = PRE_DIR / "ovul_data_ava30.xlsx"
    else:
        OUT_PATH = PRE_DIR / f"ovul_data_{TAG}.xlsx"

    os.environ["OVUL_AVA_TRIM_HEAD"] = str(TRIM_HEAD_MIN)
    os.environ["OVUL_AVA_TRIM_TAIL"] = str(TRIM_TAIL_MIN)
    os.environ["OVUL_WINDOW"] = window_mode
    if window_mode == "sleep":
        win_desc = f"trim +{TRIM_HEAD_MIN}/-{TRIM_TAIL_MIN}min"
    elif window_mode == "0206":
        win_desc = "resolved 02:00–06:00"
    else:
        win_desc = "recording midpoint ±2h"
    print(f"Ava style ({TAG}): {win_desc} + LOESS(frac={LOESS_FRAC})")
    all_rows, skip_counts, _ = run_mapping_loop(
        build_night_row, desc=f"Building master ({TAG} per-night aggregation)", workers=workers
    )
    if not all_rows:
        raise RuntimeError(
            "No rows were generated. Check file_name in the mapping against file names in the raw data folder."
        )

    df = pd.DataFrame(all_rows)
    col_order = (
        ["participant_id", "file_name_x", "cycle", "measure_date"]
        + HRV_AGG_COLS
        + TEMP_AGG_COLS
        + QUALITY_COLS
        + ["ecg_start", "used_from", "used_to", "ovulation_flag"]
    )
    df = df[[c for c in col_order if c in df.columns]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_PATH, index=False, engine="openpyxl")
    print(f"Saved: {OUT_PATH}  ({len(df)} night rows)  skip={skip_counts}")
    print(
        f"Paper method: {win_desc} + LOESS(frac={LOESS_FRAC}). "
        "0 valid HRV windows or 0 temp points → skip."
    )
    print(f"Next: 01 input → {OUT_PATH.name}")


def _self_check() -> None:
    """LOESS clip: with uneven x + outliers at the tail, smoothed max ≤ observed max."""
    rng = np.random.default_rng(3140)
    y = np.array([36.7] * 8 + [36.8] * 8 + [36.9] * 7 + [29.4, 28.0, 27.5], dtype=float)
    xg = np.sort(rng.uniform(0, 4 * 3600 - 60, 23))
    xo = np.sort(rng.uniform(4 * 3600 - 300, 4 * 3600, 3))
    x = np.concatenate([xg, xo])
    y_hi, y_lo = float(y.max()), float(y.min())
    out = loess_smooth(y, x=x)
    assert float(np.nanmax(out)) <= y_hi + 1e-9, out
    assert float(np.nanmin(out)) >= y_lo - 1e-9, out
    # Without the clip, this case went above 100°C
    sm_raw = lowess(y, x, frac=0.3, return_sorted=True)
    raw = np.interp(x, sm_raw[:, 0], sm_raw[:, 1])
    assert float(raw.max()) > 50.0
    print("self-check OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ava per-night aggregation: trim +90/−30 + LOESS obs clip")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--self-check", action="store_true", help="loess_smooth clip smoke test")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
    else:
        main(workers=args.workers)
