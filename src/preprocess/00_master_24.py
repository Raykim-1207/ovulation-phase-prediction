# -*- coding: utf-8 -*-
"""
Path ②: in-night sequence — calendar **02:00~06:00** window overlapping the recording (24×10 min).

Middle 4 hours (based on the file span): ``00_master_24mid.py``.
measure_date (label) = mapped resolved_date. Both ECG and temp required (QC).
Output: ovul_data/pre/ovul_data_24.xlsx
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from tqdm.auto import tqdm

from _raw_common import (
    BASE_DIR,
    MAP_EXCEL,
    MAP_PATH,
    OUT_DIR,
    PRE_DIR,
    load_mapping_qc,
    load_temp_json,
    resolve_0206_span,
    span_mid4h,
)

# ---------------------------------------------------------------------------
# Paths — this file is for the 02-06 window only
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = PRE_DIR / "ovul_data_24.xlsx"
WINDOW_MODE = "0206"

FS = 256
SEG_MIN = 10
N_WIN = 24  # 4 hours / 10 min
SAMPLES_10MIN = FS * 60 * SEG_MIN
NIGHT_START_HOUR = 2
NIGHT_END_HOUR = 6  # from 02:00 up to (not including) 06:00

# Minimum to keep a night: skip if fewer than this many of the 24 slots have HRV observed.
MIN_OBS_SLOTS = 12
# Slots whose flatline (0.0) fraction exceeds this value are treated as missing.
FLATLINE_FRAC = 0.5

HRV_COLS = [
    "avgHR", "maxNN", "avgNN", "sdnn", "nn50", "pnn50", "rmssd",
    "tp", "ulf", "vlf", "lf", "hf", "lfhf",
]
TEMP_COLS = ["skin_temp", "temp_night_source"]
# Missingness mask channels (GRU-D)
MASK_COLS = ["hrv_mask", "temp_mask", "hrv_dt", "temp_dt"]

RAW_STRIP_PREFIXES = ("CHECKUP_", "temp_")

_trapz = getattr(np, "trapezoid", None) or np.trapz


def norm_code(s: object) -> str:
    """Normalize keys between MAP file_name (date_code) and RAW (CHECKUP_…, temp_CHECKUP_…)."""
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    t = str(s).strip()
    if t.lower() in ("", "nan", "none"):
        return ""
    if re.fullmatch(r"-?\d+\.0", t):
        t = t[:-2]
    for ext in (".csv", ".json", ".CSV", ".JSON", ".xlsx", ".txt"):
        if t.endswith(ext):
            t = t[: -len(ext)]
    t = re.sub(r"\s+", "", t)
    for pref in RAW_STRIP_PREFIXES:
        if len(t) >= len(pref) and t[: len(pref)].lower() == pref.lower():
            t = t[len(pref) :]
            break
    return t.lower()


def build_raw_indexes(
    ecg_files: list[str],
    temp_files: list[str],
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]], int]:
    """ECG: exact normalized key. temp: date-time prefix bucket → O(#candidates) matching."""
    ecg_idx: dict[str, str] = {}
    for p in ecg_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        nk = norm_code(stem)
        if nk and nk not in ecg_idx:
            ecg_idx[nk] = p

    temp_buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_temp_files = 0
    for p in temp_files:
        b = os.path.basename(p)
        stem = b[len("temp_") :] if b.lower().startswith("temp_") else b
        stem = os.path.splitext(stem)[0]
        nk = norm_code(stem)
        if not nk:
            continue
        bucket = nk.split("_", 1)[0]
        temp_buckets[bucket].append((nk, p))
        n_temp_files += 1
    return ecg_idx, dict(temp_buckets), n_temp_files


def find_ecg_path(ecg_idx: dict[str, str], code: str) -> str | None:
    return ecg_idx.get(norm_code(code))


def find_temp_path(temp_buckets: dict[str, list[tuple[str, str]]], code: str) -> str | None:
    """
    MAP 20230608230452_TPKY1 ↔ temp_…TPKY1001_… (same date-time bucket).

    Priority: exact → startswith (shortest extension = most specific) → substring.
    """
    nk = norm_code(code)
    if not nk:
        return None
    cands = temp_buckets.get(nk.split("_", 1)[0], [])
    if not cands:
        return None

    exact = [p for tn, p in cands if tn == nk]
    if exact:
        return min(exact, key=lambda x: len(os.path.basename(x)))

    prefix_hits = [(tn, p) for tn, p in cands if tn.startswith(nk)]
    if prefix_hits:
        # minimize len(tn)-len(nk) → reduces TPKY1 vs TPKY1001/TPKY12 ambiguity
        return min(
            prefix_hits,
            key=lambda x: (len(x[0]) - len(nk), len(os.path.basename(x[1]))),
        )[1]

    sub_hits = [(tn, p) for tn, p in cands if nk in tn]
    if not sub_hits:
        return None
    return min(
        sub_hits,
        key=lambda x: (len(x[0]) - len(nk), len(os.path.basename(x[1]))),
    )[1]


def window_index_0206(measure_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar day D → D 02:00~05:50 (10 min × 24)."""
    base = pd.Timestamp(measure_date).normalize() + timedelta(hours=NIGHT_START_HOUR)
    return pd.date_range(base, periods=N_WIN, freq="10min")


def window_end_0206(measure_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(measure_date).normalize() + timedelta(hours=NIGHT_END_HOUR)


def window_index_mid4h(
    ecg_start: pd.Timestamp,
    ecg_end: Optional[pd.Timestamp],
    n_samples: int | None = None,
) -> pd.DatetimeIndex:
    """Middle of actual file span (map ∩ file) ±2h → 10 min × 24. ``n_samples`` fixes the file length."""
    t0, _ = span_mid4h(ecg_start, ecg_end, n_samples=n_samples)
    return pd.date_range(t0, periods=N_WIN, freq="10min")


def resolve_window_index(
    measure_date: pd.Timestamp,
    ecg_start: pd.Timestamp,
    ecg_end: Optional[pd.Timestamp],
    mode: str,
    n_samples: int | None = None,
) -> tuple[Optional[pd.DatetimeIndex], Optional[str]]:
    """
    Returns (window_idx, skip_reason).
    0206: 02–06 of the **resolved measure_date** (same as the label day). Skip if no overlap with the recording.
    mid4h: middle 4h based on file length.
    """
    if mode == "mid4h":
        return window_index_mid4h(ecg_start, ecg_end, n_samples=n_samples), None
    t0, t1, err = resolve_0206_span(measure_date, ecg_start, ecg_end)
    if err:
        return None, err
    return pd.date_range(t0, periods=N_WIN, freq="10min"), None


@lru_cache(maxsize=64)
def _load_ecg_values_cached(path: str) -> np.ndarray:
    """Load the whole ECG csv at once (once per measurement; lru reuses the same path)."""
    try:
        return np.loadtxt(path, dtype=np.float64)
    except Exception:
        return np.genfromtxt(path, dtype=np.float64)


def check_ecg_span(
    ecg_start: pd.Timestamp,
    ecg_end: Optional[pd.Timestamp],
    night_start: pd.Timestamp,
    night_end: pd.Timestamp,
) -> Optional[str]:
    """Return a skip reason only when the recording does **not overlap the window at all**."""
    ecg_start = pd.Timestamp(ecg_start)
    if ecg_start >= night_end:
        return "ecg_start_after_night"
    if pd.notna(ecg_end):
        ecg_end = pd.Timestamp(ecg_end)
        if ecg_end <= night_start:
            return "ecg_end_before_night"
    return None


def bandpass_ecg(x: np.ndarray, fs: int = FS, low: float = 5.0, high: float = 25.0) -> np.ndarray:
    sos = butter(3, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def rr_from_filtered(x: np.ndarray, fs: int = FS) -> np.ndarray | None:
    """R-R intervals (ms) from a band-passed 10-minute segment.

    The old criterion (prominence = 0.3*std, refractory 0.25 s) had too low a threshold and
    counted T waves as beats. Mean nightly heart rate rose to 106 bpm and RMSSD exceeded SDNN on 90 %
    of nights — values impossible in a sleeping person. Against 34 measured nights, the criterion below
    gives 60.6 bpm and RMSSD>SDNN on 28 %, back within textbook range.

      · Refractory 0.40 s = 150 bpm ceiling. No sinus rhythm exceeds this during sleep.
      · Threshold is half of the **R-wave amplitude itself** (98th percentile), not the signal std.
        The std mixes in T waves and baseline wander, so it cannot anchor an R-wave threshold.

    After dropping out-of-range RR, the remaining array is no longer a contiguous beat sequence.
    Differencing across the gaps inflates RMSSD, so instead of dropping we split and return only
    the longest intact run.
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
    # Longest intact run, so differences never jump across a gap.
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


def hrv_and_mask_per_slot(
    vals: np.ndarray,
    ecg_start: pd.Timestamp,
    window_idx: pd.DatetimeIndex,
) -> tuple[list[dict[str, float]], np.ndarray]:
    """
    ECG extracted with an **independent offset** for each of the 24 slots → HRV + observation mask.

    - off = (slot start - ecg_start)*FS. Missing if negative (not covered, late bedtime) or out of range.
    - Missing if flatline (0.0) fraction > FLATLINE_FRAC or RR detection fails.
    - Missing slots: HRV=NaN, mask=0.  (kept, not dropped)
    """
    ecg_start = pd.Timestamp(ecg_start)
    nan_row = {k: np.nan for k in HRV_COLS}
    hrv_rows: list[dict[str, float]] = []
    mask = np.zeros(len(window_idx), dtype=np.int8)

    x = bandpass_ecg(vals) if vals.size >= FS * 5 else None
    for w, slot_start in enumerate(window_idx):
        off = int(round((slot_start - ecg_start).total_seconds() * FS))
        if x is None or off < 0 or off + SAMPLES_10MIN > vals.size:
            hrv_rows.append(nan_row.copy())
            continue
        seg_raw = vals[off : off + SAMPLES_10MIN]
        if np.mean(seg_raw == 0.0) > FLATLINE_FRAC:
            hrv_rows.append(nan_row.copy())
            continue
        rr = rr_from_filtered(x[off : off + SAMPLES_10MIN])
        if rr is None:
            hrv_rows.append(nan_row.copy())
            continue
        hrv_rows.append(hrv_time(rr))
        mask[w] = 1
    return hrv_rows, mask


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """GRU-D Δt: slots elapsed since the last observation. δ_0=0, δ_t=1+(δ_{t-1} if m_{t-1}==0 else 0)."""
    n = len(mask)
    dt = np.zeros(n, dtype=float)
    for t in range(1, n):
        dt[t] = 1.0 + (dt[t - 1] if mask[t - 1] == 0 else 0.0)
    return dt


def hrv_time(rr_ms: np.ndarray) -> dict[str, float]:
    rr = np.asarray(rr_ms, dtype=float)
    med = float(np.median(rr))
    if med <= 0:
        return {k: np.nan for k in HRV_COLS}
    keep = (rr > 0.5 * med) & (rr < 1.8 * med)
    rr = rr[keep]
    if rr.size < 2:
        return {k: np.nan for k in HRV_COLS}
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
    fq = hrv_freq(rr)
    out.update(fq)
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
    # Approximate the heart-rate tachogram with piecewise-constant RR
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


def temp_24_for_windows(
    raw: pd.DataFrame,
    window_idx: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Measured JSON → floor time to 10 min → reindex onto the 24-slot window grid.
    **No coverage gate**: missing slots stay NaN (→ temp_mask=0).
    """
    night_start = window_idx[0]
    # The window end must be derived from window_idx. With a fixed calendar 06:00, a mid4h window
    # crossing midnight inverts the interval and the whole night is dropped as no_temp_in_window.
    night_end = window_idx[-1] + timedelta(minutes=SEG_MIN)
    sub = raw.loc[(raw.index >= night_start) & (raw.index < night_end)].copy()
    if sub.empty:
        return np.full(N_WIN, np.nan)
    sub.index = sub.index.floor("10min")
    ten = sub.groupby(level=0)["valueC"].mean()
    return ten.reindex(window_idx).to_numpy(dtype=float)


def build_rows_for_code(
    code: str,
    patient: str,
    measure_date: Optional[pd.Timestamp],
    ecg_path: str | None,
    tmp_path: str | None,
    ecg_start: Optional[pd.Timestamp] = None,
    ecg_end: Optional[pd.Timestamp] = None,
    window_mode: str = "0206",
) -> tuple[list[dict], Optional[str]]:
    """
    Returns (rows, skip_reason).

    Window: 0206 = 02–06 of the **resolved day** / mid4h = file middle ±2h (24×10 min).
    measure_date (label) is always the mapped resolved_date (= calendar day of the 0206 window). ECG and temp: per-slot mask.
    """
    if not ecg_path:
        return [], "no_ecg"
    if not tmp_path:
        return [], "no_temp"
    if pd.isna(measure_date):
        return [], "no_date"
    if pd.isna(ecg_start):
        return [], "no_ecg_start"

    try:
        vals = _load_ecg_values_cached(str(ecg_path))
    except Exception:
        return [], "ecg_load_error"

    window_idx, win_err = resolve_window_index(
        pd.Timestamp(measure_date),
        pd.Timestamp(ecg_start),
        ecg_end,
        window_mode,
        n_samples=vals.size,
    )
    if win_err or window_idx is None:
        return [], win_err or "no_window"
    night_start = window_idx[0]
    night_end = window_idx[-1] + timedelta(minutes=SEG_MIN)
    span_err = check_ecg_span(ecg_start, ecg_end, night_start, night_end)
    if span_err:
        return [], span_err

    try:
        tdf = load_temp_json(tmp_path)
        t24 = temp_24_for_windows(tdf, window_idx)
    except Exception:
        return [], "temp_load_error"
    temp_mask = np.isfinite(t24).astype(np.int8)
    if int(temp_mask.sum()) == 0:
        return [], "no_temp_in_window"

    hrv_rows, hrv_mask = hrv_and_mask_per_slot(vals, ecg_start, window_idx)
    if int(hrv_mask.sum()) < MIN_OBS_SLOTS:
        return [], "too_sparse"

    hrv_dt = compute_delta(hrv_mask)
    temp_dt = compute_delta(temp_mask)
    src_tag = "resolved_02-06" if window_mode == "0206" else "mid4h"

    rows: list[dict] = []
    for w in range(N_WIN):
        hrv = hrv_rows[w]
        ts = window_idx[w]
        raw_t = float(t24[w]) if w < len(t24) else np.nan

        row = {
            "participant_id": str(patient).strip(),
            "window": w + 1,
            "file_name_x": str(code).strip(),
            "cycle": np.nan,
            # for label and cycle day — compare resolved date regardless of slot time
            "measure_date": pd.Timestamp(measure_date).normalize(),
            "measure_time": ts,
            "ovulation_flag": -1,
            "skin_temp": raw_t,
            "temp_night_source": src_tag,
        }
        row.update({k: hrv.get(k, np.nan) for k in HRV_COLS})
        row["hrv_mask"] = int(hrv_mask[w])
        row["temp_mask"] = int(temp_mask[w])
        row["hrv_dt"] = float(hrv_dt[w])
        row["temp_dt"] = float(temp_dt[w])
        rows.append(row)
    return rows, None


def _self_check() -> None:
    """resolved 02–06 + mid4h window + file-length mid4h + midnight temp."""
    md = pd.Timestamp("2023-06-12")
    w0 = window_index_0206(md)[0]
    assert w0 == pd.Timestamp("2023-06-12 02:00:00"), w0
    assert window_end_0206(md) == pd.Timestamp("2023-06-12 06:00:00")
    # mid4h: 22:00–06:00 (8h file) → mid 02:00 → 00:00~03:50
    n8 = FS * 3600 * 8
    mid_idx = window_index_mid4h(
        pd.Timestamp("2023-06-12 22:00:00"),
        pd.Timestamp("2023-06-13 06:00:00"),
        n_samples=n8,
    )
    assert mid_idx[0] == pd.Timestamp("2023-06-13 00:00:00"), mid_idx[0]
    assert len(mid_idx) == N_WIN
    # map 8h, file 3h → file-based window (21:30)
    mid_short = window_index_mid4h(
        pd.Timestamp("2023-06-12 22:00:00"),
        pd.Timestamp("2023-06-13 06:00:00"),
        n_samples=FS * 3600 * 3,
    )
    assert mid_short[0] == pd.Timestamp("2023-06-12 21:30:00"), mid_short[0]
    # 0206: resolved=D, recording D 23:00–D+1 07 → mismatch with D 02–06 → skip
    w_bad, err_bad = resolve_window_index(
        md,
        pd.Timestamp("2023-06-12 23:00:00"),
        pd.Timestamp("2023-06-13 07:00:00"),
        "0206",
    )
    assert err_bad == "no_0206_overlap" and w_bad is None
    # if resolved=D+1, same recording → D+1 02–06 (label and window agree)
    w0206, err = resolve_window_index(
        pd.Timestamp("2023-06-13"),
        pd.Timestamp("2023-06-12 23:00:00"),
        pd.Timestamp("2023-06-13 07:00:00"),
        "0206",
    )
    assert err is None and w0206 is not None
    assert w0206[0] == pd.Timestamp("2023-06-13 02:00:00"), w0206[0]
    # temp must survive in a mid4h window that crosses midnight
    cross_idx = window_index_mid4h(
        pd.Timestamp("2023-06-12 21:00:00"),
        pd.Timestamp("2023-06-13 05:00:00"),
        n_samples=FS * 3600 * 8,
    )
    assert cross_idx[0] == pd.Timestamp("2023-06-12 23:00:00"), cross_idx[0]
    raw_t = pd.DataFrame(
        {"valueC": [36.5, 36.6]},
        index=pd.to_datetime(["2023-06-12 23:05:00", "2023-06-13 02:05:00"]),
    )
    t24 = temp_24_for_windows(raw_t, cross_idx)
    assert np.isfinite(t24[0]) and np.isfinite(t24[18]), t24
    print("24 self-check: 0206=resolved day only + mid4h file-span + midnight-crossing temp OK")


def _job_24(job: tuple) -> tuple[str | None, list[dict]]:
    """ProcessPool: (code, patient, dt, p_ecg, p_tmp, start, end, window_mode)."""
    code, patient, dt, p_ecg, p_tmp, t0, t1, mode = job
    if not p_ecg:
        return "no_ecg", []
    if not p_tmp:
        return "no_temp", []
    rows, reason = build_rows_for_code(
        code, patient, dt, p_ecg, p_tmp, ecg_start=t0, ecg_end=t1, window_mode=mode
    )
    return reason, rows


def main(workers: int = 8, window_mode: str | None = None, out_path: Path | None = None):
    """If window_mode/out_path are omitted, use module constants or env (OVUL_24_*)."""
    global OUT_PATH, WINDOW_MODE
    workers = max(1, int(workers))
    if window_mode is None:
        window_mode = os.environ.get("OVUL_24_WINDOW", WINDOW_MODE)
    WINDOW_MODE = str(window_mode)
    if WINDOW_MODE not in ("0206", "mid4h"):
        raise ValueError(f"window_mode={WINDOW_MODE!r}")
    if out_path is not None:
        OUT_PATH = Path(out_path)
    elif os.environ.get("OVUL_24_OUT"):
        OUT_PATH = PRE_DIR / os.environ["OVUL_24_OUT"]
    elif WINDOW_MODE == "mid4h":
        OUT_PATH = PRE_DIR / "ovul_data_24mid.xlsx"
    else:
        OUT_PATH = PRE_DIR / "ovul_data_24.xlsx"
    _self_check()
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MAP_PATH: {MAP_PATH}")
    print(f"Night window: {WINDOW_MODE} → {OUT_PATH.name}")
    print(f"workers={workers}")
    mapping = load_mapping_qc()
    ecg_files = glob.glob(str(BASE_DIR / "*" / "CHECKUP_*.csv"))
    temp_files = glob.glob(str(BASE_DIR / "*" / "temp_*.json"))
    ecg_idx, temp_buckets, n_temp_indexed = build_raw_indexes(ecg_files, temp_files)
    print(f"ECG csv: {len(ecg_files)}, temp json: {len(temp_files)}, QC mapping rows: {len(mapping)}")
    print(
        f"ECG index keys: {len(ecg_idx)}, "
        f"temp files (indexed): {n_temp_indexed}, temp date-time buckets: {len(temp_buckets)}"
    )

    jobs = []
    for _, r in mapping.iterrows():
        code = str(r["file_name"]).strip()
        jobs.append(
            (
                code,
                str(r["participant_id"]).strip(),
                r["date"],
                find_ecg_path(ecg_idx, code),
                find_temp_path(temp_buckets, code),
                r.get("start_time"),
                r.get("end_time"),
                WINDOW_MODE,
            )
        )

    all_rows: list[dict] = []
    n_ok_meas = 0
    skip_counts: dict[str, int] = {}

    def _consume(reason: str | None, rows: list[dict]) -> None:
        nonlocal n_ok_meas
        if reason:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            return
        all_rows.extend(rows)
        n_ok_meas += 1

    desc = f"Build master (24_{WINDOW_MODE})"
    if workers <= 1:
        for job in tqdm(jobs, desc=desc, unit="mapping_row", dynamic_ncols=True):
            _consume(*_job_24(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_job_24, j) for j in jobs]
            for fut in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=desc,
                unit="mapping_row",
                dynamic_ncols=True,
            ):
                _consume(*fut.result())

    if not all_rows:
        raise RuntimeError(
            "No rows produced. Check the mapping file_name against file names in the raw data folder."
        )

    df = pd.DataFrame(all_rows)
    col_order = (
        ["participant_id", "window", "file_name_x", "cycle", "measure_date"]
        + HRV_COLS
        + TEMP_COLS
        + MASK_COLS
        + ["measure_time", "ovulation_flag"]
    )
    df = df[[c for c in col_order if c in df.columns]]
    if "file_name_x" in df.columns and "window" in df.columns:
        df = df.sort_values(["file_name_x", "window"], kind="mergesort").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT_PATH, index=False, engine="openpyxl")
    print(
        f"Saved: {OUT_PATH}  (rows {len(df)}, measurements {n_ok_meas} nights)  "
        f"Skipped: {dict(sorted(skip_counts.items(), key=lambda x: -x[1]))}"
    )
    print("Note: values are saved with NaN as is. Interpolation and per-subject normalization happen in the training loader (using mask/dt).")
    print(f"Next: 01 input → {OUT_PATH.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="24 slots: resolved day 02:00–06:00")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(workers=args.workers)



