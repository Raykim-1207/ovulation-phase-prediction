# -*- coding: utf-8 -*-
"""
00_* shared: mapping/ECG/temperature loading, RR/HRV computation + **raw QC first**.

Usage order
---------
1) ``python _raw_common.py``  → ``pre/qc_keep_codes.csv`` / ``qc_drop_reasons.csv``
   (drops: no ECG / no temperature / recording <MIN_RECORD_MIN(240) min / flatline / outside closed cycle)
   measure_date ``date`` = compare CSV ``resolved_date`` (not overwritten)
2) each ``00_master_*.py``  → aggregates only QC-passed codes

For raw ECG (csv) + temp (json) paths. Style-specific aggregation lives in each 00_*.

measure_date(``date``)
---------------
compare CSV's ``resolved_date`` is fixed as ``date`` (QC keep does not overwrite it).
The menses calendar is loaded here; cycles without ``next_menses_start`` (and measurements in them) are removed in QC.
valid day = both ECG and temp present + dur≥MIN_RECORD_MIN.
Naive times are taken as KST; tz-aware times are converted to Asia/Seoul then made naive.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FS = 256
SEG_MIN = 10
SAMPLES_10MIN = FS * 60 * SEG_MIN
# 24 / 24mid calendar and center windows (same as the per-night 0206/mid4h trim)
NIGHT_START_HOUR = 2
NIGHT_END_HOUR = 6
MID4H_HALF = timedelta(hours=2)
WINDOW_MODES = ("sleep", "0206", "mid4h")
# 10-min window: skip just that window if the zero fraction exceeds this
FLATLINE_FRAC = 0.5
# ponytail: whole-file zero_frac>0.5 → drop the night. Clifford fSQI (flatline %) /
# Bukhari flatline=absent idea lifted to the recording level; empirical threshold
# (same value as the per-window FLATLINE_FRAC). Upgrade: span-level or SQI ensemble.
FILE_ZERO_FRAC = 0.5
# same policy as mapping ge240.
MIN_RECORD_MIN = 240.0

KST = "Asia/Seoul"

# resolved date source (compare = source of truth for dates)
OVUL_ROOT = Path(os.environ.get("OVUL_RAW_ROOT", "").strip() or Path(__file__).resolve().parents[2])
MAP_PATH = Path(os.environ.get("OVUL_MAP_CSV", "").strip() or OVUL_ROOT / "measure_code_map.csv")
MAP_EXCEL = MAP_PATH  # import compatibility for downstream scripts
MAP_SRC = MAP_PATH
# Root of raw ECG/temperature files. Not in the public repo — set via OVUL_RAW_DIR.
BASE_DIR = Path(os.environ.get("OVUL_RAW_DIR", "").strip() or OVUL_ROOT / "raw")
# intermediate/final outputs: ovul_data/ (00–02 under pre/)
_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "ovul_data"
# The output root can be moved via an env var. To avoid overwriting old outputs when re-running
# modified preprocessing, set a new directory with OVUL_DATA_DIR_ROOT. QC output (qc_keep_codes)
# depends only on raw data and needs no recomputation, so the old one is read if missing in the new root.
_env_root = os.environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_DIR = Path(_env_root) if _env_root else _DEFAULT_OUT

PRE_DIR = OUT_DIR / "pre"
QC_KEEP_PATH = PRE_DIR / "qc_keep_codes.csv"
QC_DROP_PATH = PRE_DIR / "qc_drop_reasons.csv"
if _env_root and not QC_KEEP_PATH.is_file():      # borrow QC from the old root
    for _old in (_DEFAULT_OUT,):
        if (_old / "pre" / "qc_keep_codes.csv").is_file():
            QC_KEEP_PATH = _old / "pre" / "qc_keep_codes.csv"
            QC_DROP_PATH = _old / "pre" / "qc_drop_reasons.csv"
            break
LABEL_DIR = Path(__file__).resolve().parents[2] / "label"
# Per-hospital consolidated ovulation-date workbook (Excel) and sheet name. Actual names ship with the data, set via env vars.
# If the sheet is empty, the first non-empty sheet is read.
LABEL_FILES = (
    (os.environ.get("OVUL_LABEL_B", "hospital_B_ovulation_dates.xlsx"),
     os.environ.get("OVUL_LABEL_B_SHEET") or None),
    (os.environ.get("OVUL_LABEL_A", "hospital_A_ovulation_dates.xlsx"),
     os.environ.get("OVUL_LABEL_A_SHEET") or None),
)

RAW_STRIP_PREFIXES = ("CHECKUP_", "temp_")

HRV_BASE = [
    "avgHR", "maxNN", "avgNN", "sdnn", "nn50", "pnn50", "rmssd",
    "tp", "ulf", "vlf", "lf", "hf", "lfhf",
]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def norm_code(s: object) -> str:
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
    # remove all when prefixes overlap, like temp_CHECKUP_…
    changed = True
    while changed:
        changed = False
        for pref in RAW_STRIP_PREFIXES:
            if len(t) >= len(pref) and t[: len(pref)].lower() == pref.lower():
                t = t[len(pref) :]
                changed = True
                break
    return t.lower()


def build_raw_indexes(
    ecg_files: list[str],
    temp_files: list[str],
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]], int]:
    ecg_idx: dict[str, str] = {}
    n_ecg_dup = 0
    for p in ecg_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        nk = norm_code(stem)
        if not nk:
            continue
        if nk not in ecg_idx:
            ecg_idx[nk] = p
        else:
            n_ecg_dup += 1
    if n_ecg_dup:
        print(f"warn: ECG norm_code duplicated {n_ecg_dup} times (keeping first path)")

    temp_buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_temp = 0
    for p in temp_files:
        nk = norm_code(os.path.splitext(os.path.basename(p))[0])
        if not nk:
            continue
        bucket = nk.split("_", 1)[0]
        temp_buckets[bucket].append((nk, p))
        n_temp += 1
    return ecg_idx, dict(temp_buckets), n_temp


def find_ecg_path(ecg_idx: dict[str, str], code: str) -> str | None:
    return ecg_idx.get(norm_code(code))


def find_temp_path(temp_buckets: dict[str, list[tuple[str, str]]], code: str) -> str | None:
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


def as_kst_naive(v: object) -> pd.Timestamp:
    """tz-aware → KST naive, naive → assumed KST (unchanged)."""
    t = pd.to_datetime(v, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        return t.tz_convert(KST).tz_localize(None)
    return t


def series_kst_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.map(as_kst_naive), errors="coerce")


# ---------------------------------------------------------------------------
# Menses calendar (shared with 01). Cycles without nmsd are excluded from here on.
# ---------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def _norm_cell_id(v) -> str | None:
    if pd.isna(v):
        return None
    if isinstance(v, float) and abs(v - round(v)) < 1e-9 and not np.isnan(v):
        return str(int(round(v)))
    s = str(v).strip()
    if not s or s in ("...", "…", "nan", "None"):
        return None
    return s


def _to_date(cell_value):
    if cell_value is None or (isinstance(cell_value, float) and np.isnan(cell_value)):
        return None
    if isinstance(cell_value, datetime):
        return cell_value.date()
    if isinstance(cell_value, pd.Timestamp):
        if pd.isna(cell_value):
            return None
        return cell_value.date()
    if isinstance(cell_value, (int, float)):
        if isinstance(cell_value, float) and cell_value == int(cell_value):
            cell_value = int(cell_value)
        if isinstance(cell_value, int) and 1 <= cell_value <= 31:
            return None
        try:
            return (pd.to_datetime("1899-12-30") + pd.to_timedelta(cell_value, "D")).date()
        except Exception:
            return None
    if isinstance(cell_value, str):
        s = cell_value.strip()
        if s in ("", "...", "…", "-", "NA", "N/A", "nan"):
            return None
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None
    return None


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    cmap = {re.sub(r"\s+", "", c).lower(): c for c in df.columns}
    for name in candidates:
        key = re.sub(r"\s+", "", name).lower()
        if key in cmap:
            return cmap[key]
    return None


def load_cycle_calendar(label_dir: Path = LABEL_DIR) -> pd.DataFrame:
    """
    Hospital A/B menses/ovulation calendar.
    Required: menses_start, next_menses_start, ovulation_date — the cycle is dropped if any is missing.
    """
    parts: list[pd.DataFrame] = []
    for fname, prefer_sheet in LABEL_FILES:
        path = Path(label_dir) / fname
        if not path.is_file():
            print(f"  [skip] no calendar: {path}")
            continue
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheet = prefer_sheet
        if sheet is None or sheet not in xl.sheet_names:
            sheet = xl.sheet_names[0]
            for s in xl.sheet_names:
                peek = pd.read_excel(path, sheet_name=s, nrows=5, engine="openpyxl")
                if not peek.empty:
                    sheet = s
                    break
        raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        raw = _normalize_columns(raw)
        pid_col = _resolve_col(raw, "Screening No.", "ScreeningNo.", "participant_id")
        msd_col = _resolve_col(raw, "menses_start")
        nmsd_col = _resolve_col(raw, "next_menses_start", "next_menses_start")
        ov_col = _resolve_col(raw, "ovulation_date", "ovulation_date")
        cyc_col = _resolve_col(raw, "cycle")
        if not all([pid_col, msd_col, ov_col]):
            raise KeyError(f"{path.name}/{sheet}: missing required columns. cols={list(raw.columns)}")
        d = pd.DataFrame(
            {
                "participant_id": raw[pid_col].map(_norm_cell_id),
                "cycle": pd.to_numeric(raw[cyc_col], errors="coerce") if cyc_col else np.nan,
                "menses_start": raw[msd_col].map(_to_date),
                "next_menses_start": raw[nmsd_col].map(_to_date) if nmsd_col else None,
                "ovulation_date": raw[ov_col].map(_to_date),
                "_source": f"{fname}:{sheet}",
            }
        )
        d = d[d["participant_id"].notna()].copy()
        parts.append(d)
        print(f"  [calendar] {fname}/{sheet}: {len(d)} rows, {d['participant_id'].nunique()} patients")

    if not parts:
        raise FileNotFoundError(f"calendar file not found: {label_dir}")
    rel = pd.concat(parts, ignore_index=True)
    rel["_has_ov"] = rel["ovulation_date"].notna().astype(int)
    rel = (
        rel.sort_values(["participant_id", "cycle", "_has_ov"], ascending=[True, True, False])
        .drop_duplicates(subset=["participant_id", "cycle"], keep="first")
        .drop(columns=["_has_ov"])
    )
    before = len(rel)
    closed = (
        rel["menses_start"].notna()
        & rel["ovulation_date"].notna()
        & rel["next_menses_start"].notna()
    )
    rel = rel.loc[closed].copy()
    print(
        f"  [calendar] merged {before} cycles → closed cycles (msd+nmsd+ov) {len(rel)} "
        f"({rel['participant_id'].nunique()} patients, excluded {before - len(rel)})"
    )
    return rel


def build_cycles_by_patient(rel: pd.DataFrame) -> dict:
    cycles: dict = {}
    for pid, sub in rel.groupby("participant_id"):
        lst = []
        for _, row in sub.iterrows():
            lst.append(
                {
                    "cycle": row["cycle"],
                    "msd": row["menses_start"],
                    "nmsd": row["next_menses_start"],
                    "ov_date": row["ovulation_date"],
                }
            )
        cycles[str(pid)] = lst
    return cycles


def date_in_closed_cycle(pid: str, d: date | None, cycles_by_patient: dict) -> bool:
    """True if the measure date lies inside the closed cycle [msd, nmsd)."""
    if d is None or pid not in cycles_by_patient:
        return False
    for c in cycles_by_patient[pid]:
        msd, nmsd = c["msd"], c["nmsd"]
        if msd is None or nmsd is None:
            continue
        if msd <= d < nmsd:
            return True
    return False


def filter_mapping_closed_cycles(
    mapping: pd.DataFrame,
    cycles_by_patient: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only measurements inside closed cycles. (keep_df, drop_df)."""
    if cycles_by_patient is None:
        cycles_by_patient = build_cycles_by_patient(load_cycle_calendar())
    dates = pd.to_datetime(mapping["date"], errors="coerce").dt.date
    pids = mapping["participant_id"].astype(str)
    ok = [
        date_in_closed_cycle(str(pid), d, cycles_by_patient)
        for pid, d in zip(pids, dates)
    ]
    mask = pd.Series(ok, index=mapping.index)
    return mapping.loc[mask].copy(), mapping.loc[~mask].copy()


def resolve_dup_measure_dates(df: pd.DataFrame) -> pd.Series:
    """
    Default: mid(start, end).
    If ``participant_id+date`` is duplicated, use start.
    If still duplicated, use end (chained collisions re-evaluated repeatedly).
    +N (pushing the calendar day forward) is the last-resort safeguard, so
    use only `+1` where possible (if `+2` would be needed, retry with start instead of mid as the base),
    and only push further if that still fails, guaranteeing zero duplicates.
    """
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.normalize().copy()

    pid = df["participant_id"].astype(str)
    start = pd.to_datetime(df["start_time"], errors="coerce")
    end = pd.to_datetime(df["end_time"], errors="coerce")
    start_d = start.dt.normalize()
    end_d = end.dt.normalize()
    mid_d = (start + (end - start) / 2).dt.normalize()
    base_d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    def _dup_mask(dates: pd.Series) -> pd.Series:
        return (
            pd.DataFrame({"pid": pid, "d": dates})
            .groupby(["pid", "d"], dropna=False)["d"]
            .transform("size")
            > 1
        )

    def _try_resolve(base_mode: str, k_cap: int | None) -> pd.Series | None:
        """
        base_mode: 'mid' or 'start'
        k_cap: maximum k used for anchor+(k) when forcing uniqueness.
               None means unbounded (the old +N behavior).
        """
        if base_mode == "mid":
            out = mid_d.copy()
            out = out.where(out.notna(), base_d)
            # on mid collision use start
            m = _dup_mask(out) & start_d.notna()
            out.loc[m] = start_d.loc[m]
        elif base_mode == "start":
            out = start_d.copy()
            out = out.where(out.notna(), base_d)
        else:
            raise ValueError(f"unknown base_mode: {base_mode}")

        # on collision move to end; chained collisions are re-evaluated repeatedly
        for _ in range(12):
            m = _dup_mask(out) & end_d.notna()
            if not m.any():
                break
            prev = out.copy()
            out.loc[m] = end_d.loc[m]
            if out.equals(prev):
                break

        if not _dup_mask(out).any():
            return out

        # 3) still duplicate -> force unique.
        # with k_cap only anchor+1 (or up to cap) is allowed. Return None if impossible.
        dup = _dup_mask(out)
        tmp = pd.DataFrame({"pid": pid, "d": out, "i": out.index})
        for p, gp in tmp.groupby("pid", dropna=False):
            idx_all = gp["i"].tolist()
            if len(idx_all) <= 1:
                continue

            # for this patient, include already-occupied non-duplicate dates
            occupied = set(
                out[(pid == p) & ~out.index.isin(idx_all)].dropna().tolist()
            )
            idx_order = sorted(
                idx_all,
                key=lambda j: (
                    out.loc[j] if pd.notna(out.loc[j]) else pd.Timestamp.max,
                    start.loc[j] if pd.notna(start.loc[j]) else pd.Timestamp.max,
                ),
            )
            for i in idx_order:
                cands = [out.loc[i], mid_d.loc[i], start_d.loc[i], end_d.loc[i]]
                pick = None
                for c in cands:
                    if pd.notna(c) and c not in occupied:
                        pick = c
                        break

                if pick is None:
                    anchor = end_d.loc[i]
                    if pd.isna(anchor):
                        anchor = out.loc[i]
                    if pd.isna(anchor):
                        anchor = pd.Timestamp("1970-01-01")

                    if k_cap is None:
                        # unbounded (last-resort safeguard)
                        k = 1
                        pick = anchor + pd.Timedelta(days=k)
                        while pick in occupied:
                            k += 1
                            pick = anchor + pd.Timedelta(days=k)
                    else:
                        pick = None
                        for k in range(1, k_cap + 1):
                            cand = anchor + pd.Timedelta(days=k)
                            if cand not in occupied:
                                pick = cand
                                break
                        if pick is None:
                            return None

                out.loc[i] = pick
                occupied.add(pick)

        if _dup_mask(out).any():
            return None
        return out

    # 1) try with mid as the base (forced uniqueness only up to +1)
    out = _try_resolve(base_mode="mid", k_cap=1)
    if out is not None and not _dup_mask(out).any():
        return out

    # 2) if that fails, retry with start as the base (only up to +1)
    out = _try_resolve(base_mode="start", k_cap=1)
    if out is not None and not _dup_mask(out).any():
        return out

    # 3) last resort: the old unbounded +N to guarantee zero duplicates
    out = _try_resolve(base_mode="start", k_cap=None)
    if out is None:
        # unreachable in theory (calendar is unbounded), guarded anyway
        return pd.to_datetime(df["date"], errors="coerce").dt.normalize().copy()
    return out


def mapping_duration_min(df: pd.DataFrame) -> pd.Series:
    """Duration (min) of mapping ``start_time``~``end_time``. NaN if missing."""
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    start = pd.to_datetime(df["start_time"], errors="coerce")
    end = pd.to_datetime(df["end_time"], errors="coerce")
    return (end - start).dt.total_seconds() / 60.0


def _n_dup_measure_rows(frame: pd.DataFrame) -> int:
    if frame.empty or "date" not in frame.columns:
        return 0
    return int(
        frame.assign(_pid=frame["participant_id"].astype(str))
        .groupby(["_pid", "date"], dropna=False)["date"]
        .transform("size")
        .gt(1)
        .sum()
    )


def load_mapping(map_path: Path = MAP_PATH) -> pd.DataFrame:
    """Load mapping (CSV/xlsx). If ``resolved_date`` exists, use it as measure_date ``date``.
    Without a date column, seed from mid(start, end) (duplicates resolved later in QC)."""
    path = Path(map_path)
    if path.suffix.lower() == ".csv":
        df = None
        last_err: Exception | None = None
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
        if df is None:
            raise last_err or UnicodeDecodeError("utf-8", b"", 0, 1, "decode failed")
    else:
        df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if "file_name" not in df.columns and "measure_code" in df.columns:
        df["file_name"] = df["measure_code"]
    if "start_time" in df.columns:
        df["start_time"] = series_kst_naive(df["start_time"])
    if "end_time" in df.columns:
        df["end_time"] = series_kst_naive(df["end_time"])
    if "resolved_date" in df.columns:
        df["date"] = df["resolved_date"]
    elif "date" not in df.columns:
        if "orig_date" in df.columns:
            df["date"] = df["orig_date"]
        elif "start_time" in df.columns and "end_time" in df.columns:
            mid = (df["start_time"] + (df["end_time"] - df["start_time"]) / 2).dt.normalize()
            df["date"] = mid.where(mid.notna(), df["start_time"].dt.normalize())
    need = {"file_name", "participant_id", "date"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"mapping missing columns: {miss} ({path})")
    df = df.dropna(subset=["file_name", "participant_id"]).copy()
    df["date"] = series_kst_naive(df["date"]).dt.normalize()
    return df


@lru_cache(maxsize=64)
def _load_ecg_values_cached(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path, dtype=np.float64)
    except Exception:
        return np.genfromtxt(path, dtype=np.float64)


def load_ecg(path: str | Path) -> np.ndarray:
    return _load_ecg_values_cached(str(path))


def zero_frac(vals: np.ndarray) -> float:
    """Fraction of samples in the whole array that are exactly 0.0."""
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return 1.0
    return float(np.mean(a == 0.0))


def zero_frac_file_fast(path: str, fs: int = FS) -> float:
    """
    Whole-file zero fraction (approximated at 1 sample per second — no full-sample loadtxt).
    Same resolution as build_drop_causes_raw.analyze_zero_fast.
    """
    n = z = 0
    line_i = 0
    with open(path, "rb") as f:
        leftover = b""
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                if leftover and line_i % fs == 0:
                    try:
                        v = float(leftover.strip() or b"nan")
                    except ValueError:
                        v = float("nan")
                    n += 1
                    z += int(v == 0.0)
                break
            data = leftover + chunk
            pos = 0
            while True:
                nl = data.find(b"\n", pos)
                if nl < 0:
                    leftover = data[pos:]
                    break
                if line_i % fs == 0:
                    try:
                        v = float(data[pos:nl])
                    except ValueError:
                        v = float("nan")
                    n += 1
                    z += int(v == 0.0)
                line_i += 1
                pos = nl + 1
    return (z / n) if n else 1.0


def _qc_one_job(args: tuple) -> dict:
    """ProcessPool worker: (code, patient, date_str, ecg_path|None, tmp_path|None)."""
    code, patient, date_s, p_ecg, p_tmp = args
    base = {
        "participant_id": patient,
        "file_name": code,
        "date": date_s,
        "ecg_path": p_ecg or "",
        "temp_path": p_tmp or "",
        "zero_frac": np.nan,
    }
    if not p_ecg:
        return {**base, "keep": 0, "reason": "no_ecg"}
    if not p_tmp:
        return {**base, "keep": 0, "reason": "no_temp"}
    try:
        zf = zero_frac_file_fast(p_ecg)
    except Exception:
        return {**base, "keep": 0, "reason": "ecg_load_error"}
    base["zero_frac"] = round(float(zf), 6)
    if zf > FILE_ZERO_FRAC:
        return {**base, "keep": 0, "reason": "too_much_flatline"}
    return {**base, "keep": 1, "reason": "kept"}


def run_raw_qc(workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Raw QC over the full mapping → keep/drop CSV.
    drop: no ECG/temperature · recording < MIN_RECORD_MIN · flatline · outside closed cycle.
    measure_date ``date`` = compare CSV ``resolved_date`` (not overwritten).
    """
    from tqdm.auto import tqdm

    mapping = load_mapping()
    cal = load_cycle_calendar()
    cycles = build_cycles_by_patient(cal)
    mapping, cal_drop = filter_mapping_closed_cycles(mapping, cycles)
    print(
        f"calendar filter: dropped {len(cal_drop)} rows outside closed cycles "
        f"({len(mapping)} mapping rows left)"
    )

    ecg_files, temp_files = list_raw_files()
    ecg_idx, temp_buckets, n_temp = build_raw_indexes(ecg_files, temp_files)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MAP_PATH: {MAP_PATH}")
    print(f"ECG csv: {len(ecg_files)}, temp json: {len(temp_files)}, mapping: {len(mapping)}")
    print(f"ECG keys: {len(ecg_idx)}, temp indexed: {n_temp}")
    print(
        f"QC: no_ecg / no_temp / dur<{MIN_RECORD_MIN:.0f}min / "
        f"zero_frac>{FILE_ZERO_FRAC} / no_closed_cycle → drop  (workers={workers})"
    )

    drop_rows: list[dict] = []
    for i in cal_drop.index:
        r = cal_drop.loc[i]
        code = str(r["file_name"]).strip()
        drop_rows.append(
            {
                "participant_id": str(r["participant_id"]).strip(),
                "file_name": code,
                "date": str(r["date"]),
                "ecg_path": "",
                "temp_path": "",
                "zero_frac": np.nan,
                "keep": 0,
                "reason": "no_closed_cycle",
            }
        )

    dur = mapping_duration_min(mapping)
    short_mask = dur.isna() | (dur < MIN_RECORD_MIN)

    for i in mapping.index[short_mask]:
        r = mapping.loc[i]
        code = str(r["file_name"]).strip()
        drop_rows.append(
            {
                "participant_id": str(r["participant_id"]).strip(),
                "file_name": code,
                "date": str(r["date"]),
                "ecg_path": find_ecg_path(ecg_idx, code) or "",
                "temp_path": find_temp_path(temp_buckets, code) or "",
                "zero_frac": np.nan,
                "keep": 0,
                "reason": "too_short",
            }
        )

    cand = mapping.loc[~short_mask]
    jobs = []
    for _, r in cand.iterrows():
        code = str(r["file_name"]).strip()
        jobs.append(
            (
                code,
                str(r["participant_id"]).strip(),
                str(r["date"]),
                find_ecg_path(ecg_idx, code),
                find_temp_path(temp_buckets, code),
            )
        )

    rows: list[dict] = list(drop_rows)
    if workers <= 1:
        for job in tqdm(jobs, desc="raw QC", unit="meas"):
            rows.append(_qc_one_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_qc_one_job, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="raw QC", unit="meas"):
                rows.append(fut.result())

    df = pd.DataFrame(rows)
    keep = df[df["keep"] == 1].copy()
    drop = df[df["keep"] == 0].copy()

    # keep: retain resolved date (reattach dates from the mapping to the QC CSV)
    if len(keep):
        keep["file_name"] = keep["file_name"].astype(str).str.strip()
        keep_codes = set(keep["file_name"])
        meta = mapping.copy()
        meta["file_name"] = meta["file_name"].astype(str).str.strip()
        resolved = meta.loc[meta["file_name"].isin(keep_codes)].copy()
        info = keep.set_index("file_name")[["zero_frac", "ecg_path", "temp_path"]]
        resolved = resolved.join(info, on="file_name")
        # dates are the mapping resolved values as is. Only warn on duplicates.
        n_dup = _n_dup_measure_rows(resolved)
        if n_dup:
            print(f"warning: patient×resolved_date duplicated in {n_dup} rows (check compare CSV)")
        keep = resolved

    PRE_DIR.mkdir(parents=True, exist_ok=True)
    keep_out = keep[["participant_id", "file_name", "date", "zero_frac", "ecg_path", "temp_path"]].copy()
    keep_out["date"] = pd.to_datetime(keep_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    drop_out = drop[["participant_id", "file_name", "date", "reason", "zero_frac", "ecg_path", "temp_path"]]
    keep_out.to_csv(QC_KEEP_PATH, index=False, encoding="utf-8-sig")
    drop_out.to_csv(QC_DROP_PATH, index=False, encoding="utf-8-sig")
    print(f"keep: {len(keep_out)} → {QC_KEEP_PATH}")
    print(f"drop: {len(drop_out)} → {QC_DROP_PATH}")
    if len(drop_out):
        print(drop_out["reason"].value_counts().to_string())
    return keep_out, drop_out


def load_qc_keep_codes() -> set[str]:
    """Keep set produced by ``run_raw_qc``. Fails with instructions if missing."""
    if not QC_KEEP_PATH.is_file():
        raise FileNotFoundError(
            f"{QC_KEEP_PATH} missing. Run first:  python {Path(__file__).name}"
        )
    s = pd.read_csv(QC_KEEP_PATH, usecols=["file_name"])["file_name"].astype(str).str.strip()
    return set(s)


def load_mapping_qc() -> pd.DataFrame:
    """Mapping ∩ QC keep. measure_date (date) is always MAP resolved_date (QC CSV date ignored)."""
    if not QC_KEEP_PATH.is_file():
        raise FileNotFoundError(
            f"{QC_KEEP_PATH} missing. Run first:  python {Path(__file__).name}"
        )
    m = load_mapping()
    keep_codes = load_qc_keep_codes()
    m = m.copy()
    m["file_name"] = m["file_name"].astype(str).str.strip()
    out = m.loc[m["file_name"].isin(keep_codes)].copy()
    print(f"QC keep: {len(keep_codes)} codes → of {len(m)} mapping rows, {len(out)} rows kept (date=resolved)")
    n_dup = _n_dup_measure_rows(out)
    if n_dup:
        print(f"measure_date duplicates remaining (keep): {n_dup} rows")
    return out


def load_temp_json(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["dateTime"] = series_kst_naive(df["dateTime"])
    df = df.dropna(subset=["dateTime"])
    df = df.sort_values("dateTime").set_index("dateTime")
    df["valueC"] = pd.to_numeric(df["valueC"], errors="coerce")
    return df[["valueC"]]


def bandpass_ecg(x: np.ndarray, fs: int = FS, low: float = 5.0, high: float = 25.0) -> np.ndarray:
    sos = butter(3, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def rr_from_filtered(x: np.ndarray, fs: int = FS) -> np.ndarray | None:
    """R-R (ms) from a band-passed 10-min segment.

    The old criteria (prominence = 0.3*std, refractory 0.25 s) had a low threshold and counted
    T waves as beats. Mean nocturnal heart rate rose to 106 bpm and RMSSD exceeded SDNN on 90 %
    of nights — values impossible in a sleeping person. In a check on 34 measured nights, the criteria
    below gave 60.6 bpm and RMSSD>SDNN in 28 %, back in the textbook range.

      · refractory 0.40 s = 150 bpm ceiling. No sinus rhythm exceeds this during sleep.
      · threshold is half of the **R-wave amplitude itself** (98th percentile), not the signal std.
        std mixes in T waves and baseline wander, so it cannot anchor an R-wave threshold.

    After dropping out-of-range RR, the array is no longer a contiguous beat sequence. Differencing
    across the gaps inflates RMSSD, so instead of dropping, split and return only the longest
    intact run.
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
    # longest intact run. Keeps differences from jumping over gaps.
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


def hrv_freq(rr_ms: np.ndarray) -> dict[str, float]:
    rr = np.asarray(rr_ms, dtype=float) / 1000.0
    empty = {k: np.nan for k in ("tp", "ulf", "vlf", "lf", "hf", "lfhf")}
    if rr.size < 8:
        return empty
    t = np.concatenate([[0], np.cumsum(rr[:-1])])
    t_end = t[-1] + rr[-1]
    if t_end <= 0:
        return empty
    fs_r = 4.0
    t_u = np.arange(0.0, t_end, 1.0 / fs_r)
    if t_u.size < 32:
        return empty
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


def recording_span(
    ecg_start: pd.Timestamp,
    n_samples: int,
    ecg_end: Optional[pd.Timestamp] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    ecg_start = pd.Timestamp(ecg_start)
    file_end = ecg_start + timedelta(seconds=n_samples / FS)
    if ecg_end is not None and pd.notna(ecg_end):
        end_raw = min(file_end, pd.Timestamp(ecg_end))
    else:
        end_raw = file_end
    return ecg_start, end_raw


def trimmed_span(
    ecg_start: pd.Timestamp,
    n_samples: int,
    head_min: float,
    tail_min: float,
    ecg_end: Optional[pd.Timestamp] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Recording [start,end] → [start+head, end−tail]."""
    start, end_raw = recording_span(ecg_start, n_samples, ecg_end=ecg_end)
    tstart = start + timedelta(minutes=head_min)
    tend = end_raw - timedelta(minutes=tail_min)
    return tstart, tend


def span_0206(measure_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar day D → D 02:00–06:00 (same as 24)."""
    base = pd.Timestamp(measure_date).normalize()
    return base + timedelta(hours=NIGHT_START_HOUR), base + timedelta(hours=NIGHT_END_HOUR)


def _overlap_minutes(
    a0: pd.Timestamp,
    a1: pd.Timestamp,
    b0: pd.Timestamp,
    b1: pd.Timestamp,
) -> float:
    lo = max(pd.Timestamp(a0), pd.Timestamp(b0))
    hi = min(pd.Timestamp(a1), pd.Timestamp(b1))
    return max(0.0, (hi - lo).total_seconds() / 60.0)


def resolve_0206_span(
    measure_date: pd.Timestamp,
    ecg_start: pd.Timestamp,
    ecg_end: Optional[pd.Timestamp],
) -> tuple[pd.Timestamp, pd.Timestamp, Optional[str]]:
    """
    0206 analysis window = **only 02:00–06:00 on the calendar day of resolved_date (measure_date)**.
    The label measure date and window date are the same. If it does not overlap the recording, ``no_0206_overlap``.
    """
    ws, we = span_0206(measure_date)
    start = pd.Timestamp(ecg_start)
    end = (
        pd.Timestamp(ecg_end)
        if ecg_end is not None and pd.notna(ecg_end)
        else start + timedelta(hours=12)
    )
    if _overlap_minutes(start, end, ws, we) <= 0:
        return ws, we, "no_0206_overlap"
    return ws, we, None


def span_mid4h(
    ecg_start: pd.Timestamp,
    ecg_end: Optional[pd.Timestamp],
    n_samples: int | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Recording center ±2h → 4 hours (same as 24mid).

    **Unified to file length**: if ``n_samples`` is given, use ``recording_span`` (map end ∩ file end).
    Prevents the window from leaving the samples when the map is long but the file short (missing HRV → fabricated interpolation).
    """
    if n_samples is not None and int(n_samples) > 0:
        start, end = recording_span(ecg_start, int(n_samples), ecg_end=ecg_end)
    else:
        start = pd.Timestamp(ecg_start)
        end = (
            pd.Timestamp(ecg_end)
            if ecg_end is not None and pd.notna(ecg_end)
            else start + timedelta(hours=8)
        )
    mid = start + (end - start) / 2
    tstart = (mid - MID4H_HALF).floor("10min")
    tend = tstart + timedelta(minutes=SEG_MIN * 24)
    return tstart, tend


def recording_overlaps_window(
    ecg_start: pd.Timestamp,
    ecg_end: Optional[pd.Timestamp],
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
) -> Optional[str]:
    """Skip reason only when the recording does not overlap the window at all."""
    ecg_start = pd.Timestamp(ecg_start)
    win_start = pd.Timestamp(win_start)
    win_end = pd.Timestamp(win_end)
    if ecg_start >= win_end:
        return "ecg_start_after_window"
    if ecg_end is not None and pd.notna(ecg_end):
        if pd.Timestamp(ecg_end) <= win_start:
            return "ecg_end_before_window"
    return None


def resolve_used_span(
    measure_date: pd.Timestamp,
    ecg_start: pd.Timestamp,
    n_samples: int,
    window_mode: str = "sleep",
    head_min: float = 30,
    tail_min: float = 30,
    ecg_end: Optional[pd.Timestamp] = None,
) -> tuple[pd.Timestamp, pd.Timestamp, Optional[str]]:
    """
    Analysis span [used_from, used_to].

    · sleep: trim head at sleep onset / tail at wake (head/tail per style)
    · 0206: 02–06 on the **resolved day** (skip if no overlap with the recording; label day = window day)
    · mid4h: center ±2h of the **actual file span**
    """
    mode = str(window_mode)
    if mode == "0206":
        return resolve_0206_span(measure_date, ecg_start, ecg_end)
    if mode == "mid4h":
        tstart, tend = span_mid4h(ecg_start, ecg_end, n_samples=n_samples)
        return tstart, tend, recording_overlaps_window(ecg_start, ecg_end, tstart, tend)
    if mode != "sleep":
        raise ValueError(f"window_mode={mode!r}")
    tstart, tend = trimmed_span(ecg_start, n_samples, head_min, tail_min, ecg_end=ecg_end)
    if tend <= tstart:
        return tstart, tend, "span_empty_after_trim"
    return tstart, tend, None


def hrv_windows_over_span(
    vals: np.ndarray,
    ecg_start: pd.Timestamp,
    tstart: pd.Timestamp,
    tend: pd.Timestamp,
    seg_min: int = SEG_MIN,
) -> list[dict[str, float]]:
    """Split the trimmed span into seg_min-minute windows and compute HRV per window."""
    ecg_start = pd.Timestamp(ecg_start)
    rows: list[dict[str, float]] = []
    if vals.size < FS * 5 or tend <= tstart:
        return rows
    x = bandpass_ecg(vals)
    step = timedelta(minutes=seg_min)
    n_samp = FS * 60 * seg_min
    w = pd.Timestamp(tstart)
    while w + step <= tend:
        off = int(round((w - ecg_start).total_seconds() * FS))
        w += step
        if off < 0 or off + n_samp > vals.size:
            continue
        seg_raw = vals[off : off + n_samp]
        if np.mean(seg_raw == 0.0) > FLATLINE_FRAC:
            continue
        rr = rr_from_filtered(x[off : off + n_samp])
        if rr is None:
            continue
        rows.append(hrv_time(rr))
    return rows


def temp_in_span(
    raw: Optional[pd.DataFrame],
    tstart: pd.Timestamp,
    tend: pd.Timestamp,
) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    sub = raw.loc[(raw.index >= tstart) & (raw.index < tend), "valueC"].dropna()
    return sub.astype(float)


def list_raw_files(base_dir: Path = BASE_DIR) -> tuple[list[str], list[str]]:
    # raw ECG only (avoid picking up other csv files)
    ecg = glob.glob(str(base_dir / "*" / "CHECKUP_*.csv"))
    temp = glob.glob(str(base_dir / "*" / "temp_*.json"))
    return ecg, temp


def _mapping_worker(job: tuple) -> tuple[str | None, dict | None]:
    """ProcessPool: (build_fn, code, patient, dt, p_ecg, p_tmp, start, end)."""
    build_fn, code, patient, dt, p_ecg, p_tmp, t0, t1 = job
    if not p_ecg:
        return "no_ecg", None
    if not p_tmp:
        return "no_temp", None
    row, reason = build_fn(code, patient, dt, p_ecg, p_tmp, t0, t1)
    if reason:
        return reason, None
    return None, row


def run_mapping_loop(
    build_fn,
    desc: str,
    workers: int = 8,
) -> tuple[list[dict], dict[str, int], int]:
    """Shared mapping loop. Iterates QC keep only. build_fn(...) → (row|None, reason)."""
    workers = max(1, int(workers))
    mapping = load_mapping_qc()
    ecg_files, temp_files = list_raw_files()
    ecg_idx, temp_buckets, n_temp = build_raw_indexes(ecg_files, temp_files)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MAP_PATH: {MAP_PATH}")
    print(f"ECG csv: {len(ecg_files)}, temp json: {len(temp_files)}, QC mapping rows: {len(mapping)}")
    print(f"ECG keys: {len(ecg_idx)}, temp indexed: {n_temp}, buckets: {len(temp_buckets)}")
    print(f"workers={workers}")

    from tqdm.auto import tqdm

    jobs = []
    for _, r in mapping.iterrows():
        code = str(r["file_name"]).strip()
        jobs.append(
            (
                build_fn,
                code,
                str(r["participant_id"]).strip(),
                r["date"],
                find_ecg_path(ecg_idx, code),
                find_temp_path(temp_buckets, code),
                r.get("start_time"),
                r.get("end_time"),
            )
        )

    all_rows: list[dict] = []
    skip_counts: dict[str, int] = {}

    def _consume(reason: str | None, row: dict | None) -> None:
        if reason:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            return
        if row is not None:
            all_rows.append(row)

    if workers <= 1:
        pbar = tqdm(jobs, desc=desc, unit="mapping_row", dynamic_ncols=True)
        for job in pbar:
            reason, row = _mapping_worker(job)
            _consume(reason, row)
            pbar.set_postfix(nights=len(all_rows), **{f"skip_{k}": v for k, v in skip_counts.items()})
        pbar.close()
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_mapping_worker, j) for j in jobs]
            for fut in tqdm(
                as_completed(futs), total=len(futs), desc=desc, unit="mapping_row", dynamic_ncols=True
            ):
                reason, row = fut.result()
                _consume(reason, row)

    skipped_no_raw = skip_counts.get("no_ecg", 0)
    print(
        f"done nights={len(all_rows)} skip={dict(sorted(skip_counts.items(), key=lambda x: -x[1]))}"
    )
    return all_rows, skip_counts, skipped_no_raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="raw QC → pre/qc_keep_codes.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--self-check", action="store_true", help="run unit smoke tests only and exit")
    args = ap.parse_args()
    if args.self_check:
        assert norm_code("temp_CHECKUP_ABC.csv") == "abc"
        assert norm_code("CHECKUP_20230530191705_61AOO") == "20230530191705_61aoo"
        assert zero_frac(np.array([])) == 1.0
        assert MIN_RECORD_MIN == 240.0
        demo_dur = pd.DataFrame(
            {
                "start_time": pd.to_datetime(["2023-06-12 01:00:00", "2023-06-12 01:00:00"]),
                "end_time": pd.to_datetime(["2023-06-12 01:30:00", "2023-06-12 08:00:00"]),
            }
        )
        dmin = mapping_duration_min(demo_dur)
        assert float(dmin.iloc[0]) == 30.0 and float(dmin.iloc[1]) == 420.0
        # drop short rows: 0.6 min + 7 h → keep only ≥240 min
        demo_short = pd.DataFrame(
            {
                "participant_id": ["P", "P"],
                "date": pd.to_datetime(["2023-06-12", "2023-06-12"]),
                "start_time": pd.to_datetime(["2023-06-12 01:14:53", "2023-06-12 01:16:12"]),
                "end_time": pd.to_datetime(["2023-06-12 01:15:29", "2023-06-12 08:04:33"]),
            }
        )
        long_only = demo_short.loc[mapping_duration_min(demo_short) >= MIN_RECORD_MIN]
        assert len(long_only) == 1
        # closed-cycle filter
        cycles = {
            "P": [
                {
                    "cycle": 1,
                    "msd": date(2023, 6, 1),
                    "nmsd": date(2023, 6, 29),
                    "ov_date": date(2023, 6, 14),
                }
            ]
        }
        assert date_in_closed_cycle("P", date(2023, 6, 12), cycles)
        assert not date_in_closed_cycle("P", date(2023, 6, 30), cycles)
        assert MAP_PATH.name == "measure_code_date_compare_ge240_806real.csv"
        ei, tb, nt = build_raw_indexes(
            ["/x/CHECKUP_a.csv", "/y/CHECKUP_a.csv"],
            ["/t/temp_CHECKUP_20230101120000_AB001.json"],
        )
        assert ei["a"] == "/x/CHECKUP_a.csv"
        assert nt == 1
        assert find_temp_path(tb, "20230101120000_AB") is not None
        # KST: UTC evening → next day KST
        utc = pd.Timestamp("2023-06-12 15:00:00", tz="UTC")
        assert as_kst_naive(utc) == pd.Timestamp("2023-06-13 00:00:00")
        assert as_kst_naive("2023-06-12 21:00:00") == pd.Timestamp("2023-06-12 21:00:00")
        md = pd.Timestamp("2023-06-12")
        s0206, e0206 = span_0206(md)
        assert s0206 == pd.Timestamp("2023-06-12 02:00:00")
        assert e0206 == pd.Timestamp("2023-06-12 06:00:00")
        sm, em = span_mid4h(
            pd.Timestamp("2023-06-12 22:00:00"),
            pd.Timestamp("2023-06-13 06:00:00"),
            n_samples=FS * 3600 * 8,
        )
        assert sm == pd.Timestamp("2023-06-13 00:00:00")
        assert em == sm + timedelta(hours=4)
        # map 8h, file 3h → mid4h uses the file span
        sm2, em2 = span_mid4h(
            pd.Timestamp("2023-06-12 22:00:00"),
            pd.Timestamp("2023-06-13 06:00:00"),
            n_samples=FS * 3600 * 3,
        )
        assert sm2 == pd.Timestamp("2023-06-12 21:30:00"), sm2
        assert em2 == sm2 + timedelta(hours=4)
        _, _, err = resolve_used_span(
            md,
            pd.Timestamp("2023-06-12 01:00:00"),
            FS * 3600 * 8,
            "0206",
            ecg_end=pd.Timestamp("2023-06-12 08:00:00"),
        )
        assert err is None
        # resolved=D, recording D 23:00–D+1 07 → no overlap with D 02–06 → skip (window is not moved to another day)
        _, _, err2 = resolve_used_span(
            md,
            pd.Timestamp("2023-06-12 23:00:00"),
            FS * 3600 * 8,
            "0206",
            ecg_end=pd.Timestamp("2023-06-13 07:00:00"),
        )
        assert err2 == "no_0206_overlap"
        # resolved=D+1: the same recording is OK
        t0, t1, err_ok = resolve_used_span(
            pd.Timestamp("2023-06-13"),
            pd.Timestamp("2023-06-12 23:00:00"),
            FS * 3600 * 8,
            "0206",
            ecg_end=pd.Timestamp("2023-06-13 07:00:00"),
        )
        assert err_ok is None
        assert t0 == pd.Timestamp("2023-06-13 02:00:00"), t0
        _, _, err3 = resolve_used_span(
            md,
            pd.Timestamp("2023-06-12 08:00:00"),
            FS * 3600,
            "0206",
            ecg_end=pd.Timestamp("2023-06-12 12:00:00"),
        )
        assert err3 == "no_0206_overlap"
        # duplicates: all separated by mid/start/end + forced uniqueness
        demo = pd.DataFrame(
            {
                "participant_id": ["P", "P", "P"],
                "date": pd.to_datetime(["2023-06-12"] * 3),
                "start_time": pd.to_datetime(
                    ["2023-06-12 01:14:53", "2023-06-12 01:16:12", "2023-06-12 23:37:52"]
                ),
                "end_time": pd.to_datetime(
                    ["2023-06-12 01:15:29", "2023-06-12 08:04:33", "2023-06-13 06:37:53"]
                ),
            }
        )
        got = resolve_dup_measure_dates(demo)
        assert got.nunique() == 3, got.tolist()
        assert _n_dup_measure_rows(demo.assign(date=got)) == 0
        # with mid as default, consecutive nights are separated first
        demo2 = pd.DataFrame(
            {
                "participant_id": ["Q", "Q"],
                "date": pd.to_datetime(["2023-11-03", "2023-11-03"]),
                "start_time": pd.to_datetime(
                    ["2023-11-02 21:28:07", "2023-11-03 20:10:31"]
                ),
                "end_time": pd.to_datetime(
                    ["2023-11-03 04:28:07", "2023-11-04 03:10:31"]
                ),
            }
        )
        got2 = resolve_dup_measure_dates(demo2)
        assert got2.nunique() == 2, got2.tolist()
        # same mid → separated by end
        demo3 = pd.DataFrame(
            {
                "participant_id": ["R", "R"],
                "date": pd.to_datetime(["2023-08-04", "2023-08-04"]),
                "start_time": pd.to_datetime(
                    ["2023-08-04 08:31:32", "2023-08-04 09:59:29"]
                ),
                "end_time": pd.to_datetime(
                    ["2023-08-04 10:00:00", "2023-08-05 01:00:00"]
                ),
            }
        )
        got3 = resolve_dup_measure_dates(demo3)
        assert list(got3) == [
            pd.Timestamp("2023-08-04"),
            pd.Timestamp("2023-08-05"),
        ], got3.tolist()
        assert OUT_DIR.name == "ovul_data" or _env_root
        print("self-check OK")
    else:
        run_raw_qc(workers=max(1, int(args.workers)))
