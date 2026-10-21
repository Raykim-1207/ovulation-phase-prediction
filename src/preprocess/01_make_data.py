# -*- coding: utf-8 -*-
"""
01_make_data — label and cycle alignment (signal-independent).

Attaches participant, cycle, day_of_cycle and multi-class labels to the 00 output (HRV + raw temperature).

Cycles without an ovulation date are **excluded** from both labels and wearable data.

Notes
--------------------------------
24 path: the slot `measure_time` is the extraction-window time, but
labels and cycle days follow the mapping's **resolved_date (=measure_date)**.
Rerun: this script → 02_temp_day_interp → 03 (24) → 04 (24 → one row per night; per-night tags finish at 02)

2 label versions × 4 class schemes = 8 columns:
  y5_mod / y4_mod / y3_mod / y2_mod   (modified version)
  y5_std / y4_std / y3_std / y2_std   (standard / JMIR mapping version)

5-class (1..5):
  1 menstrual  2 follicular  3 fertile  4 early luteal  5 late luteal
4-class (1..4): 1, 2, 3 unchanged, 4 = early + late luteal merged
3-class (1..3): 1 menstrual, 2 fertile, 3 non-fertile other (follicular + luteal)
2-class (1..2): 1 fertile, 2 non-fertile

Modified-version intervals (day_of_cycle +1 = menses start day):
  P1: menses +1..+5
  P2: menses +6 .. 4 days before ovulation
  P3: 3 days before ovulation .. day after ovulation
  P4: 2 days after ovulation .. 5 days before next menses
  P5: 4 days before next menses .. until next menses start

Standard-version intervals:
  P1: menses +1..+5
  P2: menses +6 .. 5 days before ovulation
  P3: 4 days before ovulation .. day after ovulation
  P4: 2 days after ovulation .. 7 days after ovulation
  P5: 8 days after ovulation .. until next menses start
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from _raw_common import (
    LABEL_DIR,
    PRE_DIR,
    build_cycles_by_patient,
    load_cycle_calendar,
    _norm_cell_id,
    _normalize_columns,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
_r = __import__("os").environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_DIR = Path(_r) if _r else (Path(__file__).resolve().parents[2] / "ovul_data")

# 00 output → 01 labelled output (per path) — intermediates go to pre/
# ava=90/30, ava30=+30/−30, 24=resolved 02–06, 24mid=recording midpoint 4h
WEARABLE_JOBS = (
    ("ava", PRE_DIR / "ovul_data_ava.xlsx", PRE_DIR / "ovul_data_labeled_ava.xlsx"),
    ("ava30", PRE_DIR / "ovul_data_ava30.xlsx", PRE_DIR / "ovul_data_labeled_ava30.xlsx"),
    ("ava24", PRE_DIR / "ovul_data_ava24.xlsx", PRE_DIR / "ovul_data_labeled_ava24.xlsx"),
    ("ava24mid", PRE_DIR / "ovul_data_ava24mid.xlsx", PRE_DIR / "ovul_data_labeled_ava24mid.xlsx"),
    ("24", PRE_DIR / "ovul_data_24.xlsx", PRE_DIR / "ovul_data_labeled_24.xlsx"),
    ("24mid", PRE_DIR / "ovul_data_24mid.xlsx", PRE_DIR / "ovul_data_labeled_24mid.xlsx"),
    ("huawei", PRE_DIR / "ovul_data_huawei.xlsx", PRE_DIR / "ovul_data_labeled_huawei.xlsx"),
    ("huawei24", PRE_DIR / "ovul_data_huawei24.xlsx", PRE_DIR / "ovul_data_labeled_huawei24.xlsx"),
    ("huawei24mid", PRE_DIR / "ovul_data_huawei24mid.xlsx", PRE_DIR / "ovul_data_labeled_huawei24mid.xlsx"),
    ("npj", PRE_DIR / "ovul_data_npj.xlsx", PRE_DIR / "ovul_data_labeled_npj.xlsx"),
    ("npj24", PRE_DIR / "ovul_data_npj24.xlsx", PRE_DIR / "ovul_data_labeled_npj24.xlsx"),
    ("npj24mid", PRE_DIR / "ovul_data_npj24mid.xlsx", PRE_DIR / "ovul_data_labeled_npj24mid.xlsx"),
    ("oura", PRE_DIR / "ovul_data_oura.xlsx", PRE_DIR / "ovul_data_labeled_oura.xlsx"),
    ("oura24", PRE_DIR / "ovul_data_oura24.xlsx", PRE_DIR / "ovul_data_labeled_oura24.xlsx"),
    ("oura24mid", PRE_DIR / "ovul_data_oura24mid.xlsx", PRE_DIR / "ovul_data_labeled_oura24mid.xlsx"),
)

# Output label columns (8) — combined file + one file per label
LABEL_COLS = (
    "y5_mod", "y4_mod", "y3_mod", "y2_mod",
    "y5_std", "y4_std", "y3_std", "y2_std",
)

# Observed nights per cycle ≥ this fraction of cycle length (msd→nmsd). For 28 days, ≥14 nights.
MIN_CYCLE_COV = 0.5


def save_per_label(df: pd.DataFrame, split_dir: Path, label_cols: tuple[str, ...] = LABEL_COLS) -> list[Path]:
    """
    One file per label column. Other y* columns are dropped; only the target label is kept.
    e.g. ovul_data/labeled_ava/y5_mod.xlsx
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    others = [c for c in label_cols if c in df.columns]
    for col in label_cols:
        if col not in df.columns:
            continue
        drop = [c for c in others if c != col]
        sub = df.drop(columns=drop, errors="ignore").copy()
        path = split_dir / f"{col}.xlsx"
        sub.to_excel(path, index=False, engine="openpyxl")
        saved.append(path)
    return saved


def _to_date(cell_value):
    if cell_value is None or (isinstance(cell_value, float) and np.isnan(cell_value)):
        return None
    if isinstance(cell_value, datetime):
        return cell_value.date()
    if isinstance(cell_value, pd.Timestamp):
        if pd.isna(cell_value):
            return None
        return cell_value.date()
    if isinstance(cell_value, date):
        return cell_value
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


def load_labels(label_dir: Path = LABEL_DIR) -> pd.DataFrame:
    """Closed cycles only (msd+nmsd+ov). Delegates to ``_raw_common.load_cycle_calendar``."""
    return load_cycle_calendar(label_dir)


def _valid_nmsd(nmsd) -> bool:
    return nmsd is not None and not (isinstance(nmsd, float) and np.isnan(nmsd))


def _in_cycle(measurement_date, msd, nmsd) -> bool:
    """Closed cycles only: msd <= d < nmsd. False if nmsd is missing."""
    if measurement_date is None or msd is None or not _valid_nmsd(nmsd):
        return False
    return msd <= measurement_date < nmsd


def _in_closed(d: date, lo: date, hi: Optional[date]) -> bool:
    """lo <= d <= hi. hi=None means no upper bound."""
    if lo is None or d < lo:
        return False
    if hi is None:
        return True
    return d <= hi


def phase5(d: date, msd: date, nmsd, ov: date, version: str) -> Optional[int]:
    """
    5-class phase (1..5). None if it falls in no interval.
    version: 'mod' | 'std'
    When intervals overlap in short cycles: fertile (P3) > menstrual (P1) > others.
    """
    if not _in_cycle(d, msd, nmsd):
        return None

    if version == "mod":
        p2_lo, p2_hi = msd + timedelta(days=5), ov - timedelta(days=4)
        p3_lo, p3_hi = ov - timedelta(days=3), ov + timedelta(days=1)
        p4_lo = ov + timedelta(days=2)
        p4_hi = (nmsd - timedelta(days=5)) if _valid_nmsd(nmsd) else None
        p5_lo = (nmsd - timedelta(days=4)) if _valid_nmsd(nmsd) else None
        p5_hi = (nmsd - timedelta(days=1)) if _valid_nmsd(nmsd) else None
    elif version == "std":
        p2_lo, p2_hi = msd + timedelta(days=5), ov - timedelta(days=5)
        p3_lo, p3_hi = ov - timedelta(days=4), ov + timedelta(days=1)
        p4_lo, p4_hi = ov + timedelta(days=2), ov + timedelta(days=7)
        p5_lo = ov + timedelta(days=8)
        p5_hi = (nmsd - timedelta(days=1)) if _valid_nmsd(nmsd) else None
    else:
        raise ValueError(version)

    # Clip to the cycle upper bound (before next menses)
    if _valid_nmsd(nmsd):
        last = nmsd - timedelta(days=1)
        p2_hi = min(p2_hi, last)
        p3_hi = min(p3_hi, last)
        if p4_hi is not None:
            p4_hi = min(p4_hi, last)
        if p5_hi is not None:
            p5_hi = min(p5_hi, last)

    # P3 fertile takes priority
    if _in_closed(d, p3_lo, p3_hi):
        return 3
    # P1: menses +1..+5 ≡ msd .. msd+4
    if msd <= d <= msd + timedelta(days=4):
        return 1
    if p2_lo <= p2_hi and _in_closed(d, p2_lo, p2_hi):
        return 2
    if version == "mod":
        if p5_lo is not None and _in_closed(d, p5_lo, p5_hi):
            return 5
        if _in_closed(d, p4_lo, p4_hi):
            return 4
    else:
        if _in_closed(d, p4_lo, p4_hi):
            return 4
        if _in_closed(d, p5_lo, p5_hi):
            return 5
    return None


def collapse_labels(y5: Optional[int]) -> dict[str, Optional[int]]:
    """5-class → 4/3/2 class."""
    if y5 is None:
        return {"y4": None, "y3": None, "y2": None}
    # 4: luteal = 4+5 → 4
    y4 = 4 if y5 >= 4 else y5
    # 3: 1 menstrual, 2 fertile(=3), 3 non-fertile(2,4,5)
    if y5 == 1:
        y3 = 1
    elif y5 == 3:
        y3 = 2
    else:
        y3 = 3
    # 2: 1 fertile, 2 non-fertile
    y2 = 1 if y5 == 3 else 2
    return {"y4": y4, "y3": y3, "y2": y2}


def annotate_row(pid: str, measurement_date, cycles_by_patient: dict) -> dict:
    """Cycle metadata + 8 labels. Labels are None if matching fails or the interval is empty."""
    empty = {
        "cycle": np.nan,
        "day_of_cycle": np.nan,
        "menses_start": pd.NaT,
        "next_menses_start": pd.NaT,
        "ovulation_date": pd.NaT,
        **{c: np.nan for c in LABEL_COLS},
        "_matched": False,
        "_labeled": False,
    }
    if measurement_date is None or pid not in cycles_by_patient:
        return empty
    for cycle in cycles_by_patient[pid]:
        msd, nmsd, ov = cycle["msd"], cycle["nmsd"], cycle["ov_date"]
        if ov is None or msd is None:
            continue
        if not _in_cycle(measurement_date, msd, nmsd):
            continue

        y5_mod = phase5(measurement_date, msd, nmsd, ov, "mod")
        y5_std = phase5(measurement_date, msd, nmsd, ov, "std")
        cm = collapse_labels(y5_mod)
        cs = collapse_labels(y5_std)
        labeled = y5_mod is not None and y5_std is not None
        return {
            "cycle": cycle["cycle"],
            "day_of_cycle": (measurement_date - msd).days + 1,
            "menses_start": msd,
            "next_menses_start": nmsd if _valid_nmsd(nmsd) else pd.NaT,
            "ovulation_date": ov,
            "y5_mod": y5_mod if y5_mod is not None else np.nan,
            "y4_mod": cm["y4"] if cm["y4"] is not None else np.nan,
            "y3_mod": cm["y3"] if cm["y3"] is not None else np.nan,
            "y2_mod": cm["y2"] if cm["y2"] is not None else np.nan,
            "y5_std": y5_std if y5_std is not None else np.nan,
            "y4_std": cs["y4"] if cs["y4"] is not None else np.nan,
            "y3_std": cs["y3"] if cs["y3"] is not None else np.nan,
            "y2_std": cs["y2"] if cs["y2"] is not None else np.nan,
            "_matched": True,
            "_labeled": labeled,
        }
    return empty


def resolve_measure_date(df: pd.DataFrame) -> pd.Series:
    """
    measure_date for labels/cycle day = the ``measure_date`` set by 00 (= compare ``resolved_date``).

    Even if the slot ``measure_time`` crosses midnight, the whole night is labelled with the same resolved date.
    The ``measure_time−1 day`` fallback was removed (bug that split the label day for mid4h 23:00–02:50).
    """
    if "measure_date" not in df.columns or not df["measure_date"].notna().any():
        raise KeyError(
            "The wearable data needs 'measure_date' (compare resolved_date). "
            "Rerun 00_* and then run 01."
        )
    return pd.to_datetime(df["measure_date"], errors="coerce").dt.date


def filter_cycles_min_coverage(df: pd.DataFrame, min_cov: float = MIN_CYCLE_COV) -> tuple[pd.DataFrame, int]:
    """
    Observed nights per participant×cycle ≥ ceil(cycle length × min_cov).
    Cycle length = (next_menses_start − menses_start).days  (days in [msd, nmsd)).
    Night key: file_name_x if present, otherwise measure_date.
    """
    need = {"participant_id", "cycle", "menses_start", "next_menses_start"}
    if not need.issubset(df.columns) or df.empty:
        return df, 0
    night_col = "file_name_x" if "file_name_x" in df.columns else "measure_date"
    if night_col not in df.columns:
        return df, 0

    msd = pd.to_datetime(df["menses_start"], errors="coerce")
    nmsd = pd.to_datetime(df["next_menses_start"], errors="coerce")
    cycle_len = (nmsd - msd).dt.days
    work = df.assign(_cycle_len=cycle_len, _night=df[night_col].astype(str))
    work = work.loc[work["_cycle_len"].notna() & (work["_cycle_len"] > 0)].copy()
    if work.empty:
        return df.iloc[0:0].copy(), len(df)

    g = work.groupby(["participant_id", "cycle"], dropna=False)
    n_nights = g["_night"].nunique()
    lens = g["_cycle_len"].first()
    need_n = np.ceil(lens.to_numpy(dtype=float) * float(min_cov)).astype(int)
    ok_keys = n_nights.index[n_nights.to_numpy() >= need_n]
    key_index = pd.MultiIndex.from_frame(df[["participant_id", "cycle"]])
    out = df.loc[key_index.isin(ok_keys)].copy()
    n_drop_rows = len(df) - len(out)
    n_drop_cycles = int(len(n_nights) - len(ok_keys))
    print(
        f"  [cov≥{min_cov:.0%}] cycle {len(n_nights)} total, dropped {n_drop_cycles} "
        f"(observed nights < ceil(cycle days×{min_cov:.0%}); rows −{n_drop_rows})"
    )
    return out, n_drop_cycles


def process_wearable(wearable_path: Path, out_path: Path, cycles: dict, tag: str) -> Path:
    print("\n" + "-" * 60)
    print(f"  [{tag}] wearable: {wearable_path}")
    if not wearable_path.is_file():
        raise FileNotFoundError(
            f"Wearable master not found: {wearable_path}\n"
            "  → run 00_master_<family>.py first."
        )
    df = pd.read_excel(wearable_path, engine="openpyxl")
    df = _normalize_columns(df)
    if "participant_id" not in df.columns:
        raise KeyError(f"[{tag}] wearable data has no participant_id: {list(df.columns)}")
    df["participant_id"] = df["participant_id"].map(_norm_cell_id)
    df = df[df["participant_id"].notna()].copy()
    n0 = len(df)
    measure_dates = resolve_measure_date(df)
    # Diagnostic: the date used for labels (= resolved date)
    df["label_date"] = pd.to_datetime(measure_dates, errors="coerce")

    ann = [
        annotate_row(str(pid), md, cycles)
        for pid, md in zip(df["participant_id"], measure_dates)
    ]
    ann_df = pd.DataFrame(ann)
    for c in ann_df.columns:
        if c.startswith("_"):
            continue
        df[c] = ann_df[c].values
    df["_matched"] = ann_df["_matched"].values
    df["_labeled"] = ann_df["_labeled"].values

    df = df.drop(columns=["ovulation_flag"], errors="ignore")

    n_match = int(df["_matched"].sum())
    n_lab = int(df["_labeled"].sum())
    df = df.loc[df["_labeled"]].drop(columns=["_matched", "_labeled"]).copy()

    n_before_cov = len(df)
    df, _ = filter_cycles_min_coverage(df, MIN_CYCLE_COV)

    print(
        f"  [{tag}] input {n0} rows → cycle-matched {n_match} → valid labels {n_lab} "
        f"→ cov≥{MIN_CYCLE_COV:.0%} {len(df)} rows "
        f"(participants {df['participant_id'].nunique() if len(df) else 0}; before cov {n_before_cov} rows)"
    )
    if df.empty:
        raise RuntimeError(f"[{tag}] no rows passed the label and coverage filters.")

    for c in LABEL_COLS:
        vc = df[c].value_counts(dropna=False).sort_index().to_dict()
        print(f"  [{tag}] {c}: {vc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"  [{tag}] saved (combined): {out_path}")

    split_dir = PRE_DIR / f"labeled_{tag}"
    per = save_per_label(df, split_dir)
    print(f"  [{tag}] saved ({len(per)} per-label files): {split_dir}/")
    for p in per:
        print(f"           → {p.name}")
    return out_path


def main():
    print("=" * 60)
    print("  [01] loading labels (cycles with ovulation_date only) → ava / 24 / …")
    print("  label measure_date = 00 measure_date (= compare resolved_date); independent of slot time")
    print("=" * 60)
    print(f"  LABEL_DIR: {LABEL_DIR}")
    rel = load_labels(LABEL_DIR)
    cycles = build_cycles_by_patient(rel)

    saved: list[Path] = []
    missing: list[str] = []
    for tag, wearable_path, out_path in WEARABLE_JOBS:
        if not wearable_path.is_file():
            print(f"\n  [skip {tag}] no input: {wearable_path}")
            missing.append(tag)
            continue
        saved.append(process_wearable(wearable_path, out_path, cycles, tag))

    if not saved:
        raise FileNotFoundError(
            "No 00 output to process. At least one of the following is required:\n"
            + "\n".join(f"  - {p}" for _, p, _ in WEARABLE_JOBS)
        )
    print("\n" + "=" * 60)
    print(f"  done: {len(saved)} saved" + (f" (skip: {missing})" if missing else ""))
    for p in saved:
        print(f"    → {p}")
    print("Next: 02_temp_day_interp.py → 03 (24)")


if __name__ == "__main__":
    main()
