# -*- coding: utf-8 -*-
"""Strict calendar rules — no ML, no cal_* eng, no day-prior learning.

Methods (ovulation estimate from menses start only):
  cd14       : menses + 13d  (= cycle day 14)
  cd15       : menses + 14d  (= cycle day 15)
  hist_cl14  : menses + (past mean cycle length − 14); fallback cd15

Phase windows match the labeling protocol (mod / std). Doctor ov-day is never used.
Next-menses is only used for late luteal phase boundaries (same as label geometry).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

STRICT_CAL_METHODS = ("cd14", "cd15", "hist_cl14")


def _valid_nmsd(nmsd) -> bool:
    return nmsd is not None and not (isinstance(nmsd, float) and np.isnan(nmsd)) and pd.notna(nmsd)


def _in_cycle(d, msd, nmsd) -> bool:
    if d is None or msd is None or pd.isna(d) or pd.isna(msd):
        return False
    if _valid_nmsd(nmsd):
        return msd <= d < nmsd
    return d >= msd


def _in_closed(d: date, lo: date, hi: Optional[date]) -> bool:
    if lo is None or d < lo:
        return False
    if hi is None:
        return True
    return d <= hi


def to_date(v) -> Optional[date]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    if isinstance(v, date) and not isinstance(v, pd.Timestamp):
        return v
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def label_version(stem: str) -> str:
    return "std" if "std" in stem.lower() else "mod"


def phase5(d: date, msd: date, nmsd, ov: date, version: str) -> Optional[int]:
    if not _in_cycle(d, msd, nmsd) or ov is None or pd.isna(ov):
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

    if _valid_nmsd(nmsd):
        last = nmsd - timedelta(days=1)
        p2_hi = min(p2_hi, last)
        p3_hi = min(p3_hi, last)
        if p4_hi is not None:
            p4_hi = min(p4_hi, last)
        if p5_hi is not None:
            p5_hi = min(p5_hi, last)

    if _in_closed(d, p3_lo, p3_hi):
        return 3
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


def collapse_from_y5(y5: Optional[int], stem: str) -> Optional[int]:
    if y5 is None:
        return None
    s = stem.lower()
    if s.startswith("y5"):
        return y5
    if s.startswith("y4"):
        return 4 if y5 >= 4 else y5
    if s.startswith("y3"):
        if y5 == 1:
            return 1
        if y5 == 3:
            return 2
        return 3
    if s.startswith("y2"):
        return 1 if y5 == 3 else 2
    return y5


def _cycle_length(msd: date, nmsd) -> Optional[int]:
    if not _valid_nmsd(nmsd):
        return None
    return int((nmsd - msd).days)


def build_hist_cl_map(
    pid: np.ndarray,
    cyc: np.ndarray,
    msd: np.ndarray,
    nmsd: np.ndarray,
) -> dict[tuple[str, int], float]:
    """(patient, cycle) → mean of PAST cycle lengths only (current cycle unused)."""
    uniq: dict[tuple[str, int], tuple[date, Optional[int]]] = {}
    for i in range(len(pid)):
        p, c = str(pid[i]), pd.to_numeric(cyc[i], errors="coerce")
        m = to_date(msd[i])
        n = to_date(nmsd[i])
        if m is None or pd.isna(c):
            continue
        uniq[(p, int(c))] = (m, _cycle_length(m, n))

    by_pid: dict[str, list[tuple[int, date, Optional[int]]]] = {}
    for (p, c), (m, cl) in uniq.items():
        by_pid.setdefault(p, []).append((c, m, cl))
    for p in by_pid:
        by_pid[p].sort(key=lambda x: (x[1], x[0]))

    out: dict[tuple[str, int], float] = {}
    for p, lst in by_pid.items():
        prev: list[float] = []
        for c, m, cl in lst:
            if prev:
                out[(p, c)] = float(np.mean(prev))
            if cl is not None and cl > 0:
                prev.append(float(cl))
    return out


def estimate_ovulation(
    method: str,
    msd: date,
    pid: str,
    cyc: int,
    hist_cl: dict[tuple[str, int], float],
) -> Optional[date]:
    if msd is None:
        return None
    if method == "cd14":
        return msd + timedelta(days=13)
    if method == "cd15":
        return msd + timedelta(days=14)
    if method == "hist_cl14":
        mean_cl = hist_cl.get((pid, cyc))
        if mean_cl is None or mean_cl <= 14:
            return msd + timedelta(days=14)
        return msd + timedelta(days=int(round(mean_cl - 14)))
    raise ValueError(method)


def predict_strict_cal(
    method: str,
    label: str,
    pid: np.ndarray,
    cyc: np.ndarray,
    msd: np.ndarray,
    nmsd: np.ndarray,
    day: np.ndarray,
    hist_cl: Optional[dict[tuple[str, int], float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (keep_mask, pred_0based) for rows with a valid rule prediction.
    pred is 0-based class id matching train labels (y_raw - 1).
    """
    if method not in STRICT_CAL_METHODS:
        raise ValueError(method)
    if hist_cl is None:
        hist_cl = build_hist_cl_map(pid, cyc, msd, nmsd)
    version = label_version(label)
    keep = np.zeros(len(pid), dtype=bool)
    pred = np.full(len(pid), -1, dtype=np.int64)
    for i in range(len(pid)):
        m = to_date(msd[i])
        n = to_date(nmsd[i])
        d = to_date(day[i])
        c = pd.to_numeric(cyc[i], errors="coerce")
        if m is None or d is None or pd.isna(c):
            continue
        ov = estimate_ovulation(method, m, str(pid[i]), int(c), hist_cl)
        y5 = phase5(d, m, n, ov, version) if ov else None
        yp = collapse_from_y5(y5, label)
        if yp is None:
            continue
        keep[i] = True
        pred[i] = int(yp) - 1
    return keep, pred


def calendar_meta_from_df(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract aligned calendar meta (not used as ML features)."""
    cols = {str(c).strip(): c for c in df.columns}
    n = len(df)
    pid = df[cols["participant_id"]].astype(str).to_numpy() if "participant_id" in cols else np.array(["?"] * n)
    cyc = (
        pd.to_numeric(df[cols["cycle"]], errors="coerce").to_numpy()
        if "cycle" in cols else np.full(n, np.nan)
    )
    msd = df[cols["menses_start"]].to_numpy() if "menses_start" in cols else np.array([None] * n)
    nmsd = df[cols["next_menses_start"]].to_numpy() if "next_menses_start" in cols else np.array([None] * n)
    if "measure_date" in cols:
        day = df[cols["measure_date"]].to_numpy()
    elif "day_of_cycle" in cols:
        # reconstruct measurement date = menses + cycle_day - 1
        cd = pd.to_numeric(df[cols["day_of_cycle"]], errors="coerce").to_numpy()
        day = np.empty(n, dtype=object)
        for i in range(n):
            m = to_date(msd[i])
            if m is None or not np.isfinite(cd[i]):
                day[i] = None
            else:
                day[i] = m + timedelta(days=int(cd[i]) - 1)
    else:
        day = np.array([None] * n)
    return {"pid": pid, "cyc": cyc, "msd": msd, "nmsd": nmsd, "day": day}


if __name__ == "__main__":
    # CD14: menses 2023-01-01 → ov=2023-01-14; day=2023-01-14 → fertile (phase3)
    msd0 = date(2023, 1, 1)
    nmsd0 = date(2023, 1, 29)
    d_fert = date(2023, 1, 14)
    ov = estimate_ovulation("cd14", msd0, "A", 1, {})
    assert ov == date(2023, 1, 14)
    assert phase5(d_fert, msd0, nmsd0, ov, "mod") == 3
    assert collapse_from_y5(3, "y2_mod") == 1
    assert collapse_from_y5(3, "y5_mod") == 3
    # hist uses past only
    pid = np.array(["A", "A", "A"])
    cyc = np.array([1, 2, 2])
    msd = np.array([date(2023, 1, 1), date(2023, 2, 1), date(2023, 2, 1)])
    nmsd = np.array([date(2023, 1, 29), date(2023, 3, 5), date(2023, 3, 5)])
    hist = build_hist_cl_map(pid, cyc, msd, nmsd)
    assert ( "A", 1) not in hist
    assert abs(hist[("A", 2)] - 28.0) < 1e-9
    print("strict_cal_rule self-check OK")
