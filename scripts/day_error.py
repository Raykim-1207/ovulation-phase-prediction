# -*- coding: utf-8 -*-
"""
day_error — day-level ovulation_date error analysis.

F1 is not clinically interpretable. This script derives a per-cycle ovulation date from
night-level predictions (pred_nights/*.csv) and reports clinically interpretable metrics.

  · ovulation_date absolute error (days)      MAE, median
  · proportion within ±1 / ±2 days
  · fertile-window (ovulation -5 to +1) capture rate
  · number of non-estimable cycles   cycles with no night predicted as peri-ovulatory

**Pre-specified rule** (only one is used, to avoid post-hoc selection):
Under the standard definition the peri-ovulatory interval is ovulation_date -4 to +1, so its midpoint is ovulation_date -1.5.
Hence estimated ovulation_date = median(date of nights predicted peri-ovulatory) + 1.5 days, rounded.
It is symmetrically robust to truncation by missing nights.
The alternative (midpoint of the longest contiguous run) is reported as a sensitivity analysis.

  python3 scripts/day_error.py <pred_nights_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
SRC = PAPER.parent
DATA = Path(os.environ.get("OVUL_DATA_DIR_ROOT", "").strip() or SRC / "ovul_data") / "finalfinal"

# Midpoint offset of the peri-ovulatory interval: (-4 + 1)/2 = -1.5
BAND_MID = -1.5
FERTILE = range(-5, 2)          # biological fertile window -5 to +1
ARM_NAME = {"cal": "Calendar only", "physio": "Physiology only",
            "combined": "Combined"}


def load_truth(family: str, method: str, label: str) -> pd.DataFrame:
    f = DATA / family / method / f"{label}.xlsx"
    df = pd.read_excel(f, engine="openpyxl")
    df = df[["participant_id", "cycle", "measure_date", "ovulation_date"]].copy()
    df.columns = ["pid", "cyc", "day", "ov_true"]
    df["day"] = pd.to_datetime(df["day"])
    df["ov_true"] = pd.to_datetime(df["ov_true"])
    return df.drop_duplicates(["pid", "cyc", "day"])


def infer_ov_cls(d: pd.DataFrame) -> int:
    """Identify the class whose y_true concentrates in ovulation_date -4..+1 as peri-ovulatory."""
    best, best_score = None, -1.0
    for c in sorted(d.y_true.unique()):
        off = d.loc[d.y_true == c, "offset"]
        score = float(((off >= -4) & (off <= 1)).mean()) * len(off) ** 0.5
        if score > best_score:
            best, best_score = int(c), score
    return best


def estimate(sub: pd.DataFrame, ov_cls: int, rule: str) -> float:
    """Return one cycle's estimated ovulation day as a measure_date offset (nan = not estimable)."""
    hit = sub.loc[sub.y_pred == ov_cls, "offset"].to_numpy()
    if hit.size == 0:
        return float("nan")
    if rule == "median":
        return float(np.median(hit)) - BAND_MID
    # longest contiguous run (gaps up to 1 day allowed)
    h = np.sort(hit)
    runs, cur = [], [h[0]]
    for a, b in zip(h[:-1], h[1:]):
        (cur.append(b) if b - a <= 2 else (runs.append(cur), cur := [b]))
    runs.append(cur)
    best = max(runs, key=len)
    return (best[0] + best[-1]) / 2.0 - BAND_MID


def analyse(path: Path, rule: str = "median") -> list[dict]:
    out = []
    for f in sorted(path.glob("*.csv")):
        d = pd.read_csv(f)
        fam, mth, lab = d.family.iat[0], d.method.iat[0], d.label.iat[0]
        arm, model, split = d.arm.iat[0], d.model.iat[0], d.split.iat[0]
        d["day"] = pd.to_datetime(d["day"])
        d = d.merge(load_truth(fam, mth, lab), on=["pid", "cyc", "day"], how="left")
        if d.ov_true.isna().any():
            raise SystemExit(f"{f.name}: ov_true join failed for {d.ov_true.isna().sum()} rows")
        d["offset"] = (d.day - d.ov_true).dt.days
        ov_cls = infer_ov_cls(d)

        errs, n_ne, cover = [], 0, []
        for _, sub in d.groupby(["pid", "cyc"]):
            e = estimate(sub, ov_cls, rule)
            if np.isnan(e):
                n_ne += 1
            else:
                errs.append(abs(e))
            # Fertile-window capture: fraction of true fertile-window nights predicted peri-ovulatory
            fw = sub[sub.offset.isin(FERTILE)]
            if len(fw):
                cover.append(float((fw.y_pred == ov_cls).mean()))
        a = np.asarray(errs, dtype=float)
        out.append({
            "label": lab, "split": split, "arm": arm, "model": model,
            "n_cycles": d.groupby(["pid", "cyc"]).ngroups,
            "n_not_estimable": n_ne,
            "mae_days": a.mean() if a.size else float("nan"),
            "median_ae": np.median(a) if a.size else float("nan"),
            "within1": float((a <= 1).mean()) if a.size else float("nan"),
            "within2": float((a <= 2).mean()) if a.size else float("nan"),
            "fertile_capture": float(np.mean(cover)) if cover else float("nan"),
        })
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / "pred_nights"
    if not path.is_dir():
        raise SystemExit(f"no such dir: {path}")
    for rule in ("median", "run"):
        df = pd.DataFrame(analyse(path, rule))
        if df.empty:
            raise SystemExit("no prediction files")
        df["arm"] = pd.Categorical(df.arm, ["cal", "physio", "combined"], ordered=True)
        df = df.sort_values(["split", "label", "arm", "model"])
        print(f"\n{'='*88}\npre-specified rule: {rule}\n{'='*88}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if rule == "median":
            df.to_csv(PAPER / "fig" / "day_error.csv", index=False)
            print(f"\n→ {PAPER / 'fig' / 'day_error.csv'}")


if __name__ == "__main__":
    main()
