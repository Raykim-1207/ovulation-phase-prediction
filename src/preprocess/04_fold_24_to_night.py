# -*- coding: utf-8 -*-
"""
04 — Folds the 03 output (24 slots, fully interpolated) into one row per night.

Control for the "24-row sequence within a night vs one row per night" comparison. Folding **after 03** aggregates values that went through
the same temperature day interpolation (02) and HRV slot interpolation (03) → the only difference left between the two representations
is the representation itself, not missing-data handling.

Whether values were originally observed carries over unchanged as the sums of `hrv_mask`·`temp_mask`
(`n_hrv_win`·`n_temp_pts`), because 03 keeps the mask separate from the values. The interpolation fraction can be used as a covariate
or for post-hoc filtering.

Input: ovul_data/final/interp_{24|24mid}/{day}_{slot}[_z]/y*.xlsx
Output: ovul_data/final/interp_{24agg|24midagg}/{day}_{slot}[_z]/y*.xlsx
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

_r = __import__("os").environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_DIR = Path(_r) if _r else (Path(__file__).resolve().parents[2] / "ovul_data")
SEG_MIN = 10

NIGHT_KEYS = ["participant_id", "file_name_x"]
HRV_COLS = [
    "avgHR", "maxNN", "avgNN", "sdnn", "nn50", "pnn50", "rmssd",
    "tp", "ulf", "vlf", "lf", "hf", "lfhf",
]
TEMP_COL = "skin_temp"
AGGS = ("mean", "min", "max")
# Columns meaningful only along the slot axis — aggregated or dropped when folding.
# Slot-level derivatives added by 02·03 (raw temperature values, interpolation flags) must be included too so they do not leak through first().
SLOT_ONLY = {
    *HRV_COLS, TEMP_COL, f"{TEMP_COL}_raw", "window", "measure_time",
    "hrv_mask", "temp_mask", "hrv_dt", "temp_dt",
    "temp_slot_imputed", "temp_day_imputed",
}

STYLE_AGG = {"24": "24agg", "24mid": "24midagg"}


def fold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """24-slot df → one row per night. Column names match the ava family (`{base}_{agg}`)."""
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in (*NIGHT_KEYS, TEMP_COL, "hrv_mask", "temp_mask") if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    spec = {f"{b}_{a}": (b, a) for b in HRV_COLS for a in AGGS if b in df.columns}
    spec |= {
        "temp_mean": (TEMP_COL, "mean"),
        "temp_min": (TEMP_COL, "min"),
        "temp_max": (TEMP_COL, "max"),
        # Mask sum = number of actually observed slots **before interpolation** (different unit from ava's raw point count)
        "n_hrv_win": ("hrv_mask", "sum"),
        "n_temp_pts": ("temp_mask", "sum"),
        "n_slots": ("hrv_mask", "size"),
    }
    # Slot flag sum = number of slots filled by interpolation in that night (not dropped)
    for flag in ("temp_slot_imputed", "temp_day_imputed"):
        if flag in df.columns:
            spec[flag] = (flag, "sum")
    if "measure_time" in df.columns:
        spec |= {"used_from": ("measure_time", "min"), "used_to": ("measure_time", "max")}

    g = df.groupby(NIGHT_KEYS, sort=False)
    # Metadata/labels constant within a night (measure date, cycle, day of cycle, y*, ovulation flag, etc.) carry over unchanged
    meta_cols = [c for c in df.columns if c not in SLOT_ONLY and c not in NIGHT_KEYS]
    out = g[meta_cols].first().join(g.agg(**spec)).reset_index()

    out["temp_range"] = out["temp_max"] - out["temp_min"]
    if "used_to" in out.columns:
        out["used_to"] = out["used_to"] + timedelta(minutes=SEG_MIN)
        out["used_min"] = (out["used_to"] - out["used_from"]).dt.total_seconds() / 60.0

    head = NIGHT_KEYS + [c for c in ("measure_date", "cycle", "day_of_cycle") if c in out.columns]
    hrv_agg = [f"{b}_{a}" for b in HRV_COLS for a in AGGS if f"{b}_{a}" in out.columns]
    temp_agg = ["temp_mean", "temp_min", "temp_max", "temp_range"]
    quality = [
        c for c in (
            "n_hrv_win", "n_temp_pts", "n_slots", "used_min",
            "temp_slot_imputed", "temp_day_imputed",
        )
        if c in out.columns
    ]
    rest = [c for c in out.columns if c not in {*head, *hrv_agg, *temp_agg, *quality}]
    return out[head + hrv_agg + temp_agg + quality + rest]


def main(style: str = "all", combo: str | None = None) -> None:
    styles = list(STYLE_AGG) if style == "all" else [style]
    n_saved = 0
    for s in styles:
        in_root = OUT_DIR / "final" / f"interp_{s}"
        out_root = OUT_DIR / "final" / f"interp_{STYLE_AGG[s]}"
        if not in_root.is_dir():
            print(f"  [skip {s}] no input: {in_root}  (run 03 first)")
            continue
        combos = sorted(d for d in in_root.iterdir() if d.is_dir() and (combo is None or d.name == combo))
        if not combos:
            print(f"  [skip {s}] no combo folder: {in_root}/{combo or '*'}")
            continue
        print(f"\n===== {s} → {STYLE_AGG[s]}  ({len(combos)} combo) =====")
        for cdir in combos:
            files = sorted(cdir.glob("y*.xlsx"))
            dest_dir = out_root / cdir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in tqdm(files, desc=f"fold[{s}/{cdir.name}]", unit="file", leave=False):
                out = fold_frame(pd.read_excel(f, engine="openpyxl"))
                out.to_excel(dest_dir / f.name, index=False, engine="openpyxl")
                n_saved += 1
            print(f"  {cdir.name}: {len(files)} files → {dest_dir}")
    print(f"\nDone: {n_saved} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="03 output (24 slots) → per-night aggregation")
    ap.add_argument("--style", default="all", choices=[*STYLE_AGG, "all"])
    ap.add_argument("--combo", default=None, help="Only a specific combination (e.g. linear_locf). Default = all")
    args = ap.parse_args()
    main(style=args.style, combo=args.combo)
