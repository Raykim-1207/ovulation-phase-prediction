# -*- coding: utf-8 -*-
"""
03 — 24-slot HRV missing-value imputation (02 interp_day → interp_24).

Value-filling baselines from Che et al. 2018 (GRU-D) plus linear interpolation:

  mean    : GRU-mean   — fill with the per-variable dataset observed mean (eq. 1)
  forward : GRU-forward — carry the last observation forward (eq. 2). Leading gaps fall back to mean
  locf    : forward + leading back-fill (practical variant)
  linear  : linear interpolation over slot index (nearest at both ends)

Imputation targets
------------------
  · slot axis (window 1..24 within a night): **HRV only**
  · skin_temp: keep the 02 result (slot axis → day axis for fully missing nights); mask=0 slots are not erased

Input:  ovul_data/pre/interp_day_{24|24mid}_{day}/*.xlsx
Output: ovul_data/final/interp_{24|24mid}/{day}_{slot}/y*.xlsx
Next:   04_fold_24_to_night.py (fold to one row per night)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
_r = __import__("os").environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_DIR = Path(_r) if _r else (Path(__file__).resolve().parents[2] / "ovul_data")
PRE_DIR = OUT_DIR / "pre"
OUT_ROOT = OUT_DIR / "final" / "interp_24"

NIGHT_KEYS = ("participant_id", "file_name_x")
WINDOW_COL = "window"

HRV_COLS = [
    "avgHR", "maxNN", "avgNN", "sdnn", "nn50", "pnn50", "rmssd",
    "tp", "ulf", "vlf", "lf", "hf", "lfhf",
]
TEMP_COL = "skin_temp"

# Slot imputation: HRV only (temperature keeps the 02 result)
SLOT_FILL_COLS = list(HRV_COLS)
FILL_COLS = SLOT_FILL_COLS

DAY_METHODS = ("linear", "locf")
METHODS = ("mean", "forward", "locf", "linear")
STYLE_TAGS = ("24", "24mid")


def in_dir_for(day: str, style: str = "24") -> Path:
    return PRE_DIR / f"interp_day_{style}_{day}"


def out_root_for(style: str = "24") -> Path:
    return OUT_DIR / "final" / f"interp_{style}"


def combo_tag(day: str, slot: str, zscore: bool = False) -> str:
    tag = f"{day}_{slot}"
    return f"{tag}_z" if zscore else tag


# ---------------------------------------------------------------------------
# 1D imputation
# ---------------------------------------------------------------------------
def _as_float(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float)


def fill_mean(x: np.ndarray, mean_val: float = 0.0) -> np.ndarray:
    """GRU-mean: missing → observed mean. NaN if no mean is available or all values are missing."""
    y = _as_float(x).copy()
    mv = float(mean_val) if np.isfinite(mean_val) else float("nan")
    if not np.isfinite(mv):
        return y
    y[~np.isfinite(y)] = mv
    return y


def fill_forward(x: np.ndarray, mean_val: float = 0.0) -> np.ndarray:
    """
    GRU-forward (eq. 2): propagate only the last observation.
    Leading gaps fall back to mean; without a mean, leading NaNs are kept.
    """
    y = _as_float(x).copy()
    mv = float(mean_val) if np.isfinite(mean_val) else float("nan")
    last = np.nan
    for i in range(len(y)):
        if np.isfinite(y[i]):
            last = y[i]
        elif np.isfinite(last):
            y[i] = last
        elif np.isfinite(mv):
            y[i] = mv
    return y


def fill_locf(x: np.ndarray, mean_val: float = 0.0) -> np.ndarray:
    """Forward-fill, then leading back-fill. If all missing, use mean (NaN if unavailable)."""
    y = _as_float(x).copy()
    mv = float(mean_val) if np.isfinite(mean_val) else float("nan")
    if not np.any(np.isfinite(y)):
        if np.isfinite(mv):
            y[:] = mv
        return y
    last = np.nan
    for i in range(len(y)):
        if np.isfinite(y[i]):
            last = y[i]
        elif np.isfinite(last):
            y[i] = last
    nxt = np.nan
    for i in range(len(y) - 1, -1, -1):
        if np.isfinite(y[i]):
            nxt = y[i]
        elif np.isfinite(nxt):
            y[i] = nxt
    if np.isfinite(mv):
        y[~np.isfinite(y)] = mv
    return y


def fill_linear(x: np.ndarray, mean_val: float = 0.0) -> np.ndarray:
    """Linear interpolation over index. With 0 observations use mean (NaN if unavailable); with 1, no constant fill."""
    y = _as_float(x).copy()
    mv = float(mean_val) if np.isfinite(mean_val) else float("nan")
    idx = np.where(np.isfinite(y))[0]
    if idx.size == 0:
        if np.isfinite(mv):
            y[:] = mv
        return y
    if idx.size == 1:
        # do not paint the whole night from a single observation
        return y
    all_i = np.arange(len(y), dtype=float)
    y[:] = np.interp(all_i, idx.astype(float), y[idx])
    return y


FILLERS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "mean": fill_mean,
    "forward": fill_forward,
    "locf": fill_locf,
    "linear": fill_linear,
}


def _mask_or_finite(mask: np.ndarray | None, values: np.ndarray) -> np.ndarray:
    if mask is not None:
        m = np.asarray(mask, dtype=float) > 0.5
        return m & np.isfinite(values)
    return np.isfinite(values)


def empirical_means(df: pd.DataFrame) -> dict[str, float]:
    """Per-variable observed mean (x̃ in GRU-mean). For HRV, only cells with hrv_mask=1."""
    means: dict[str, float] = {}
    hrv_ok = None
    if "hrv_mask" in df.columns:
        hrv_ok = df["hrv_mask"].astype(float) > 0.5

    for col in HRV_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if hrv_ok is not None:
            s = s[hrv_ok]
        means[col] = float(s.mean()) if s.notna().any() else float("nan")
    return means


def interpolate_series(
    values: np.ndarray,
    mask: np.ndarray | None,
    method: str,
    mean_val: float = float("nan"),
) -> np.ndarray:
    """Keep observations (mask=1), fill only missing values with method. The caller keeps the mask."""
    v = _as_float(values)
    obs = _mask_or_finite(mask, v)
    work = v.copy()
    work[~obs] = np.nan
    filled = FILLERS[method](work, mean_val)
    out = filled.copy()
    out[obs] = v[obs]
    return out


def patient_zscore(df: pd.DataFrame, cols: list[str], pid_col: str = "participant_id") -> pd.DataFrame:
    """Per-participant z-score. Missing values stay NaN (not filled with 0, to avoid confusion with the mask)."""
    out = df.copy()
    for _, idx in out.groupby(pid_col, dropna=False).groups.items():
        sub = out.loc[list(idx), cols]
        mu = sub.mean(axis=0, skipna=True)
        sd = sub.std(axis=0, skipna=True, ddof=0)
        sd = sd.where(sd > 1e-8, np.nan)
        z = (sub - mu) / sd
        out.loc[list(idx), cols] = z.to_numpy()
    return out


# ---------------------------------------------------------------------------
# Per-night (slot) application
# ---------------------------------------------------------------------------
def interpolate_night(
    night: pd.DataFrame,
    method: str,
    means: dict[str, float],
) -> pd.DataFrame:
    """Within-night window axis: impute HRV only. skin_temp/temp_mask keep the 02 result."""
    out = night.copy()
    if WINDOW_COL in out.columns:
        order = np.argsort(pd.to_numeric(out[WINDOW_COL], errors="coerce").to_numpy())
        ordered = out.iloc[order]
    else:
        ordered = out

    hrv_mask = (
        ordered["hrv_mask"].to_numpy(dtype=float)
        if "hrv_mask" in ordered.columns
        else None
    )

    for col in HRV_COLS:
        if col not in ordered.columns:
            continue
        filled = interpolate_series(
            ordered[col].to_numpy(dtype=float),
            hrv_mask,
            method,
            mean_val=means.get(col, float("nan")),
        )
        out.loc[ordered.index, col] = filled

    return out


def interpolate_dataframe(
    df: pd.DataFrame,
    method: str,
    zscore: bool = False,
    means: dict[str, float] | None = None,
) -> pd.DataFrame:
    if method not in FILLERS:
        raise ValueError(f"unknown method={method!r}, choose from {METHODS}")
    keys = [k for k in NIGHT_KEYS if k in df.columns]
    if len(keys) < 2:
        raise KeyError(f"missing night keys: need {NIGHT_KEYS}, got {list(df.columns)}")

    if means is None:
        means = empirical_means(df)

    parts: list[pd.DataFrame] = []
    for _, night in df.groupby(keys, sort=False):
        parts.append(interpolate_night(night, method, means))
    out = pd.concat(parts, axis=0).sort_index()

    z_cols = [c for c in SLOT_FILL_COLS if c in out.columns]
    if zscore and z_cols and "participant_id" in out.columns:
        out = patient_zscore(out, z_cols)
    return out


def process_file_all_methods(
    path: Path,
    day: str,
    methods: list[str],
    zscore: bool,
    out_root: Path,
) -> list[Path]:
    """Load the Excel file once → save several slot methods (saves I/O)."""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    means = empirical_means(df)
    saved: list[Path] = []
    for method in methods:
        tag = combo_tag(day, method, zscore=zscore)
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out = interpolate_dataframe(df, method=method, zscore=zscore, means=means)
        dest = out_dir / path.name
        out.to_excel(dest, index=False, engine="openpyxl")
        saved.append(dest)
    return saved


def _self_check() -> None:
    combos = [combo_tag(d, s) for d in DAY_METHODS for s in METHODS]
    assert len(combos) == 8 and len(set(combos)) == 8
    x = np.array([1.0, np.nan, 3.0])
    assert np.allclose(fill_linear(x), [1.0, 2.0, 3.0])
    assert np.allclose(fill_locf(x), [1.0, 1.0, 3.0])
    # all missing → no fabricated mean
    empty = np.array([np.nan, np.nan])
    assert np.all(np.isnan(fill_mean(empty, mean_val=float("nan"))))
    # linear with one observation → no constant fill
    one = np.array([np.nan, 2.0, np.nan])
    assert np.isnan(fill_linear(one)[0]) and fill_linear(one)[1] == 2.0
    print("03 self-check OK")


def main(argv: list[str] | None = None) -> None:
    _self_check()
    p = argparse.ArgumentParser(
        description="interp_day_{24|24mid} → slot HRV imputation (temperature keeps 02)"
    )
    p.add_argument(
        "--style",
        default="24",
        choices=[*STYLE_TAGS, "all"],
        help="24=resolved 02-06, 24mid=middle 4h; default=24",
    )
    p.add_argument(
        "--day-method",
        default="all",
        choices=[*DAY_METHODS, "all"],
        help="02 input (interp_day_{style}_{day}); default=all → 3",
    )
    p.add_argument(
        "--method",
        default="all",
        choices=[*METHODS, "all"],
        help="slot imputation (default: all = 4 methods)",
    )
    p.add_argument(
        "--zscore",
        action="store_true",
        help="per-participant z-score after imputation (HRV)",
    )
    p.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="single input folder (use together with a single --day-method)",
    )
    p.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args(argv)

    styles = list(STYLE_TAGS) if args.style == "all" else [args.style]
    days = list(DAY_METHODS) if args.day_method == "all" else [args.day_method]
    methods = list(METHODS) if args.method == "all" else [args.method]
    if args.in_dir is not None and (len(days) != 1 or len(styles) != 1):
        raise SystemExit("--in-dir can only be used with a single --style and --day-method.")

    saved: list[Path] = []
    for style in styles:
        out_root = args.out_root if args.out_root is not None else out_root_for(style)
        print("=" * 60)
        print(f"  [03] interp_day_{style} day×slot HRV imputation")
        print(f"  day:  {days}")
        print(f"  slot: {methods}  → {len(days) * len(methods)} folders  zscore={args.zscore}")
        print(f"  out:  {out_root}")
        print(f"  HRV only: {SLOT_FILL_COLS}  (temperature=keeps 02)")
        print("=" * 60)

        for day in days:
            in_dir = args.in_dir if args.in_dir is not None else in_dir_for(day, style)
            if not in_dir.is_dir():
                print(f"  [skip] no input: {in_dir}")
                continue
            files = sorted(in_dir.glob("y*.xlsx"))
            if not files:
                print(f"  [skip] no y*.xlsx: {in_dir}")
                continue

            print(f"\n  --- {style}/{day}  ← {in_dir} ({len(files)} files) ---")
            for f in tqdm(files, desc=f"interp[{style}/{day}]", unit="file"):
                saved.extend(
                    process_file_all_methods(
                        f,
                        day=day,
                        methods=methods,
                        zscore=args.zscore,
                        out_root=out_root,
                    )
                )

    print("\n" + "=" * 60)
    print(f"  done: {len(saved)} files")
    for q in sorted({s.parent for s in saved}):
        print(f"    → {q}/")


if __name__ == "__main__":
    main()
