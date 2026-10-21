# -*- coding: utf-8 -*-
"""
02 — within-cycle daily (nightly) temp interpolation (+ huawei/npj/oura enrich).

Fills temperature gaps in the 01 labelled output only **within participant and cycle**.
  · ava / ava30 / ava24 / ava24mid / huawei* / npj* / oura*: interpolate the primary temp only (≥2 observations).
  · 24 / 24mid: temperature only — within-night 10-min slot axis first, day axis if the whole night is missing. HRV is in 03.
  · huawei/npj/oura: enrich_*_cycle after interpolation (cycle waveform features).

Next: 03 (24/24mid) → 04 (fold 24 slots into one row per night).
For per-night tags (ava*/huawei*/npj*/oura*) the 02 output is final → final/interp_{tag}/{method}/.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
_r = __import__("os").environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_DIR = Path(_r) if _r else (Path(__file__).resolve().parents[2] / "ovul_data")
PRE_DIR = OUT_DIR / "pre"
FINAL_DIR = OUT_DIR / "final"
# 24/24mid are written to final by 03/04. Other per-night tags are written to final by 02.
SLOT_TAGS = frozenset({"24", "24mid"})

TAGS = (
    "ava", "ava30", "ava24", "ava24mid",
    "24", "24mid",
    "huawei", "huawei24", "huawei24mid",
    "npj", "npj24", "npj24mid",
    "oura", "oura24", "oura24mid",
)
ENRICH_TAGS = (
    "huawei", "huawei24", "huawei24mid",
    "npj", "npj24", "npj24mid",
    "oura", "oura24", "oura24mid",
)
LABELED_PATH = {tag: PRE_DIR / f"ovul_data_labeled_{tag}.xlsx" for tag in TAGS}


def out_path_for(tag: str, method: str) -> Path:
    return PRE_DIR / f"ovul_data_interp_day_{tag}_{method}.xlsx"


def split_dir_for(tag: str, method: str) -> Path:
    return PRE_DIR / f"interp_day_{tag}_{method}"


def final_split_dir_for(tag: str, method: str) -> Path | None:
    """Final dir for per-night tags: final/interp_{tag}/{method}/. None for 24* (03/04)."""
    if tag in SLOT_TAGS:
        return None
    return FINAL_DIR / f"interp_{tag}" / method

# Interpolate only the primary temperature. min/max/range/nadir are excluded because interpolating them
# independently gives min>max, inconsistent range and clock-wrap distortion (missing nights stay NaN).
TEMP_COL_BY_TAG = {
    "ava": "temp_mean",
    "ava30": "temp_mean",
    "ava24": "temp_mean",
    "ava24mid": "temp_mean",
    "huawei": "bbt",
    "huawei24": "bbt",
    "huawei24mid": "bbt",
    "npj": "temp_mean",
    "npj24": "temp_mean",
    "npj24mid": "temp_mean",
    "oura": "temp_mean",
    "oura24": "temp_mean",
    "oura24mid": "temp_mean",
}
EXTRA_TEMP_COLS_BY_TAG = {
    "ava": (),
    "ava30": (),
    "ava24": (),
    "ava24mid": (),
    "huawei": (),
    "huawei24": (),
    "huawei24mid": (),
    "npj": (),
    "npj24": (),
    "npj24mid": (),
    "oura": (),
    "oura24": (),
    "oura24mid": (),
}

NIGHT_KEY_COLS = ("participant_id", "file_name_x")
CYCLE_KEYS = ("participant_id", "cycle")
TEMP_24 = "skin_temp"
WINDOW_COL = "window"
METHODS = ("linear", "locf")

LABEL_COLS = (
    "y5_mod", "y4_mod", "y3_mod", "y2_mod",
    "y5_std", "y4_std", "y3_std", "y2_std",
)


def save_per_label(df: pd.DataFrame, split_dir: Path, label_cols: tuple[str, ...] = LABEL_COLS) -> list[Path]:
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


# ---------------------------------------------------------------------------
# 1D fillers (cycle-day axis)
# ---------------------------------------------------------------------------
def fill_linear_by_day(days: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation over day_of_cycle values. Not filled with fewer than 2 observations (avoids fabricating constants)."""
    y = np.asarray(values, dtype=float).copy()
    d = np.asarray(days, dtype=float)
    ok = np.isfinite(y) & np.isfinite(d)
    if ok.sum() < 2:
        return y
    miss = (~np.isfinite(y)) & np.isfinite(d)
    if miss.any():
        y[miss] = np.interp(d[miss], d[ok], y[ok])
    return y


def fill_locf_by_day(days: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Forward-fill then leading back-fill. Not filled with fewer than 2 observations."""
    y = np.asarray(values, dtype=float).copy()
    if np.isfinite(y).sum() < 2:
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
    return y


FILLERS = {
    "linear": fill_linear_by_day,
    "locf": fill_locf_by_day,
}


def _interp_cycle_series(
    days: np.ndarray,
    values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """returns (filled, was_imputed_mask)."""
    raw = np.asarray(values, dtype=float).copy()
    filled = FILLERS[method](days, raw)
    imputed = (~np.isfinite(raw)) & np.isfinite(filled)
    # never overwrite observations
    out = raw.copy()
    out[imputed] = filled[imputed]
    return out, imputed


# ---------------------------------------------------------------------------
# One row per night
# ---------------------------------------------------------------------------
def interpolate_nightly(
    df: pd.DataFrame,
    temp_cols: list[str],
    method: str,
) -> pd.DataFrame:
    out = df.copy()
    for col in temp_cols:
        if col not in out.columns:
            continue
        raw_col = f"{col}_raw"
        if raw_col not in out.columns:
            out[raw_col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    flag_col = "temp_day_imputed"
    out[flag_col] = 0

    if "day_of_cycle" not in out.columns or "cycle" not in out.columns:
        return out

    primary = temp_cols[0] if temp_cols else None

    for (_, _), g in out.groupby(list(CYCLE_KEYS), dropna=False):
        if g["day_of_cycle"].isna().all():
            continue
        g = g.sort_values("day_of_cycle", kind="mergesort")
        idx = g.index.to_list()
        days = pd.to_numeric(out.loc[idx, "day_of_cycle"], errors="coerce").to_numpy(dtype=float)

        primary_imputed = None
        for col in temp_cols:
            if col not in out.columns:
                continue
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy(dtype=float)
            filled, imputed = _interp_cycle_series(days, vals, method)
            out.loc[idx, col] = filled
            if col == primary:
                primary_imputed = imputed

        if primary_imputed is not None and primary_imputed.any():
            out.loc[np.array(idx)[primary_imputed], flag_col] = 1

    return out


# ---------------------------------------------------------------------------
# 24 slots
# ---------------------------------------------------------------------------
def _night_temp(night: pd.DataFrame, observed_only: bool = False) -> float:
    """
    Nightly temperature scalar.
    observed_only=True  → temp_mask=1 only (pre-interpolation diagnostics / interpolation input)
    observed_only=False → observed first, else finite skin_temp (including interpolated)
    """
    if TEMP_24 not in night.columns:
        return float("nan")
    v = pd.to_numeric(night[TEMP_24], errors="coerce")
    if "temp_mask" in night.columns:
        m = night["temp_mask"].astype(float) > 0.5
        obs = v[m]
        obs = obs[np.isfinite(obs)]
        if len(obs):
            return float(obs.mean())
        if observed_only:
            return float("nan")
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def interpolate_24(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    night_keys = [c for c in NIGHT_KEY_COLS if c in out.columns]
    if len(night_keys) < 2:
        raise KeyError(f"24 path: missing night key columns: {NIGHT_KEY_COLS}")

    raw_col = f"{TEMP_24}_raw"
    if raw_col not in out.columns:
        out[raw_col] = pd.to_numeric(out[TEMP_24], errors="coerce")
    out[TEMP_24] = pd.to_numeric(out[TEMP_24], errors="coerce")
    out["temp_day_imputed"] = 0
    out["temp_slot_imputed"] = 0

    # Fill the slot axis (10-min grid within a night) first. The 24 path guarantees ≥1 observed temperature
    # per night via the 00 no_temp_in_window gate, so the day axis (per-night scalar) alone would leave mask=0 slots forever.
    for _, night in out.groupby(night_keys, sort=False):
        idx = night.index.to_numpy()
        if WINDOW_COL in night.columns:
            w = pd.to_numeric(night[WINDOW_COL], errors="coerce").to_numpy(dtype=float)
        else:
            w = np.arange(len(idx), dtype=float)
        order = np.argsort(w, kind="mergesort")
        idx, w = idx[order], w[order]
        vals = pd.to_numeric(out.loc[idx, TEMP_24], errors="coerce").to_numpy(dtype=float)
        if "temp_mask" in out.columns:
            # Interpolation input is observations only (prevents already filled values leaking in on rerun)
            obs = out.loc[idx, "temp_mask"].astype(float).to_numpy() > 0.5
            vals = np.where(obs, vals, np.nan)
        filled, imputed = _interp_cycle_series(w, vals, method)
        # The day-axis filler rejects <2 observations. The slot axis fills the rest even from a single observation
        # (00 guarantees ≥1). temp_mask keeps the observed flag; only filled cells get temp_slot_imputed=1.
        still = ~np.isfinite(filled)
        n_obs = int(np.isfinite(vals).sum())
        if still.any() and n_obs == 1:
            filled[still] = float(vals[np.isfinite(vals)][0])
            imputed = imputed | still
        if imputed.any():
            out.loc[idx[imputed], TEMP_24] = filled[imputed]
            out.loc[idx[imputed], "temp_slot_imputed"] = 1

    # Day axis: only fully missing nights are pulled from neighbouring nights (what the slot axis could not fill)
    night_rows = []
    for key, night in out.groupby(night_keys, sort=False):
        pid, fname = key
        row0 = night.iloc[0]
        night_rows.append(
            {
                "participant_id": pid,
                "file_name_x": fname,
                "cycle": row0.get("cycle", np.nan),
                "day_of_cycle": row0.get("day_of_cycle", np.nan),
                # Interpolation input is observations only (avoids contamination even if filled values exist)
                "_night_temp": _night_temp(night, observed_only=True),
                "_idx": night.index.to_numpy(),
            }
        )
    nights = pd.DataFrame(night_rows)

    for (_, _), g in nights.groupby(list(CYCLE_KEYS), dropna=False):
        if g["day_of_cycle"].isna().all():
            continue
        g = g.sort_values("day_of_cycle", kind="mergesort")
        days = pd.to_numeric(g["day_of_cycle"], errors="coerce").to_numpy(dtype=float)
        # Duplicate nights on the same cycle day: overlapping interp x breaks, so add a tiny offset in order of appearance
        # (participant×resolved duplicates are a QC warning; here we only guard)
        if len(days) > 1 and np.any(np.diff(np.sort(days[np.isfinite(days)])) == 0):
            days = days + np.arange(len(days), dtype=float) * 1e-6
        vals = g["_night_temp"].to_numpy(dtype=float)
        filled, imputed = _interp_cycle_series(days, vals, method)

        for i, (_, nrow) in enumerate(g.iterrows()):
            if not imputed[i] or not np.isfinite(filled[i]):
                continue
            slot_idx = nrow["_idx"]
            t_fill = float(filled[i])
            # Keep observed slots; fill only missing ones (mask=0 or NaN)
            if "temp_mask" in out.columns:
                miss = out.loc[slot_idx, "temp_mask"].astype(float) <= 0.5
            else:
                miss = ~np.isfinite(pd.to_numeric(out.loc[slot_idx, TEMP_24], errors="coerce"))
            if np.any(miss.to_numpy() if hasattr(miss, "to_numpy") else miss):
                mi = slot_idx[np.asarray(miss, dtype=bool)]
                out.loc[mi, TEMP_24] = t_fill
                out.loc[mi, "temp_day_imputed"] = 1

    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def resolve_temp_cols(df: pd.DataFrame, tag: str) -> list[str]:
    primary = TEMP_COL_BY_TAG.get(tag)
    cols: list[str] = []
    if primary and primary in df.columns:
        cols.append(primary)
    elif "temp_mean" in df.columns:
        cols.append("temp_mean")
    elif "bbt" in df.columns:
        cols.append("bbt")
    for c in EXTRA_TEMP_COLS_BY_TAG.get(tag, ()):
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def enrich_cycle(tag: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply 00_*'s enrich_*_cycle only for huawei/npj/oura (+24/24mid)."""
    base = tag
    if tag.startswith("huawei"):
        base = "huawei"
    elif tag.startswith("npj"):
        base = "npj"
    elif tag.startswith("oura"):
        base = "oura"
    if base not in ("huawei", "npj", "oura"):
        return df
    mod_name = f"00_master_{base}"
    mod_path = SRC_DIR / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if base == "huawei":
        out = mod.enrich_huawei_cycle(df)
    elif base == "npj":
        out = mod.enrich_npj_cycle(df)
    else:
        out = mod.enrich_oura_cycle(df)
    print(f"  [{tag}] enrich_{base}_cycle applied")
    return out


def process_one(df: pd.DataFrame, tag: str, method: str) -> Path:
    """Apply method interpolation (+enrich) to an already loaded labelled df and save."""
    out_path = out_path_for(tag, method)
    print("\n" + "-" * 60)
    print(f"  [{tag}/{method}]")

    is_24 = tag in ("24", "24mid") or ("window" in df.columns and TEMP_24 in df.columns)
    if is_24:
        print(f"  [{tag}/{method}] mode: 24 slots — slot-axis interpolation → day axis for fully missing nights")
        before = _night_temp_miss_rate(df, observed_only=True)
        out = interpolate_24(df, method=method)
        after_obs = _night_temp_miss_rate(out, observed_only=True)
        after_eff = _night_temp_miss_rate(out, observed_only=False)
        n_imp = int(out["temp_day_imputed"].sum())
        n_slot_imp = int(out["temp_slot_imputed"].sum())
        slot_miss = float(pd.to_numeric(out[TEMP_24], errors="coerce").isna().mean())
        print(
            f"  [{tag}/{method}] night temp missing rate (observed): {before:.1%} → {after_obs:.1%}  "
            f"| effective (incl. interpolated): {after_eff:.1%}  "
            f"(slot-axis imputed={n_slot_imp}, day-axis imputed={n_imp}, remaining slot missing rate={slot_miss:.1%})"
        )
    else:
        temp_cols = resolve_temp_cols(df, tag)
        if not temp_cols:
            raise KeyError(f"[{tag}] no temp column: cols={list(df.columns)[:40]}")
        print(f"  [{tag}/{method}] mode: one row per night — cols={temp_cols}")
        primary = temp_cols[0]
        before = float(pd.to_numeric(df[primary], errors="coerce").isna().mean())
        out = interpolate_nightly(df, temp_cols=temp_cols, method=method)
        after = float(pd.to_numeric(out[primary], errors="coerce").isna().mean())
        n_imp = int(out["temp_day_imputed"].sum())
        print(f"  [{tag}/{method}] {primary} missing rate: {before:.1%} → {after:.1%}  (rows imputed={n_imp})")

    out["temp_day_method"] = method
    out = enrich_cycle(tag, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_path, index=False, engine="openpyxl")
    print(f"  [{tag}/{method}] saved (combined): {out_path}")

    split_dir = split_dir_for(tag, method)
    per = save_per_label(out, split_dir)
    print(f"  [{tag}/{method}] saved ({len(per)} per-label files): {split_dir}/")
    final_dir = final_split_dir_for(tag, method)
    if final_dir is not None:
        per_f = save_per_label(out, final_dir)
        print(f"  [{tag}/{method}] saved ({len(per_f)} final files): {final_dir}/")
    return out_path


def _night_temp_miss_rate(df: pd.DataFrame, observed_only: bool = True) -> float:
    """Per-night temp missing rate (diagnostic)."""
    night_keys = [c for c in NIGHT_KEY_COLS if c in df.columns]
    if len(night_keys) < 2:
        return float("nan")
    miss = []
    for _, night in df.groupby(night_keys, sort=False):
        miss.append(not np.isfinite(_night_temp(night, observed_only=observed_only)))
    return float(np.mean(miss)) if miss else float("nan")


def _self_check() -> None:
    """Smoke test of the interpolation logic. Asserts on failure."""
    d = pd.DataFrame(
        {
            "participant_id": ["P"] * 5,
            "cycle": [1] * 5,
            "day_of_cycle": [1, 2, 3, 4, 5],
            "temp_mean": [36.0, np.nan, 36.4, np.nan, 36.8],
        }
    )
    o = interpolate_nightly(d, ["temp_mean"], "linear")
    assert abs(float(o.loc[1, "temp_mean"]) - 36.2) < 1e-9
    assert abs(float(o.loc[3, "temp_mean"]) - 36.6) < 1e-9
    assert int(o["temp_day_imputed"].sum()) == 2

    rows = []
    for night, fname, day, base, half_miss in (
        (1, "a", 1, 36.0, False),
        (2, "b", 2, np.nan, True),  # fully missing → day
        (3, "c", 3, 36.6, True),  # half missing → slot axis
    ):
        for w in range(1, 25):
            if night == 2:
                t, m = np.nan, 0
            elif night == 3 and w > 12:
                t, m = np.nan, 0
            else:
                t, m = base, 1
            rows.append(
                {
                    "participant_id": "P",
                    "file_name_x": fname,
                    "cycle": 1,
                    "day_of_cycle": day,
                    "window": w,
                    "skin_temp": t,
                    "temp_mask": m,
                }
            )
    # Night with one observed slot — constant broadcast + temp_slot_imputed
    for w in range(1, 25):
        rows.append(
            {
                "participant_id": "P",
                "file_name_x": "d",
                "cycle": 1,
                "day_of_cycle": 4,
                "window": w,
                "skin_temp": 36.4 if w == 5 else np.nan,
                "temp_mask": 1 if w == 5 else 0,
            }
        )
    o24 = interpolate_24(pd.DataFrame(rows), "linear")
    n2 = o24[o24["file_name_x"] == "b"]
    assert np.isfinite(n2["skin_temp"]).all()
    assert abs(float(n2["skin_temp"].mean()) - 36.3) < 1e-9
    assert int(n2["temp_day_imputed"].sum()) == 24
    n3 = o24[o24["file_name_x"] == "c"]
    assert np.isfinite(n3["skin_temp"]).all()
    assert abs(float(n3.loc[n3["temp_mask"] == 0, "skin_temp"].iloc[0]) - 36.6) < 1e-9
    assert int(n3["temp_slot_imputed"].sum()) == 12
    assert int(n3["temp_day_imputed"].sum()) == 0
    n4 = o24[o24["file_name_x"] == "d"]
    assert np.isfinite(n4["skin_temp"]).all()
    assert (n4["skin_temp"] == 36.4).all()
    assert int(n4["temp_slot_imputed"].sum()) == 23
    assert int(n4.loc[n4["temp_mask"] == 1, "temp_slot_imputed"].sum()) == 0
    print("self-check OK")


def main():
    ap = argparse.ArgumentParser(description="02 within-cycle daily temp interpolation")
    ap.add_argument(
        "--method",
        default="all",
        choices=[*METHODS, "all"],
        help="interpolation method (default: all = save linear and locf separately)",
    )
    ap.add_argument(
        "--tag",
        default=None,
        choices=list(TAGS),
        help="process only specific sources (default: all)",
    )
    ap.add_argument("--self-check", action="store_true", help="run the unit smoke test only and exit")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return
    methods = list(METHODS) if args.method == "all" else [args.method]

    print("=" * 60)
    print(f"  [02] within-cycle daily temp interpolation  (methods={methods})")
    print("=" * 60)

    saved: list[Path] = []
    missing: list[str] = []
    for tag in TAGS:
        if args.tag is not None and tag != args.tag:
            continue
        labeled_path = LABELED_PATH[tag]
        if not labeled_path.is_file():
            print(f"\n  [skip {tag}] no input: {labeled_path}")
            missing.append(tag)
            continue
        print(f"\n  [{tag}] LABELED: {labeled_path}")
        df = pd.read_excel(labeled_path, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        for method in methods:
            saved.append(process_one(df, tag, method=method))

    if not saved:
        raise FileNotFoundError(
            "No 01 output to process. At least one of the following is required:\n"
            + "\n".join(f"  - {LABELED_PATH[t]}" for t in TAGS)
        )
    print("\n" + "=" * 60)
    print(f"  done: {len(saved)} saved" + (f" (skip: {missing})" if missing else ""))
    for p in saved:
        print(f"    → {p}")
    print(
        "Next: 03 (24/24mid HRV) → 04 (24 slots → one row per night); "
        "per-night tags final → final/interp_{tag}/{method}/"
    )


if __name__ == "__main__":
    main()
