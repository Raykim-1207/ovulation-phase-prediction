# -*- coding: utf-8 -*-
"""
05 — clean final → finalfinal.

1) Body temp < TEMP_MIN (default 35°C) → set to NaN, then **re-interpolate** (rows are kept)
   · one row per night / 24agg: along the day_of_cycle axis, using the file's ``temp_day_method`` (linear|locf)
   · 24 slots: interpolate along the window axis **among slots of the same night**. Nights still fully missing
     are interpolated along the day axis from neighboring nights' scalars (e.g. a participant whose whole night is <35)
2) Drop HRV/temperature feature columns whose finite values are all 0 (ulf*)
3) For huawei/npj/oura, rerun enrich_*_cycle after the temp fix

Input: ovul_data/final/**
Output: ovul_data/finalfinal/**  (same relative paths)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
_r = __import__("os").environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_DIR = Path(_r) if _r else (HERE.parents[1] / "ovul_data")
IN_ROOT = OUT_DIR / "final"
OUT_ROOT = OUT_DIR / "finalfinal"

TEMP_MIN = 35.0
NIGHT_KEYS = ("participant_id", "file_name_x")
CYCLE_KEYS = ("participant_id", "cycle")
TEMP_PRIMARY = ("skin_temp", "bbt", "temp_mean")
TEMP_24 = "skin_temp"
WINDOW_COL = "window"

KEEP_ALWAYS = {
    "participant_id", "file_name_x", "cycle", "measure_date", "label_date", "day_of_cycle",
    "menses_start", "next_menses_start", "ovulation_date", "measure_time",
    "ecg_start", "used_from", "used_to", "temp_night_source", "window",
    "temp_day_method",
    "y5_mod", "y4_mod", "y3_mod", "y2_mod",
    "y5_std", "y4_std", "y3_std", "y2_std",
    "hrv_mask", "temp_mask", "hrv_dt", "temp_dt",
    "temp_slot_imputed", "temp_day_imputed",
    "n_hrv_win", "n_temp_pts", "used_min", "record_min", "n_slots",
}


def _load_02():
    spec = importlib.util.spec_from_file_location(
        "ovul_02_temp", HERE / "02_temp_day_interp.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_M02 = None


def m02():
    global _M02
    if _M02 is None:
        _M02 = _load_02()
    return _M02


def primary_temp_col(df: pd.DataFrame) -> str | None:
    for c in TEMP_PRIMARY:
        if c in df.columns:
            return c
    return None


def resolve_method(df: pd.DataFrame, fallback: str = "linear") -> str:
    if "temp_day_method" in df.columns:
        vals = df["temp_day_method"].dropna().astype(str).str.strip()
        if len(vals):
            m = vals.iloc[0]
            if m in ("linear", "locf"):
                return m
    return fallback


def tag_from_rel(rel: Path) -> str:
    """interp_huawei24/linear/y5_mod.xlsx → huawei24 ; interp_24/... → 24."""
    fam = rel.parts[0]
    name = fam[len("interp_") :] if fam.startswith("interp_") else fam
    if False:  # a legacy suffix no longer exists
        pass
    return name


def is_drop_col_candidate(col: str) -> bool:
    if col in KEEP_ALWAYS:
        return False
    if col.endswith("_raw") or col.endswith("_imputed"):
        return False
    return True


def all_zero_feature_cols(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if not is_drop_col_candidate(str(c)):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            continue
        if bool((finite == 0).all()):
            out.append(str(c))
    return out


def _slot_fill_night(
    vals: np.ndarray,
    windows: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one night along the window axis. With one observation, broadcast a constant (same as 02)."""
    order = np.argsort(windows, kind="mergesort")
    w = windows[order]
    v = vals[order]
    filled, imputed = m02()._interp_cycle_series(w, v, method)
    still = ~np.isfinite(filled)
    n_obs = int(np.isfinite(v).sum())
    if still.any() and n_obs == 1:
        filled[still] = float(v[np.isfinite(v)][0])
        imputed = imputed | still
    # unsort
    out = np.empty_like(filled)
    imp = np.zeros(len(vals), dtype=bool)
    out[order] = filled
    imp[order] = imputed
    return out, imp


def reinterp_cold_temps(
    df: pd.DataFrame,
    temp_min: float,
    method: str,
) -> tuple[pd.DataFrame, dict]:
    """
    temp < temp_min → NaN → re-interpolate.
    Returns (df, stats).
    """
    out = df.copy()
    tc = primary_temp_col(out)
    stats = {
        "n_cold_cells": 0,
        "n_refilled": 0,
        "n_still_nan": 0,
        "mode": None,
    }
    if tc is None:
        return out, stats

    t = pd.to_numeric(out[tc], errors="coerce").to_numpy(dtype=float)
    cold = np.isfinite(t) & (t < temp_min)
    stats["n_cold_cells"] = int(cold.sum())
    if cold.sum() == 0:
        stats["mode"] = "noop"
        return out, stats

    # keep raw (if absent, snapshot current values — including cold ones)
    raw_col = f"{tc}_raw"
    if raw_col not in out.columns:
        out[raw_col] = t

    t_work = t.copy()
    t_work[cold] = np.nan
    out[tc] = t_work

    is_24 = ("window" in out.columns) and (tc == TEMP_24) and all(
        k in out.columns for k in NIGHT_KEYS
    )

    if is_24:
        stats["mode"] = "slot+day_fallback"
        if "temp_slot_imputed" not in out.columns:
            out["temp_slot_imputed"] = 0
        if "temp_day_imputed" not in out.columns:
            out["temp_day_imputed"] = 0

        # 1) slot axis within a night
        for _, night in out.groupby(list(NIGHT_KEYS), sort=False):
            idx = night.index.to_numpy()
            w = pd.to_numeric(out.loc[idx, WINDOW_COL], errors="coerce").to_numpy(
                dtype=float
            )
            vals = pd.to_numeric(out.loc[idx, tc], errors="coerce").to_numpy(
                dtype=float
            )
            filled, imputed = _slot_fill_night(vals, w, method)
            if imputed.any():
                out.loc[idx[imputed], tc] = filled[imputed]
                out.loc[idx[imputed], "temp_slot_imputed"] = 1

        # 2) fully missing nights → day axis (neighboring nights)
        if "cycle" in out.columns and "day_of_cycle" in out.columns:
            night_rows = []
            for key, night in out.groupby(list(NIGHT_KEYS), sort=False):
                vals = pd.to_numeric(night[tc], errors="coerce")
                finite = vals[np.isfinite(vals)]
                night_rows.append(
                    {
                        "participant_id": key[0],
                        "file_name_x": key[1],
                        "cycle": night.iloc[0].get("cycle", np.nan),
                        "day_of_cycle": night.iloc[0].get("day_of_cycle", np.nan),
                        "_night_temp": float(finite.mean()) if len(finite) else np.nan,
                        "_idx": night.index.to_numpy(),
                        "_all_nan": len(finite) == 0,
                    }
                )
            nights = pd.DataFrame(night_rows)
            for (_, _), g in nights.groupby(list(CYCLE_KEYS), dropna=False):
                if g["day_of_cycle"].isna().all():
                    continue
                g = g.sort_values("day_of_cycle", kind="mergesort")
                days = pd.to_numeric(g["day_of_cycle"], errors="coerce").to_numpy(dtype=float)
                if len(days) > 1 and np.any(
                    np.diff(np.sort(days[np.isfinite(days)])) == 0
                ):
                    days = days + np.arange(len(days), dtype=float) * 1e-6
                vals = g["_night_temp"].to_numpy(dtype=float)
                filled, imputed = m02()._interp_cycle_series(days, vals, method)
                for i, (_, nrow) in enumerate(g.iterrows()):
                    if not imputed[i] or not np.isfinite(filled[i]):
                        continue
                    if not bool(nrow["_all_nan"]):
                        continue  # do not overwrite with day axis a night already partly filled from slots
                    slot_idx = nrow["_idx"]
                    out.loc[slot_idx, tc] = float(filled[i])
                    out.loc[slot_idx, "temp_day_imputed"] = 1
    else:
        stats["mode"] = "cycle_day"
        if "temp_day_imputed" not in out.columns:
            out["temp_day_imputed"] = 0
        if "cycle" not in out.columns or "day_of_cycle" not in out.columns:
            stats["n_still_nan"] = int(
                pd.to_numeric(out[tc], errors="coerce").isna().sum()
            )
            return out, stats

        for (_, _), g in out.groupby(list(CYCLE_KEYS), dropna=False):
            if g["day_of_cycle"].isna().all():
                continue
            g = g.sort_values("day_of_cycle", kind="mergesort")
            idx = g.index.to_numpy()
            days = pd.to_numeric(out.loc[idx, "day_of_cycle"], errors="coerce").to_numpy(
                dtype=float
            )
            vals = pd.to_numeric(out.loc[idx, tc], errors="coerce").to_numpy(
                dtype=float
            )
            filled, imputed = m02()._interp_cycle_series(days, vals, method)
            if imputed.any():
                out.loc[idx[imputed], tc] = filled[imputed]
                out.loc[idx[imputed], "temp_day_imputed"] = 1

    t_after = pd.to_numeric(out[tc], errors="coerce")
    # cells that were cold and then filled
    stats["n_refilled"] = int(cold.sum() - (cold & t_after.isna().to_numpy()).sum())
    stats["n_still_nan"] = int(t_after.isna().sum())
    return out, stats


def enrich_quiet(tag: str, df: pd.DataFrame) -> pd.DataFrame:
    """Same as 02.enrich_cycle, without printing."""
    base = tag
    if tag.startswith("huawei"):
        base = "huawei"
    elif tag.startswith("npj"):
        base = "npj"
    elif tag.startswith("oura"):
        base = "oura"
    else:
        return df
    mod_path = HERE / f"00_master_{base}.py"
    spec = importlib.util.spec_from_file_location(f"enrich_{base}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if base == "huawei":
        return mod.enrich_huawei_cycle(df)
    if base == "npj":
        return mod.enrich_npj_cycle(df)
    return mod.enrich_oura_cycle(df)


def process_file(
    src: Path,
    dst: Path,
    in_root: Path,
    temp_min: float,
    dry_run: bool = False,
) -> dict:
    df = pd.read_excel(src, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    n0 = len(df)
    method = resolve_method(df)
    rel = src.relative_to(in_root)
    tag = tag_from_rel(rel)

    df2, st = reinterp_cold_temps(df, temp_min=temp_min, method=method)
    if tag.startswith(("huawei", "npj", "oura")):
        df2 = enrich_quiet(tag, df2)

    zero_cols = all_zero_feature_cols(df2)
    if zero_cols:
        df2 = df2.drop(columns=zero_cols, errors="ignore")

    info = {
        "src": str(rel),
        "tag": tag,
        "method": method,
        "rows": n0,
        "mode": st["mode"],
        "n_cold_cells": st["n_cold_cells"],
        "n_refilled": st["n_refilled"],
        "n_still_nan": st["n_still_nan"],
        "cols_dropped_zero": zero_cols,
    }

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        df2.to_excel(dst, index=False, engine="openpyxl")
    return info


def _process_one(job: tuple) -> dict:
    src, dst, in_root, temp_min, dry_run = job
    return process_file(
        Path(src), Path(dst), Path(in_root), temp_min=temp_min, dry_run=dry_run
    )


def _self_check() -> None:
    # nightly: cold → neighbor linear
    d = pd.DataFrame(
        {
            "participant_id": ["P"] * 5,
            "cycle": [1] * 5,
            "day_of_cycle": [1, 2, 3, 4, 5],
            "file_name_x": list("abcde"),
            "temp_mean": [36.0, 33.0, 36.4, 34.0, 36.8],
            "temp_day_method": ["linear"] * 5,
            "ulf_mean": [0.0] * 5,
        }
    )
    out, st = reinterp_cold_temps(d, temp_min=35.0, method="linear")
    assert st["n_cold_cells"] == 2
    assert abs(float(out.loc[1, "temp_mean"]) - 36.2) < 1e-6  # between 36.0 and 36.4
    assert abs(float(out.loc[3, "temp_mean"]) - 36.6) < 1e-6
    assert all_zero_feature_cols(out) == ["ulf_mean"]

    # 24 slots: trailing cold filled from last good (np.interp edge)
    rows = []
    for w in range(1, 25):
        rows.append(
            {
                "participant_id": "P",
                "file_name_x": "n1",
                "cycle": 1,
                "day_of_cycle": 10,
                "window": w,
                "skin_temp": 37.0 if w <= 16 else 25.0,
                "temp_mask": 1,
                "temp_day_method": "linear",
            }
        )
    d24 = pd.DataFrame(rows)
    out24, st24 = reinterp_cold_temps(d24, temp_min=35.0, method="linear")
    assert st24["n_cold_cells"] == 8
    assert (pd.to_numeric(out24["skin_temp"]) >= 35).all()
    assert abs(float(out24.loc[out24["window"] == 20, "skin_temp"].iloc[0]) - 37.0) < 1e-6

    # all-cold night → day fallback from neighbor
    rows2 = []
    for fname, day, base in (("a", 1, 36.5), ("b", 2, 25.0), ("c", 3, 37.0)):
        for w in range(1, 25):
            rows2.append(
                {
                    "participant_id": "Q",
                    "file_name_x": fname,
                    "cycle": 1,
                    "day_of_cycle": day,
                    "window": w,
                    "skin_temp": base,
                    "temp_mask": 1,
                    "temp_day_method": "linear",
                }
            )
    out_fb, st_fb = reinterp_cold_temps(pd.DataFrame(rows2), 35.0, "linear")
    b = out_fb[out_fb["file_name_x"] == "b"]
    assert st_fb["n_cold_cells"] == 24
    assert np.isfinite(pd.to_numeric(b["skin_temp"])).all()
    # linear between 36.5 and 37.0 on day 2 → 36.75
    assert abs(float(b["skin_temp"].mean()) - 36.75) < 1e-6
    print("05 self-check OK")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="05 final → finalfinal (cold NaN→reinterp + drop all-zero cols)"
    )
    ap.add_argument("--in-root", type=Path, default=IN_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--temp-min", type=float, default=TEMP_MIN)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return

    in_root: Path = args.in_root
    out_root: Path = args.out_root
    if not in_root.is_dir():
        raise FileNotFoundError(f"input missing: {in_root}")
    files = sorted(in_root.rglob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no xlsx: {in_root}")

    print("=" * 60)
    print("  [05] final → finalfinal  (cold→NaN→reinterp, drop all-zero)")
    print(f"  in:  {in_root}")
    print(f"  out: {out_root}" + ("  (dry-run)" if args.dry_run else ""))
    print(f"  temp_min={args.temp_min}  files={len(files)}")
    print("=" * 60)
    _self_check()

    jobs = [
        (
            str(src),
            str(out_root / src.relative_to(in_root)),
            str(in_root),
            args.temp_min,
            args.dry_run,
        )
        for src in files
    ]
    n_workers = max(1, min(8, os.cpu_count() or 4))
    summaries: list[dict] = []
    zero_col_set: set[str] = set()
    cold_cells = 0
    refilled = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="05 clean", unit="file"):
            info = fut.result()
            summaries.append(info)
            zero_col_set.update(info["cols_dropped_zero"])
            cold_cells += info["n_cold_cells"]
            refilled += info["n_refilled"]

    print("\n" + "=" * 60)
    print(f"  done: {len(summaries)} files")
    print(f"  cold cells (<{args.temp_min}): {cold_cells}  refilled≈{refilled}")
    print(f"  all-zero cols: {sorted(zero_col_set) or '(none)'}")
    by_fam: dict[str, list[dict]] = {}
    for s in summaries:
        fam = Path(s["src"]).parts[0]
        by_fam.setdefault(fam, []).append(s)
    for fam, items in sorted(by_fam.items()):
        cc = sum(i["n_cold_cells"] for i in items)
        rf = sum(i["n_refilled"] for i in items)
        z = sorted({c for i in items for c in i["cols_dropped_zero"]})
        print(f"  · {fam}: cold={cc} refilled≈{rf} zero_cols={z}")
    if args.dry_run:
        print("  (dry-run: files not saved)")
    else:
        print(f"  → {out_root}")


if __name__ == "__main__":
    main()
