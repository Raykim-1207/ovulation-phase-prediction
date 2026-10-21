# -*- coding: utf-8 -*-
"""
06_add_m_cycle — add measured cycle length and group columns to the final xlsx.

  m_cycle       : (next_menses_start − menses_start).days
  m_cycle_group : regular | mild_irregular | high_irregular
    regular         27–32
    mild_irregular  24–26 or 33–35
    high_irregular  ≤23 or ≥36

Do not use as a feature (next menses is observed after the fact → training input leakage).
For subgroup evaluation only.

  python3 .../06_add_m_cycle.py
  python3 .../06_add_m_cycle.py --root /path/to/finalfinal --dry-run
  python3 .../06_add_m_cycle.py --self-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "ovul_data" / "finalfinal"

MSD = "menses_start"
NMSD = "next_menses_start"
COL_LEN = "m_cycle"
COL_GRP = "m_cycle_group"


def m_cycle_group(length: float) -> str:
    if length is None or (isinstance(length, float) and not np.isfinite(length)):
        return "unknown"
    n = int(length)
    if 27 <= n <= 32:
        return "regular"
    if 24 <= n <= 26 or 33 <= n <= 35:
        return "mild_irregular"
    if n <= 23 or n >= 36:
        return "high_irregular"
    return "unknown"


def add_m_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if MSD not in out.columns or NMSD not in out.columns:
        raise KeyError(f"need {MSD} and {NMSD}")
    msd = pd.to_datetime(out[MSD], errors="coerce")
    nmsd = pd.to_datetime(out[NMSD], errors="coerce")
    length = (nmsd - msd).dt.days
    out[COL_LEN] = length
    out[COL_GRP] = [m_cycle_group(x) if pd.notna(x) else "unknown" for x in length]
    return out


def _self_check() -> None:
    assert m_cycle_group(27) == "regular"
    assert m_cycle_group(32) == "regular"
    assert m_cycle_group(24) == "mild_irregular"
    assert m_cycle_group(26) == "mild_irregular"
    assert m_cycle_group(33) == "mild_irregular"
    assert m_cycle_group(35) == "mild_irregular"
    assert m_cycle_group(23) == "high_irregular"
    assert m_cycle_group(21) == "high_irregular"
    assert m_cycle_group(36) == "high_irregular"
    assert m_cycle_group(40) == "high_irregular"
    df = pd.DataFrame({
        MSD: ["2024-01-01", "2024-02-01"],
        NMSD: ["2024-01-29", "2024-03-10"],  # 28d regular, 38d high
    })
    out = add_m_cycle_columns(df)
    assert list(out[COL_LEN]) == [28, 38]
    assert list(out[COL_GRP]) == ["regular", "high_irregular"]
    print("06_add_m_cycle self-check OK")


def main() -> None:
    p = argparse.ArgumentParser(description="Add m_cycle / m_cycle_group to finalfinal")
    p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        _self_check()
        return

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"missing data root: {root}")

    files = sorted(
        path for path in root.rglob("*.xlsx")
        # seq24 sheets are huge; night tabular is enough for m_cycle meta
        if "interp_24" not in path.parts
    )
    if not files:
        raise SystemExit(f"no xlsx under {root}")

    n_ok = n_skip = n_fail = 0
    grp_counts: dict[str, int] = {}
    for path in files:
        try:
            df = pd.read_excel(path, engine="openpyxl")
            df.columns = [str(c).strip() for c in df.columns]
            if MSD not in df.columns or NMSD not in df.columns:
                n_skip += 1
                continue
            out = add_m_cycle_columns(df)
            for g, c in out[COL_GRP].value_counts().items():
                grp_counts[str(g)] = grp_counts.get(str(g), 0) + int(c)
            if args.dry_run:
                n_ok += 1
                continue
            out.to_excel(path, index=False, engine="openpyxl")
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {path.relative_to(root)}: {e}", file=sys.stderr)

    mode = "dry-run" if args.dry_run else "wrote"
    print(f"06_add_m_cycle {mode}: ok={n_ok} skip={n_skip} fail={n_fail} files={len(files)}")
    print("  group row-counts (sum over files):", dict(sorted(grp_counts.items())))


if __name__ == "__main__":
    main()
