# -*- coding: utf-8 -*-
"""Merge scattered grid outputs into a single summary.csv.

The grid ran in parallel with one process per family, so there are several output roots.
The table generators read only one summary.csv, so they are gathered here.

  python3 scripts/merge_grid.py
Output -> results/grid_all/summary.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "grid_all"
NEED = 15          # 15 design x window cells


def main():
    srcs = sorted(set(list(R.glob("grid/summary.csv"))
                      + list(R.glob("grid/shard_*/summary.csv"))
                      + list(R.glob("grid_par/*/summary.csv"))
                      + list(R.glob("grid_par/*/shard_*/summary.csv"))))
    assert srcs, "no summary.csv found to merge"
    d = pd.concat([pd.read_csv(f) for f in srcs], ignore_index=True)
    key = ["arm", "family", "method", "label", "split", "group", "model"]
    before = len(d)
    d = d.drop_duplicates(subset=key, keep="last")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "summary.csv", index=False)

    s = d[(d.arm == "combined") & (d.label == "y4_std")
          & (d.split == "sg") & (d.group == "ml")]
    fams = sorted(x.replace("interp_", "") for x in s.family.unique())
    print(f"sources {len(srcs)} files · {before} rows → {len(d)} rows after dedup")
    print(f"families for the table {len(fams)}/{NEED}: {' '.join(fams)}")
    if len(fams) < NEED:
        print(f"  still missing {NEED - len(fams)} families")
    return 0 if len(fams) >= NEED else 1


if __name__ == "__main__":
    raise SystemExit(main())
