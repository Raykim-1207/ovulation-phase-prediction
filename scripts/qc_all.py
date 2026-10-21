# -*- coding: utf-8 -*-
"""Checks in one pass that HRV in every family falls within physiological ranges.

After the detector fix only one family was inspected by eye. All fifteen families were rebuilt, so check
them all. Stop if any family has a sleeping mean heart rate outside 45~85 bpm or RMSSD>SDNN in the majority
of nights — results run on that family cannot be trusted.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("OVUL_DATA_DIR_ROOT",
                           str(Path(__file__).resolve().parents[1] / "ovul_data"))) / "finalfinal"


def main():
    files = sorted(ROOT.glob("interp_*/*/y4_std.xlsx"))
    assert files, f"No files to check: {ROOT}"
    bad = []
    print(f"{'family/interp':34s} {'nights':>6s} {'HR':>7s} {'SDNN':>7s} {'RMSSD':>7s} {'R>S':>6s}")
    print("-" * 72)
    for f in files:
        d = pd.read_excel(f)
        tag = f"{f.parents[1].name.replace('interp_','')}/{f.parent.name}"
        if "hr" not in d or "rmssd" not in d:      # slot families have different column names
            print(f"{tag:34s} {len(d):6d}   (no HRV summary columns — skipped)")
            continue
        hr, rs = d.hr.mean(), (d.rmssd > d.sdnn).mean()
        ok = 45 < hr < 85 and rs < 0.5
        print(f"{tag:34s} {len(d):6d} {hr:7.1f} {d.sdnn.mean():7.1f} "
              f"{d.rmssd.mean():7.1f} {rs*100:5.1f}%" + ("" if ok else "   ← out of range"))
        if not ok:
            bad.append((tag, round(hr, 1), round(rs, 3)))
    print()
    assert not bad, f"Families outside physiological range: {bad}"
    print(f"Check passed — {len(files)} files")


if __name__ == "__main__":
    main()
