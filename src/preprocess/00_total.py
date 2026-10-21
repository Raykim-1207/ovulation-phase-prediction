# -*- coding: utf-8 -*-
"""Run the 00_* master scripts sequentially (next starts when one finishes).

Prerequisite: if qc_keep_codes.csv already exists, QC need not be rerun.
Resume: skip when the output xlsx exists. ``--force`` reruns everything.
Parallel: pass ``--workers N`` (default 8) to each 00_*.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_r = __import__("os").environ.get("OVUL_DATA_DIR_ROOT", "").strip()
OUT_ROOT = Path(_r) if _r else (HERE.parents[1] / "ovul_data")

PRE_DIR = OUT_ROOT / "pre"

JOBS = [
    ("00_master_24.py", PRE_DIR / "ovul_data_24.xlsx"),
    ("00_master_24mid.py", PRE_DIR / "ovul_data_24mid.xlsx"),
    # for the ava family, the LOESS obs-clip variant takes precedence
    ("00_master_ava.py", PRE_DIR / "ovul_data_ava.xlsx"),
    ("00_master_ava30.py", PRE_DIR / "ovul_data_ava30.xlsx"),
    ("00_master_ava24.py", PRE_DIR / "ovul_data_ava24.xlsx"),
    ("00_master_ava24mid.py", PRE_DIR / "ovul_data_ava24mid.xlsx"),
    ("00_master_huawei.py", PRE_DIR / "ovul_data_huawei.xlsx"),
    ("00_master_huawei24.py", PRE_DIR / "ovul_data_huawei24.xlsx"),
    ("00_master_huawei24mid.py", PRE_DIR / "ovul_data_huawei24mid.xlsx"),
    ("00_master_npj.py", PRE_DIR / "ovul_data_npj.xlsx"),
    ("00_master_npj24.py", PRE_DIR / "ovul_data_npj24.xlsx"),
    ("00_master_npj24mid.py", PRE_DIR / "ovul_data_npj24mid.xlsx"),
    ("00_master_oura.py", PRE_DIR / "ovul_data_oura.xlsx"),
    ("00_master_oura24.py", PRE_DIR / "ovul_data_oura24.xlsx"),
    ("00_master_oura24mid.py", PRE_DIR / "ovul_data_oura24mid.xlsx"),
]


def _done(out: Path) -> bool:
    return out.is_file() and out.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 00_* sequentially (+skip completed)")
    ap.add_argument("--force", action="store_true", help="rerun everything even if outputs exist")
    ap.add_argument("--workers", type=int, default=8, help="number of per-night ProcessPool workers")
    args = ap.parse_args()
    workers = max(1, int(args.workers))

    keep = PRE_DIR / "qc_keep_codes.csv"
    if not keep.is_file():
        raise FileNotFoundError(f"{keep} missing. Run the _raw_common.py QC first.")

    print(f"OUT pre dir: {PRE_DIR}", flush=True)
    print(f"Using QC keep: {keep}  workers={workers}", flush=True)

    for i, (name, out) in enumerate(JOBS, 1):
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if not args.force and _done(out):
            print(f"\n===== [{i}/{len(JOBS)}] {name} SKIP (exists: {out.name}) =====\n", flush=True)
            continue
        print(f"\n===== [{i}/{len(JOBS)}] {name} → {out.name} =====\n", flush=True)
        r = subprocess.run(
            [sys.executable, str(path), f"--workers={workers}"],
            cwd=str(HERE),
        )
        if r.returncode != 0:
            print(f"FAIL {name} exit={r.returncode}", flush=True)
            raise SystemExit(r.returncode)
        if not _done(out):
            print(f"FAIL {name}: no output {out}", flush=True)
            raise SystemExit(1)
        print(f"OK {name} → {out.name}", flush=True)
    print(f"\nAll done ({len(JOBS)} jobs)", flush=True)


if __name__ == "__main__":
    main()
