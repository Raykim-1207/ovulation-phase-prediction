# -*- coding: utf-8 -*-
"""
Path ② mid4h: in-night sequence — **middle ±2 hours** of the recording [start, end] (24×10 min).

The resolved 02–06 version is ``00_master_24.py``.
Output: ovul_data/pre/ovul_data_24mid.xlsx

Re-runs the parent script via env so that ProcessPool pickling is safe.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE24 = HERE / "00_master_24.py"


def main(workers: int = 8) -> None:
    env = os.environ.copy()
    env["OVUL_24_WINDOW"] = "mid4h"
    env["OVUL_24_OUT"] = "ovul_data_24mid.xlsx"
    r = subprocess.run(
        [sys.executable, str(BASE24), f"--workers={workers}"],
        cwd=str(HERE),
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="24 slots: middle 4 hours of recording")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(workers=args.workers)
