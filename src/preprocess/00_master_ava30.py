# -*- coding: utf-8 -*-
"""
Path 1, ava30: Ava-style per-night aggregation — trim +30/−30 min (LOESS obs clip).

Re-runs the parent script via env so that ProcessPool pickling stays safe.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AVA = HERE / "00_master_ava.py"


def main(workers: int = 8) -> None:
    env = os.environ.copy()
    # Set the +30/-30 trim explicitly instead of the 90/30 default (the parent script reads env)
    env["OVUL_AVA_TRIM_HEAD"] = "30"
    env["OVUL_AVA_TRIM_TAIL"] = "30"
    env["OVUL_AVA_TAG"] = "ava30"
    env["OVUL_AVA_OUT"] = "ovul_data_ava30.xlsx"
    r = subprocess.run(
        [sys.executable, str(AVA), f"--workers={workers}"],
        cwd=str(HERE),
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ava per-night aggregation: trim +30/−30")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(workers=args.workers)

