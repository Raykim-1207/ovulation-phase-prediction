# -*- coding: utf-8 -*-
"""ava24mid: Ava-style aggregation + recording center ±2h (same as 24mid). LOESS obs clip."""
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
    env["OVUL_WINDOW"] = "mid4h"
    env["OVUL_AVA_TAG"] = "ava24mid"
    env["OVUL_AVA_OUT"] = "ovul_data_ava24mid.xlsx"
    r = subprocess.run(
        [sys.executable, str(AVA), f"--workers={workers}"],
        cwd=str(HERE),
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ava per night: recording center 4h (LOESS clip)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(workers=args.workers)
