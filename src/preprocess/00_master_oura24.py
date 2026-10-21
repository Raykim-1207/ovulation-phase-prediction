# -*- coding: utf-8 -*-
"""oura24: Oura aggregation + resolved 02:00–06:00 window."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "00_master_oura.py"


def main(workers: int = 8) -> None:
    env = os.environ.copy()
    env["OVUL_WINDOW"] = "0206"
    env["OVUL_TAG"] = "oura24"
    env["OVUL_OUT"] = "ovul_data_oura24.xlsx"
    r = subprocess.run(
        [sys.executable, str(BASE), f"--workers={workers}"],
        cwd=str(HERE),
        env=env,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Oura: resolved 02–06")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(workers=args.workers)
