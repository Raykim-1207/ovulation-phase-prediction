# -*- coding: utf-8 -*-
"""Build the 15-cell design x window table.

Fig 2 lays out fifteen cells, but their performance was not tabulated anywhere in the
manuscript. Supplementary Table S2a lists only "the best window per design", collapsing
the window axis.

There are two sources to read.

  OVUL_BOOT_DIR  — grid_bootstrap output. OvCast + participant-level bootstrap,
                   1,000 replicates, 4-class. **Use this.** Model and estimator must match
                   Tables 2 and 3 on the same page so readers can read the cells alongside them.
  OVUL_GRID_CSV  — older grid (median of four off-the-shelf learners, single split). For reference only.

  OVUL_BOOT_DIR=results/grid_boot python3 scripts/make_grid15.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_g = os.environ.get("OVUL_GRID_CSV", "").strip()
SRC = Path(_g) if _g else (Path(__file__).resolve().parent.parent
                           / "results" / "grid_single_split" / "summary.csv")
_b = os.environ.get("OVUL_BOOT_DIR", "").strip()
BOOT = Path(_b) if _b else None

# family name -> (design letter, window number). Same numbering as Table 4.
#   1 trim 90/30   2 trim 30/30   3 02:00-06:00   4 midpoint +-2h
CELL = {"ava": ("a", 1), "ava30": ("a", 2), "ava24": ("a", 3), "ava24mid": ("a", 4),
        "huawei": ("b", 2), "huawei24": ("b", 3), "huawei24mid": ("b", 4),
        "npj": ("c", 2), "npj24": ("c", 3), "npj24mid": ("c", 4),
        "oura": ("d", 2), "oura24": ("d", 3), "oura24mid": ("d", 4),
        "24": ("e", 3), "24agg": ("f", 3)}
# Sources are already cited in the Fig 2 caption and Methods; repeating them here only costs a column.
NAME = {"a": r"\textbf{a} Smoothed nightly level",
        "b": r"\textbf{b} Day-over-day change",
        "c": r"\textbf{c} Within-night block contrast",
        "d": r"\textbf{d} Personal baseline deviation",
        "e": r"\textbf{e} Within-night sequence",
        "f": r"\textbf{f} Sequence summary"}
CARRIED = ("b", 4)          # cell carried forward to the bootstrap (pre-specified)


def load_boot():
    """Per cell, the mean ov-F1 over 1,000 replicates. Replicate 0 is the fit on the original data and is counted too.

    Intervals are not shown — with fifteen cells x four windows, adding intervals makes
    the cells unreadable. The caption states the width in one line, and the carried
    cell's interval is in Table 3.
    """
    out = {}
    for fam, (des, win) in CELL.items():
        # __keepcal is a control output that keeps the calendar columns. It must never enter the main table.
        hit = sorted(f for f in BOOT.glob(f"interp_{fam}__*.csv")
                     if "keepcal" not in f.name)
        if not hit:
            continue
        d = pd.read_csv(hit[0])
        d = d[d.arm == "combined"]
        if "OvCast_err" in d and d.OvCast_err.notna().any():
            print(f"%% {fam}: fit errors in {int(d.OvCast_err.notna().sum())} replicates",
                  file=sys.stderr)
        # Fewer than 1,000 replicates means a quick scoring. It must not pass as the final
        # value, so the cell gets a dagger and the footnote explains it.
        out.setdefault(des, {})[win] = (float(d.OvCast_ov.mean()),
                                        float(d.OvCast_ov.quantile(.025)),
                                        float(d.OvCast_ov.quantile(.975)),
                                        len(d))
    return out


def load():
    d = pd.read_csv(SRC)
    d = d[(d.arm == "combined") & (d.label == "y4_std")
          & (d.split == "sg") & (d.group == "ml")]
    out = {}
    for fam, (des, win) in CELL.items():
        s = d[d.family == "interp_" + fam]
        if len(s):
            out.setdefault(des, {})[win] = (float(s.ov_f1.median()), None, None, 0)
    return out


def selfcheck(g):
    """Are there fifteen cells? Any empty cell means the grid did not finish."""
    n = sum(len(v) for v in g.values())
    assert n == 15, f"only {n}/15 cells — the grid did not finish or a family name changed"
    assert CARRIED[1] in g.get(CARRIED[0], {}), "the carried cell is empty"


def main():
    g = load_boot() if BOOT else load()
    strict = "--allow-partial" not in sys.argv
    if strict:
        selfcheck(g)
    else:
        n = sum(len(v) for v in g.values())
        print(f"%% warning: only {n}/15 cells filled", file=sys.stderr)

    # Bold only the single best cell across all fifteen. Bolding per row gives four bolds
    # and blurs "which is best".
    best = max(x[0] for got in g.values() for x in got.values())
    rows = []
    for des in "abcdef":
        got = g.get(des, {})
        cells = []
        for w in (1, 2, 3, 4):
            if w not in got:
                cells.append("---")
                continue
            m, lo, hi, n = got[w]
            t = r"\textbf{%.3f}" % m if m == best else f"{m:.3f}"
            if lo is not None:
                # One line. adjustbox fits any overflow to the page.
                t = r"%s\,[%.3f--%.3f]" % (t, lo, hi)
            if n and n < 1000:
                t += r"$^{\dagger}$"
                print(f"%% {des}x{w}: {n} replicates — quick", file=sys.stderr)
            cells.append(t)
        rows.append(f"{NAME[des]} & " + " & ".join(cells) + " \\\\")

    allv = [x[0] for got in g.values() for x in got.values()]
    print("\n".join(rows))
    print(f"%% span {min(allv):.3f}--{max(allv):.3f} (width {max(allv)-min(allv):.3f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
