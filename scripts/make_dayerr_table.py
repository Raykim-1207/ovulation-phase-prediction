# -*- coding: utf-8 -*-
"""Supplementary table: per-cycle ovulation_date error.

The ovulation_date estimation rule is prespecified and owned solely by ovulation_day_error — the estimated ovulation day is
the median offset of nights predicted as the ovulatory phase, shifted back by the midpoint of the ovulatory window (-1.5).
This script only reads that output.

The calendar rule has no fit and does not change across resamples, so the output has only one row for it. Subtracting that single value
from all 1,000 replicates gives the paired difference — it is subtracting a constant, not finding and matching
a calendar row per replicate.

  python3 scripts/make_dayerr_table.py            > table body (1,000 training resamples)
  python3 scripts/make_dayerr_table.py --tests    > block below (test participant level, original fit)

The block below is sourced from the CSVs of vis_day_error_test.py and day_error_strong_cal.py.
4-class is the prespecified primary test; the rest are uncorrected secondary results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "results" / "dayerr" / "boot.csv"
FIG = Path(__file__).resolve().parent / "fig"
TESTS = FIG / "fig_day_error_testcohort_tests.csv"      # combined vs day 14, participant level
STRONG = FIG / "day_error_strong_cal_tests.csv"          # combined vs personal-history rule
LABELS = ["2-class", "3-class", "4-class", "5-class"]
ROWS = [("Calendar rule (cycle day 14)", "cd14 rule", "cal"),
        ("Physiology signal only", "OvCast", "physio"),
        ("Combined", "OvCast", "combined")]


def cell(u, col, dp=2, pct=False):
    v = u[col].dropna()
    f = (lambda x: f"{100*x:.0f}") if pct else (lambda x: f"{x:.{dp}f}")
    if len(v) < 2 or v.std() < 1e-9:
        return f(v.mean())
    return "%s\\,[%s--%s]" % (f(v.mean()), f(v.quantile(.025)), f(v.quantile(.975)))


def sgn(x, dp):
    """Signed number. Negatives are written as LaTeX $-$."""
    v = round(x, dp) + 0.0                 # turn -0.0 into 0
    return ("$-$" if v < 0 else "+" if v > 0 else "") + f"{abs(v):.{dp}f}"


def ci(d, lo, hi, dp, k=1):
    return rf"{sgn(k*d, dp)}\,[{sgn(k*lo, dp)} to {sgn(k*hi, dp)}]"


def tests():
    """Combined − day-14 rule. Pairs only cycles estimable by both arms."""
    t = pd.read_csv(TESTS)
    t = t[~t.fill_with_calendar]
    out = []
    for k in (2, 3, 4, 5):
        m = t[(t.classes == k) & (t.endpoint == "mae")].iloc[0]
        w = t[(t.classes == k) & (t.endpoint == "within1")].iloc[0]
        tag = r"$^{\dagger}$" if k == 4 else ""
        out.append(rf"{k}-class{tag} & {m.n_cycles} ({m.n_patients}) & "
                   rf"{ci(m['diff'], m.lo, m.hi, 2)} & {m.p:.3f} & "
                   rf"{ci(w['diff'], w.lo, w.hi, 0, 100)} & {w.p:.3f} \\")
    g = pd.read_csv(STRONG).set_index("comparator").loc["hist_cl14"]
    out += [r"\midrule",
            rf"4-class, personalized rule$^{{\ddagger}}$ & {g.n_cycles} ({g.n_patients}) & "
            rf"{ci(g['diff'], g.lo, g.hi, 2)} & {g.p:.3f} & -- & -- \\"]
    return "\n".join(out)


def main():
    import sys
    if "--tests" in sys.argv:
        print(tests())
        return
    d = pd.read_csv(SRC)
    out = []
    for lab in LABELS:
        t = d[d.label == lab]
        out.append(r"\multicolumn{4}{@{}l}{\textit{%s labels}} \\" % lab)
        for name, model, arm in ROWS:
            u = t[(t.model == model) & (t.arm == arm)]
            assert len(u), f"missing: {lab}/{model}/{arm}"
            out.append(name + " & " + " & ".join(
                [cell(u, "mae_days"), cell(u, "within1", pct=True),
                 cell(u, "within2", pct=True)]) + r" \\")
        out.append(r"\midrule")
    print("\n".join(out[:-1]))


if __name__ == "__main__":
    main()
