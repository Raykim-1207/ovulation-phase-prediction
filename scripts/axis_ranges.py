# -*- coding: utf-8 -*-
"""Recompute the contribution of the five axes from the grid.

Exactly as described in the main text — fix four axes at the reference configuration
(4-class · combined · Hospital A→Hospital B) and free one axis. Each level is collapsed
to the **median** of the remaining configurations, and the axis range is the max−min of
the level medians. Using the maximum would pick a different setting per level and break
the comparison.

Refuses to compute if coverage is uneven. If levels contain different numbers of
configurations, the medians are not fair, and using them unknowingly can make the axis
order appear to change — this has actually happened.

  python3 scripts/axis_ranges.py [--allow-partial]
"""
from __future__ import annotations
import glob, sys
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])     # repository root
M, REF_LAB, REF_SPLIT = "ov_f1", "y4_std", "sg"
LEARNERS = ["DT", "LR", "SVM", "LightGBM"]
LABELS = ["y2_std", "y3_std", "y4_std", "y5_std"]
DAY = ["locf", "linear"]
SLOT = [f"{a}_{b}" for a in DAY for b in ["locf", "linear", "mean", "forward"]]
DESIGN = {"ava": "a", "ava30": "a", "ava24": "a", "ava24mid": "a",
          "huawei": "b", "huawei24": "b", "huawei24mid": "b",
          "npj": "c", "npj24": "c", "npj24mid": "c",
          "oura": "d", "oura24": "d", "oura24mid": "d",
          "24": "e", "24agg": "f"}
SEQ = {"e", "f"}                       # designs that keep 24 values and also impute along the slot axis


def load() -> pd.DataFrame:
    fs = sum((glob.glob(f"{ROOT}/results/{p}") for p in
              ("grid_fill/*/summary.csv", "grid_par/*/summary.csv",
               "grid_all/summary.csv")), [])
    assert fs, "no grid output"
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.drop_duplicates(["arm", "family", "method", "label", "split",
                           "group", "model"])
    d["design"] = d.family.str.replace("^interp_", "", regex=True).map(DESIGN)
    assert d.design.notna().all(), "some families are not mapped"
    return d


def coverage(ml: pd.DataFrame) -> list[str]:
    """Does every design have all expected (imputation x label) combinations?"""
    bad = []
    for des, s in ml.groupby("design"):
        want = SLOT if des in SEQ else DAY
        have = {(r.method, r.label) for r in s.itertuples()}
        miss = {(m, l) for m in want for l in LABELS} - have
        if miss:
            bad.append(f"{des}: {len(miss)}/{len(want)*len(LABELS)} combinations missing")
    return bad


def main() -> None:
    d = load()
    ml = d[(d.split == REF_SPLIT) & d.model.isin(LEARNERS)]
    bad = coverage(ml)
    if bad:
        print("uneven coverage — levels have different numbers of configurations, so medians are not fair:",
              file=sys.stderr)
        for b in bad:
            print("   " + b, file=sys.stderr)
        if "--allow-partial" not in sys.argv:
            sys.exit("finish the grid and rerun (override with --allow-partial)")

    def rng(sub, axis):
        med = sub.groupby(axis)[M].median()
        return med.max() - med.min(), med.min(), med.max(), len(med)

    ref = ml[ml.label == REF_LAB]
    cal = d[(d.model == "rule_cd14") & (d.label == REF_LAB)
            & (d.split == REF_SPLIT)][M].median()
    lev = {"calendar": cal,
           "physio": ref[ref.arm == "physio"][M].median(),
           "combined": ref[ref.arm == "combined"][M].median()}
    base = ref[ref.arm == "combined"]
    out = {"input composition": (max(lev.values()) - min(lev.values()),
                                 min(lev.values()), max(lev.values()), 3),
           "interpolation": rng(base, "method"),
           "feature extraction": rng(base, "design"),
           "model": rng(base, "model"),
           "label granularity": rng(ml[ml.arm == "combined"], "label")}

    print(f"{'axis':20s}{'range':>8s}{'min':>8s}{'max':>8s}{'levels':>6s}")
    for k, (r, lo, hi, n) in sorted(out.items(), key=lambda x: -x[1][0]):
        print(f"{k:20s}{r:8.3f}{lo:8.3f}{hi:8.3f}{n:6d}")

    print("\nimputation range by design")
    for des in "abcdef":
        s = base[base.design == des]
        if not len(s):
            continue
        m = s.groupby("method")[M].median()
        tag = "keeps 24" if des in SEQ else "1 row/night"
        print(f"  {des}  {tag:10s} range {m.max()-m.min():.3f}  ({len(m)} imputation methods)")


def selfcheck() -> None:
    """Is the reference configuration actually in the grid? Otherwise every number is an empty median."""
    ml = load()
    s = ml[(ml.split == REF_SPLIT) & (ml.label == REF_LAB)
           & (ml.arm == "combined") & ml.model.isin(LEARNERS)]
    assert len(s) > 0, "reference configuration (4-class·combined·sg) is not in the grid"
    assert s.design.nunique() == 6, f"only {s.design.nunique()} designs"
    print("selfcheck OK")


if __name__ == "__main__":
    (selfcheck if "--selfcheck" in sys.argv else main)()
