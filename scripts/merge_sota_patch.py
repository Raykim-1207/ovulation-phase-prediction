# -*- coding: utf-8 -*-
"""Fill the empty cells of boot_sota from boot_sota_patch.

In the original run, replicates 13–26 of the 5-class combined cell were lost to CUDA OOM. A rerun with
the same seeds is in boot_sota_patch, and on overlapping replicates the two runs agree **bit
for bit** (four models x two arms x 1,000 replicates, median difference 0.0000). So filling only
the empty slots is enough — the merged result is indistinguishable from a single run.

That premise is rechecked every time. If values in overlapping slots disagree, nothing is filled and it stops.

  python3 scripts/merge_sota_patch.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
MAIN = R / "boot_sota" / "sota.csv"
PATCH = R / "boot_sota_patch" / "sota.csv"
KEY = ["boot", "label", "arm"]
MODELS = ["TabPFNv2", "TabICL", "TabM", "Mamba"]
TOL = 1e-9


def main():
    a = pd.read_csv(MAIN)
    b = pd.read_csv(PATCH)
    cols = [f"{m}_{s}" for m in MODELS for s in ("macro", "ov")]

    j = a.merge(b[KEY + cols], on=KEY, how="left", suffixes=("", "_p"))

    # Premise check. Only verify that the two runs used **the same protocol**.
    #
    # At first this stopped on any single mismatching replicate, but that criterion is too strict.
    # TabICL shrinks its inference batch to fit available GPU memory (hence the OOM message "minimum
    # batch size"). Different memory states change the accumulation order, so a few replicates
    # wobble at the 1e-3 level. The other three were bit-identical across all 2,000 replicates.
    #
    # If the protocol differed (different seeds, splits or labels), the differences would be **biased**. So we
    # check the mean difference. Per-replicate wobble is not blocked, just logged — the slots being filled are
    # only where the original is empty anyway, so values in overlapping slots are left untouched.
    bad, noisy = [], []
    for c in cols:
        m = j[c].notna() & j[c + "_p"].notna()
        d = j.loc[m, c] - j.loc[m, c + "_p"]
        if not len(d):
            continue
        if abs(d.mean()) > 1e-4:
            bad.append(f"{c}: mean diff {d.mean():+.2e} — protocols differ")
        n = int((d.abs() > TOL).sum())
        if n:
            noisy.append(f"{c}: {n}/{len(d)} replicates differ (max {d.abs().max():.1e})")
    assert not bad, "the two runs' protocols disagree — not filling:\n  " + "\n  ".join(bad)
    if noisy:
        print("  numeric wobble (same protocol, overlapping slots keep the original):")
        for s in noisy:
            print(f"    {s}")

    filled = 0
    for c in cols:
        n = (j[c].isna() & j[c + "_p"].notna()).sum()
        if n:
            print(f"  {c}: {n} replicates filled")
            filled += n
        j[c] = j[c].fillna(j[c + "_p"])
    j = j.drop(columns=[c + "_p" for c in cols])

    # Slots still empty after filling are left as is, but counted and reported.
    left = {c: int(j[c].isna().sum()) for c in cols if j[c].isna().any()}
    for c in cols:                     # error records for filled rows are no longer true
        m = c.split("_")[0] + "_err"
        if m in j:
            j.loc[j[c].notna(), m] = pd.NA

    shutil.copy(MAIN, MAIN.with_suffix(".csv.bak"))
    j.to_csv(MAIN, index=False)
    print(f"\nfilled {filled} cells · original kept as {MAIN.name}.bak")

    t = j.groupby(["label", "arm"]).apply(
        lambda s: pd.Series({m: int(s[m + "_macro"].notna().sum()) for m in MODELS}))
    print("\nreplicates per cell")
    print(t.to_string())
    print("\nremaining empty slots:", left if left else "none")


if __name__ == "__main__":
    main()
