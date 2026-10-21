# -*- coding: utf-8 -*-
"""strata_testcohort — test combined vs calendar per cycle-regularity stratum, with test participants as the unit.

The manuscript (main3.tex 'Performance stratified by cycle regularity') states that combined
beats calendar in regular cycles with "every interval excluding zero". Those intervals resample
the training participants (the 18 test participants fixed). Here the test participants are
re-examined as the unit, using the same method as vis_fig2_testcohort.

  stratum       same bounds as strata.strata_of (measured cycle length 27–32 regular, 24–26·33–35 mildly irregular)
  P             exact paired permutation at participant level (only participants with nights in that stratum)
  CI            participant-cluster bootstrap, 10,000 replicates
  multiplicity  BH-FDR per stratum, family = 4 granularities x 2 metrics

  python3 scripts/strata_testcohort.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
sys.path[:0] = [str(PAPER), str(PAPER.parent / "src" / "train")]
import day_error as DE                                                 # noqa: E402
import vis_fig2_testcohort as V                                        # noqa: E402
from strata import strata_of                                   # noqa: E402

STRATA_CSV = PAPER.parent / "results" / "strata" / "strata.csv"


def stratum(m, k):
    t = pd.read_excel(DE.DATA / "interp_huawei24mid" / "locf" / f"y{k}_std.xlsx",
                      engine="openpyxl")[["participant_id", "cycle", "menses_start", "next_menses_start"]]
    t.columns = ["pid", "cyc", "msd", "nmsd"]
    t = t.drop_duplicates(["pid", "cyc"])
    t["len"] = (pd.to_datetime(t.nmsd) - pd.to_datetime(t.msd)).dt.days.astype(float)
    s = m[["pid", "cyc"]].merge(t, on=["pid", "cyc"], how="left")
    assert len(s) == len(m) and s.len.notna().all()
    return strata_of(s.len.to_numpy())


def analyse():
    rng = np.random.default_rng(V.SEED)
    ref = pd.read_csv(STRATA_CSV)
    rows = []
    for k in V.CLASSES:
        _, m = V.confusions(k)
        st = stratum(m, k)
        for name in ("regular", "mildly_irregular"):
            s = m[st == name]
            pids = sorted(s.pid.unique())
            idx = s.pid.map({p: i for i, p in enumerate(pids)}).to_numpy()
            cm = {}
            for arm in ("calendar", "combined"):
                c = np.zeros((len(pids), k, k))
                np.add.at(c, (idx, s.y_true.to_numpy(), s[arm].to_numpy()), 1)
                cm[arm] = c
            P = len(pids)
            sw, exact = V.swap_matrix(P, rng)
            sw = sw.astype(float)
            W = rng.multinomial(P, np.full(P, 1 / P), size=V.B).astype(float)
            A, C = cm["combined"], cm["calendar"]
            r0 = ref[(ref.boot == 0) & (ref.label == f"{k}-class") & (ref.arm == "combined")].iloc[0]
            assert r0[f"{name}_n"] == len(s), (k, name, r0[f"{name}_n"], len(s))
            for met in ("macro", "ov"):
                f = V.metric(met, k)
                a0, c0 = f(A.sum(0)), f(C.sum(0))
                # Self-check: point estimates must match the original fit (boot 0) of the manuscript's stratified output.
                assert np.isclose(a0, r0[f"{name}_model_{met}"]), (k, name, met)
                assert np.isclose(c0, r0[f"{name}_cal_{met}"]), (k, name, met)
                Af, Cf = A.reshape(P, -1), C.reshape(P, -1)
                dd = (f((sw @ Cf + (1 - sw) @ Af).reshape(-1, k, k))
                      - f((sw @ Af + (1 - sw) @ Cf).reshape(-1, k, k)))
                lo, hi = np.quantile(f(np.einsum("bp,pij->bij", W, A))
                                     - f(np.einsum("bp,pij->bij", W, C)), [.025, .975])
                pa, pc = f(A), f(C)
                rows.append(dict(stratum=name, classes=k, metric=met, n_nights=len(s),
                                 n_patients=P, combined=a0, calendar=c0, diff=a0 - c0,
                                 lo=lo, hi=hi, p=V.perm_p(dd, a0 - c0, exact),
                                 patients_better=int((pa > pc).sum()),
                                 patients_worse=int((pa < pc).sum())))
    t = pd.DataFrame(rows)
    for _, g in t.groupby("stratum"):
        o = np.argsort(g.p.to_numpy(), kind="stable")[::-1]
        q = g.p.to_numpy()[o] * len(g) / np.arange(len(g), 0, -1)
        t.loc[g.index[o], "p_fdr"] = np.minimum(1, np.minimum.accumulate(q))
    return t


def main():
    t = analyse()
    t.to_csv(V.FIG / "strata_testcohort_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
