# -*- coding: utf-8 -*-
"""day_error_strong_cal — combined vs strong calendar rule, 4-class ovulation_date error.

Analysis plan fixed before running.
  Primary sensitivity  combined vs hist_cl14 (mean of previous cycle lengths − 14). Descriptive  combined vs cd14, cd15.
  Cycle set     cycles where hist_cl14 is defined from history (excluding day-15 substitution) and combined is estimable.
  Test          same as vis_day_error_test.stats — participant-level exact permutation + cluster bootstrap.

  python3 scripts/day_error_strong_cal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
sys.path[:0] = [str(PAPER), str(PAPER.parent / "src")]
import day_error as DE                                           # noqa: E402
import strict_cal_rule as SC                                     # noqa: E402
from vis_day_error_test import cycle_errors, stats               # noqa: E402
from vis_fig2_testcohort import PRED, FIG, SEED                  # noqa: E402

RULES = ("cd14", "cd15", "hist_cl14")
OLD = PAPER / "fig_etc" / "day_error.csv"       # before the R-wave fix


def rule_errors():
    """Per-cycle signed error of the three rules, and whether hist_cl14 is defined from history."""
    nights = pd.read_csv(PRED / "y4_std_cal.csv")
    nights["day"] = pd.to_datetime(nights["day"])
    src = pd.read_excel(DE.DATA / "interp_huawei24mid" / "locf" / "y4_std.xlsx", engine="openpyxl")
    meta = src[["participant_id", "cycle", "measure_date", "menses_start", "next_menses_start", "ovulation_date"]].copy()
    meta.columns = ["pid", "cyc", "day", "msd", "nmsd", "ov_true"]
    for c in ("day", "msd", "nmsd", "ov_true"):
        meta[c] = pd.to_datetime(meta[c])
    meta = meta.drop_duplicates(["pid", "cyc", "day"])
    # History uses only past cycles among the participant's analyzed cycles (never the current one).
    cyc_meta = meta.drop_duplicates(["pid", "cyc"])
    hist = SC.build_hist_cl_map(cyc_meta.pid.to_numpy(), cyc_meta.cyc.to_numpy(),
                                cyc_meta.msd.to_numpy(), cyc_meta.nmsd.to_numpy())
    d = nights[["pid", "cyc", "day", "y_true"]].merge(meta, on=["pid", "cyc", "day"], how="left")
    assert d.msd.notna().all() and len(d) == len(nights)
    d["offset"] = (d.day - d.ov_true).dt.days
    ov = DE.infer_ov_cls(d.assign(y_true=d.y_true))
    out = {}
    for rule in RULES:
        keep, pred = SC.predict_strict_cal(rule, "y4_std", d.pid.to_numpy(), d.cyc.to_numpy(),
                                           d.msd.to_numpy(), d.nmsd.to_numpy(),
                                           d.day.to_numpy(), hist)
        assert keep.all(), f"{rule}: nights without a rule prediction {(~keep).sum()}"
        s = d.assign(y_pred=pred)
        out[rule] = {k: DE.estimate(g, ov, "median") for k, g in s.groupby(["pid", "cyc"])}
    e = pd.DataFrame(out)
    e.index.names = ["pid", "cyc"]
    e = e.reset_index()
    e["has_history"] = [(p, int(c)) in hist for p, c in zip(e.pid, e.cyc)]
    return e


def main():
    rng = np.random.default_rng(SEED)
    r = rule_errors()
    base = cycle_errors(4)[["pid", "cyc", "calendar", "combined"]]
    e = base.merge(r, on=["pid", "cyc"], how="outer", validate="one_to_one")
    assert len(e) == len(base)

    # Self-check: cd14 from the same pipeline matches the calendar-arm error for every cycle.
    assert np.allclose(e.cd14, e.calendar, equal_nan=True), "cd14 reproduction failed"
    # Precheck: does the current night set reproduce the three rules' MAE from before the R-wave fix?
    # Only when that output exists (it is not in the public repository).
    if OLD.is_file():
        old = pd.read_csv(OLD)
        print("cycle set check (before R-wave fix vs now, all 35 cycles):")
        for rule in RULES:
            o = old[old.model == f"rule_{rule}"].iloc[0]
            now = e[rule].abs().mean()
            print(f"  {rule:9s} before {o.mae_days:.6f} (n={o.n_cycles})  now {now:.6f} "
                  f"(n={e[rule].notna().sum()})  {'same' if np.isclose(o.mae_days, now) else 'different'}")

    excl = dict(no_history=int((~e.has_history).sum()),
                combined_not_estimable=int(e.combined.isna().sum()))
    s = e[e.has_history & e.combined.notna() & e[list(RULES)].notna().all(axis=1)]
    excl["both"] = int((~e.has_history & e.combined.isna()).sum())
    print(f"\ncycle {len(e)}, analyzed {len(s)}: excluded {excl}")

    rows = []
    for rule in RULES:
        st = stats(s.assign(calendar=s[rule]), "mae", rng)
        rows.append(dict(comparator=rule, role="sensitivity" if rule == "hist_cl14" else "descriptive",
                         combined_mae=s.combined.abs().mean(), rule_mae=s[rule].abs().mean(), **st))
    t = pd.DataFrame(rows)
    t.to_csv(FIG / "day_error_strong_cal_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
