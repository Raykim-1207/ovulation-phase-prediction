# -*- coding: utf-8 -*-
"""vis_day_error_test — Tests and plots ovulation_date error at the test-cohort **participant level**.

Uses the prespecified estimation rule of day_error.py (median + 1.5 days) unchanged. Only the unit of
uncertainty changes — instead of resampling training data, test hospital (B) participants are redrawn.

Prespecified (fixed before seeing results, 2026-10-20):
  Primary   4-class, mean difference in per-cycle absolute error, combined − calendar. No multiplicity correction.
            4-class is the manuscript's reference granularity (vis_results.REF_LABEL).
  P         exact paired permutation test — flip the signs of all of a participant's cycle differences together (2^P).
  CI        participant cluster bootstrap, 10,000 replicates, 95 % percentile.
  Pairing   only cycles estimable by both arms are paired (same exclusion as the MAE in day_error.py).
            Sensitivity: cycles where combined is not estimable are filled with the calendar estimate (day 14) — what the app actually does.
  Secondary 4 granularities x {absolute error, ±1-day hit} = 8 tests, Holm. The primary test is included in this family.

  python3 scripts/vis_day_error_test.py
  python3 scripts/vis_day_error_test.py --selfcheck
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import day_error as DE                                          # noqa: E402
from vis_fig2_testcohort import (ARMS, INK, GRID, CI, FIG, PRED, B, SEED,  # noqa: E402
                                 swap_matrix, perm_p)       # shared font/color/folder settings

CLASSES, PRIMARY = (2, 3, 4, 5), 4
OUT = "fig_day_error_testcohort" + os.environ.get("OVUL_OUT_SUFFIX", "")


def cycle_errors(k):
    """Signed error (days) of the three arms per cycle. Not estimable = nan."""
    rows = {}
    for arm, f, *_ in ARMS:
        d = pd.read_csv(PRED / f"y{k}_std_{f}.csv")
        d["day"] = pd.to_datetime(d["day"])
        d = d.merge(DE.load_truth(d.family.iat[0], d.method.iat[0], d.label.iat[0]),
                    on=["pid", "cyc", "day"], how="left")
        assert d.ov_true.notna().all()
        d["offset"] = (d.day - d.ov_true).dt.days
        ov = DE.infer_ov_cls(d)
        rows[arm] = {key: DE.estimate(s, ov, "median") for key, s in d.groupby(["pid", "cyc"])}
    e = pd.DataFrame(rows)
    e.index.names = ["pid", "cyc"]
    return e.reset_index()


def paired(e, fill=False):
    """Combined/calendar pairs. If fill=True, combined non-estimable cycles are filled with calendar."""
    e = e.copy()
    if fill:
        e["combined"] = e.combined.fillna(e.calendar)
    e = e.dropna(subset=["combined", "calendar"])
    return e


def stats(e, endpoint, rng):
    """Participant-level exact permutation P and cluster bootstrap CI. Statistic = mean cycle difference."""
    a, c = e.combined.abs(), e.calendar.abs()
    d = (a - c) if endpoint == "mae" else ((a <= 1).astype(float) - (c <= 1).astype(float))
    g = e.pid.to_numpy()
    pids = np.unique(g)
    S = np.array([d[g == p].sum() for p in pids])        # per-participant sum of differences
    n = np.array([(g == p).sum() for p in pids])
    obs = S.sum() / n.sum()
    sw, exact = swap_matrix(len(pids), rng)
    p = perm_p((1 - 2 * sw.astype(float)) @ S / n.sum(), obs, exact)
    W = rng.multinomial(len(pids), np.full(len(pids), 1 / len(pids)), size=B)
    lo, hi = np.quantile((W @ S) / (W @ n), [.025, .975])
    return dict(diff=obs, lo=lo, hi=hi, p=p, n_patients=len(pids), n_cycles=int(n.sum()),
                patients_better=int((S < 0).sum() if endpoint == "mae" else (S > 0).sum()),
                patients_worse=int((S > 0).sum() if endpoint == "mae" else (S < 0).sum()))


def arm_summary(e, rng):
    """Per-arm MAE and ±1-day hit rate with cluster bootstrap CI (each arm's estimable cycles)."""
    out = []
    for arm, *_ in ARMS:
        s = e.dropna(subset=[arm])
        g, ae = s.pid.to_numpy(), s[arm].abs().to_numpy()
        pids = np.unique(g)
        n = np.array([(g == p).sum() for p in pids])
        for ep, v in (("mae", ae), ("within1", (ae <= 1).astype(float))):
            S = np.array([v[g == p].sum() for p in pids])
            W = rng.multinomial(len(pids), np.full(len(pids), 1 / len(pids)), size=B)
            lo, hi = np.quantile((W @ S) / (W @ n), [.025, .975])
            out.append(dict(arm=arm, endpoint=ep, value=S.sum() / n.sum(), lo=lo, hi=hi,
                            n_cycles=int(n.sum()), not_estimable=int(e[arm].isna().sum())))
    return out


def analyse():
    rng = np.random.default_rng(SEED)
    est, tests, errs = [], [], {}
    for k in CLASSES:
        e = cycle_errors(k)
        errs[k] = e
        est += [dict(classes=k, **r) for r in arm_summary(e, rng)]
        for fill in (False, True):
            for ep in ("mae", "within1"):
                tests.append(dict(classes=k, endpoint=ep, fill_with_calendar=fill,
                                  **stats(paired(e, fill), ep, rng)))
    t = pd.DataFrame(tests)
    for fill in (False, True):                           # family = 8 tests; sensitivity kept separate
        m = t.fill_with_calendar == fill
        p = t.loc[m, "p"].to_numpy()
        o = np.argsort(p, kind="stable")
        adj = np.empty_like(p)
        adj[o] = np.minimum(1, np.maximum.accumulate(p[o] * np.arange(len(p), 0, -1)))
        t.loc[m, "p_holm"] = adj
    return pd.DataFrame(est), t, errs


def fmt_p(p):
    return "P < 0.001" if p < .001 else f"P = {p:.3f}" if p < .1 else f"P = {p:.2f}"


def bars(ax, est, t, ep):
    x = np.arange(len(CLASSES))
    off = dict(physio=-.26, calendar=0, combined=.26)
    for arm, _, name, color in ARMS:
        s = est[(est.endpoint == ep) & (est.arm == arm)].sort_values("classes")
        k = 100 if ep == "within1" else 1
        ax.bar(x + off[arm], s.value * k, width=.24, color=color, label=name, zorder=3)
        ax.errorbar(x + off[arm], s.value * k, yerr=[(s.value - s.lo) * k, (s.hi - s.value) * k],
                    fmt="none", ecolor=CI, elinewidth=.8, capsize=1.8, capthick=.8, zorder=4)
    top = 5 if ep == "mae" else 100
    for i, kk in enumerate(CLASSES):
        r = t[(t.classes == kk) & (t.endpoint == ep) & ~t.fill_with_calendar].iloc[0]
        y = top * 1.0
        x1, x2 = i + off["calendar"], i + off["combined"]
        ax.plot([x1, x1, x2, x2], [y - top * .02, y, y, y - top * .02], lw=.6, color=INK,
                clip_on=False)
        # Exact P (uncorrected). The primary test is the single 4-class test so it is not corrected; the rest are exploratory and
        # Holm values go in the CSV and figure caption. Primary test in bold.
        ax.text((x1 + x2) / 2, y + top * .012, fmt_p(r.p).replace("P = ", "P=").replace(
                "P < ", "P<"), ha="center", va="bottom", fontsize=5.5, color=INK, clip_on=False,
                weight="bold" if kk == PRIMARY and ep == "mae" else "normal")
    ax.set(xticks=x, xticklabels=[str(c) for c in CLASSES], xlim=(-.55, len(CLASSES) - .45),
           ylim=(0, top))
    ax.set_xlabel("Class")
    ax.set_ylabel("Ovulation-day absolute error (days)" if ep == "mae"
                  else "Cycles dated within ±1 day (%)")
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def paired_patients(ax, e, t):
    """Primary test: per-participant mean absolute error, calendar → combined. One line = one person."""
    s = paired(e)
    m = s.assign(calendar=s.calendar.abs(), combined=s.combined.abs()).groupby("pid")[
        ["calendar", "combined"]].mean()
    cal_c, comb_c = ARMS[1][3], ARMS[2][3]
    for _, r in m.iterrows():
        better = r.combined < r.calendar
        ax.plot([0, 1], [r.calendar, r.combined], color=INK if better else "#9a9ca3",
                lw=.6, alpha=.8, zorder=2)
    ax.scatter(np.zeros(len(m)), m.calendar, s=12, color=cal_c, zorder=3, lw=0)
    ax.scatter(np.ones(len(m)), m.combined, s=12, color=comb_c, zorder=3, lw=0)
    r = t[(t.classes == PRIMARY) & (t.endpoint == "mae") & ~t.fill_with_calendar].iloc[0]
    ax.set(xticks=[0, 1], xticklabels=["Calendar\nrule", "Combined"], xlim=(-.35, 1.35),
           ylim=(0, max(8, np.ceil(m.values.max()))))
    ax.set_ylabel("Mean absolute error per participant (days)")
    role = "primary" if OUT.endswith("testcohort") else "sensitivity"   # B->A is sensitivity
    ax.set_title(f"{PRIMARY}-class, {role}\nΔ {r['diff']:+.2f} d ({r.lo:+.2f} to {r.hi:+.2f})"
                 f"\n{fmt_p(r.p)}", fontsize=6, color=INK, pad=4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def build(est, t, errs):
    W, H = 180, 86
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax_a = fig.add_axes([14 / W, 18 / H, 54 / W, 50 / H])
    ax_b = fig.add_axes([82 / W, 18 / H, 54 / W, 50 / H])
    ax_c = fig.add_axes([150 / W, 18 / H, 24 / W, 50 / H])
    bars(ax_a, est, t, "mae")
    bars(ax_b, est, t, "within1")
    paired_patients(ax_c, errs[PRIMARY], t)
    for s, x in zip("abc", (2, 70, 138)):
        fig.text(x / W, (H - 3) / H, s, fontsize=8, weight="bold", ha="left", va="top")
    h, l = ax_a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, 0), ncol=3, frameon=False,
               handlelength=1.0, columnspacing=1.4)
    return fig


def selfcheck():
    """Does it reproduce the MAE of day_error.analyse exactly, and is sign flipping symmetric?

    Does not compare with fig/day_error.csv — that file is stale, built from predictions before the R-peak fix.
    """
    ref = pd.DataFrame(DE.analyse(PRED))
    name = {"physio": "physio", "calendar": "cal", "combined": "combined"}
    for k in CLASSES:
        e = cycle_errors(k)
        for arm in name:
            r = ref[(ref.label == f"y{k}_std") & (ref.arm == name[arm])].iloc[0]
            assert np.isclose(e[arm].abs().mean(), r.mae_days), (k, arm)
            assert e[arm].isna().sum() == r.n_not_estimable, (k, arm)
    # If all differences are 0, P = 1; if only one person has a difference, P = 1 (symmetric under sign flips)
    toy = pd.DataFrame(dict(pid=["a", "a", "b"], combined=[1., 2, 3], calendar=[1., 2, 3]))
    assert stats(toy, "mae", np.random.default_rng(0))["p"] == 1
    print("selfcheck OK")


def main():
    if "--selfcheck" in sys.argv:
        return selfcheck()
    est, t, errs = analyse()
    est.to_csv(FIG / f"{OUT}_estimates.csv", index=False)
    t.to_csv(FIG / f"{OUT}_tests.csv", index=False)
    fig = build(est, t, errs)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{OUT}.{ext}", dpi=600)
    pd.set_option("display.width", 200)
    print(est.round(3).to_string())
    print(t.round(4).to_string())
    print(f"-> {FIG / OUT}.pdf")


if __name__ == "__main__":
    main()
