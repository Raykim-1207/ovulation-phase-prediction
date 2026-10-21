# -*- coding: utf-8 -*-
"""vis_fig2_testcohort — Redraws Fig. 2 with uncertainty at the **test-cohort participant level**.

The bars in vis_results.py come from 1,000 resamples of training participants only, so the 18 test participants are fixed
and the calendar rule gets a zero-width interval. The question is whether combined vs calendar holds up
when test participants are redrawn, so here the unit is the test hospital (B) participant.

  Point est.: pooled F1 of a single fit (pred_nights/*.csv)
  95 % CI  : participant cluster bootstrap, 10,000 replicates (all three arms, calendar included)
  P        : exact paired permutation test — swap the two arms' predictions wholesale per participant (2^18)
  Multiplicity: Benjamini–Hochberg FDR (shown), one family of 8 = 4 granularities x 2 metrics per comparison pair.
             The p_holm column keeps the earlier definition (one family of 24 tests) for reference.
             FDR follows this project's existing practice (validate_selection_bias.py).

  python3 scripts/vis_fig2_testcohort.py            # figure + CSV
  python3 scripts/vis_fig2_testcohort.py --selfcheck
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
FIG = PAPER / "fig"
# The reverse-direction (B->A) sensitivity analysis runs the same code, changing only the prediction folder and output suffix.
PRED = Path(os.environ.get("OVUL_PRED_DIR", "").strip() or PAPER / "pred_nights")
OUT = "fig2_results_testcohort" + os.environ.get("OVUL_OUT_SUFFIX", "")
N_MC = 100_000
B, SEED = 10_000, 20261020
CLASSES = (2, 3, 4, 5)
OV = {2: 0, 3: 1, 4: 2, 5: 2}                  # same as cohort.OV_CLS
ARMS = (("physio", "physio", "Physiological signals only", "#a8c4d6"),
        ("calendar", "cal", "Calendar rule only (cycle day 14)", "#84729e"),
        ("combined", "combined", "Combined", "#bf3566"))
PAIRS = (("combined", "calendar"), ("combined", "physio"), ("physio", "calendar"))
INK, MUTED, GRID, CI = "#1a1a1e", "#6b6b73", "#e3e4e7", "#464c54"

for _f in ("Arial.ttf", "Arial_Bold.ttf"):          # same font files as vis_results.py
    _p = Path("/usr/share/fonts/truetype/msttcorefonts") / _f
    if _p.is_file():
        font_manager.fontManager.addfont(str(_p))
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def f1(cm):
    """[..., nc, nc] (true x pred) -> per-class F1 [..., nc]. Same as sklearn zero_division=0."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    d = 2 * tp + (cm.sum(-2) - tp) + (cm.sum(-1) - tp)
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.)


def swap_matrix(P, rng):
    """Per-participant swap indicators [M, P] and whether exact. If P <= 20, all 2^P; otherwise N_MC random draws.

    ponytail: 2^25 (25 participants in the reverse test) does not fit in memory, so fall back to Monte Carlo. The floor of P
    becomes 1/(N_MC+1); raise N_MC if a smaller P is needed.
    """
    if P <= 20:
        return np.array(list(itertools.product((0, 1), repeat=P)), bool), True
    return rng.random((N_MC, P)) < .5, False


def perm_p(null, obs, exact):
    hits = (np.abs(null) >= abs(obs) - 1e-12).sum()
    return hits / len(null) if exact else (hits + 1) / (len(null) + 1)


def metric(name, k):
    return (lambda c: f1(c).mean(-1)) if name == "macro" else (lambda c: f1(c)[..., OV[k]])


def confusions(k):
    """Per-participant confusion matrices {arm: [P, k, k]}. Checks that the three arms refer to the same nights."""
    # Align by row position, not by merging on keys. One participant has 11 pairs of same-day records, so (pid, day)
    # collides, but the three arm files were written in the same test row order (ovulation_day_error.dump_nights).
    key = ["pid", "cyc", "day", "y_true"]
    m = None
    for arm, f, *_ in ARMS:
        d = pd.read_csv(PRED / f"y{k}_std_{f}.csv")
        if m is None:
            m = d[key].copy()
        assert d[key].equals(m[key]), f"y{k}: {arm} nights do not match the other arms"
        m[arm] = d.y_pred.to_numpy()
    pids = sorted(m.pid.unique())
    idx = m.pid.map({p: i for i, p in enumerate(pids)}).to_numpy()
    cm = {}
    for arm, *_ in ARMS:
        c = np.zeros((len(pids), k, k))
        np.add.at(c, (idx, m.y_true.to_numpy(), m[arm].to_numpy()), 1)
        cm[arm] = c
    return cm, m


def analyse():
    rng = np.random.default_rng(SEED)
    est, tests = [], []
    for k in CLASSES:
        cm, _ = confusions(k)
        P = len(next(iter(cm.values())))
        # Cluster bootstrap: resampling participants with replacement = multinomial weights on confusion matrices.
        W = rng.multinomial(P, np.full(P, 1 / P), size=B).astype(float)
        boot = {a: np.einsum("bp,pij->bij", W, c) for a, c in cm.items()}
        sw, exact = swap_matrix(P, rng)
        sw = sw.astype(float)
        for met in ("macro", "ov"):
            f = metric(met, k)
            for arm, *_ in ARMS:
                lo, hi = np.quantile(f(boot[arm]), [.025, .975])
                est.append(dict(classes=k, metric=met, arm=arm,
                                value=f(cm[arm].sum(0)), lo=lo, hi=hi))
            for a, b in PAIRS:
                A, Bc = cm[a], cm[b]
                d0 = f(A.sum(0)) - f(Bc.sum(0))
                # Exact permutation: 2^P cases swapping the two arms wholesale per person. A person's nights
                # move together, so correlation between nights is not counted as independent.
                Af, Bf = A.reshape(P, -1), Bc.reshape(P, -1)
                first = (sw @ Bf + (1 - sw) @ Af).reshape(-1, k, k)
                second = (sw @ Af + (1 - sw) @ Bf).reshape(-1, k, k)
                dd = f(first) - f(second)
                lo, hi = np.quantile(f(boot[a]) - f(boot[b]), [.025, .975])
                pa, pb = f(A), f(Bc)
                tests.append(dict(classes=k, metric=met, first=a, second=b, n_patients=P,
                                  diff=d0, lo=lo, hi=hi,
                                  p=perm_p(dd, d0, exact),
                                  patients_better=int((pa > pb).sum()),
                                  patients_worse=int((pa < pb).sum())))
    t = pd.DataFrame(tests)
    o = np.argsort(t.p.to_numpy(), kind="stable")
    t.loc[t.index[o], "p_holm"] = np.minimum(1, np.maximum.accumulate(
        t.p.to_numpy()[o] * np.arange(len(t), 0, -1)))
    # Benjamini–Hochberg, one family per comparison pair (4 granularities x 2 metrics = 8). Each bracket in the figure
    # asks about one pair, so that defines the family. Cumulative minimum of p * n / rank from the largest P down.
    for _, g in t.groupby(["first", "second"]):
        o = np.argsort(g.p.to_numpy(), kind="stable")[::-1]
        q = g.p.to_numpy()[o] * len(g) / np.arange(len(g), 0, -1)
        t.loc[g.index[o], "p_fdr"] = np.minimum(1, np.minimum.accumulate(q))
    return pd.DataFrame(est), t


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def bars(ax, est, t, met):
    x = np.arange(len(CLASSES))
    off = dict(physio=-.26, calendar=0, combined=.26)
    for arm, _, name, color in ARMS:
        e = est[(est.metric == met) & (est.arm == arm)].sort_values("classes")
        ax.bar(x + off[arm], e.value, width=.24, color=color, label=name, zorder=3)
        ax.errorbar(x + off[arm], e.value, yerr=[e.value - e.lo, e.hi - e.value],
                    fmt="none", ecolor=CI, elinewidth=.8, capsize=1.8, capthick=.8, zorder=4)
    # Brackets are combined vs each single physiological signal. Combined vs calendar is not significant at n=18 (P 0.12–0.38),
    # so instead of cluttering the figure with n.s., the text reports the difference, CI, and ovulation-day error.
    for i, k in enumerate(CLASSES):
        for lvl, other in enumerate(("physio",)):
            r = t[(t.metric == met) & (t.classes == k) & (t["first"] == "combined")
                  & (t["second"] == other)].iloc[0]
            y = 1.0 + lvl * .085
            x1, x2 = i + off[other], i + off["combined"]
            ax.plot([x1, x1, x2, x2], [y - .02, y, y, y - .02], lw=.6, color=INK, clip_on=False)
            s = stars(r.p_fdr)
            ax.text((x1 + x2) / 2, y + (.004 if s != "n.s." else .012), s, ha="center",
                    va="bottom", fontsize=7 if s != "n.s." else 6, color=INK, clip_on=False)
    ax.set(xticks=x, xticklabels=[str(k) for k in CLASSES], xlim=(-.55, len(CLASSES) - .45),
           ylim=(0, 1), yticks=np.linspace(0, 1, 6))
    ax.set_xlabel("Class")
    ax.set_ylabel("Macro-F1" if met == "macro" else "Periovulatory F1")
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def build(est, t):
    W, H = 180, 86
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    axes = [fig.add_axes([x / W, 18 / H, 70 / W, 52 / H]) for x in (16, 106)]
    for ax, met, s, x in zip(axes, ("macro", "ov"), "ab", (2, 92)):
        bars(ax, est, t, met)
        fig.text(x / W, (H - 3) / H, s, fontsize=8, weight="bold", ha="left", va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, 0), ncol=3,
               frameon=False, handlelength=1.0, columnspacing=1.4)
    return fig


def selfcheck():
    """Does confusion-matrix F1 match sklearn, and is the permutation P equal to 1 at zero difference?"""
    from sklearn.metrics import f1_score
    for k in CLASSES:
        cm, m = confusions(k)
        for arm, *_ in ARMS:
            ref = f1_score(m.y_true, m[arm], labels=list(range(k)), average=None, zero_division=0)
            assert np.allclose(f1(cm[arm].sum(0)), ref), (k, arm)
    assert np.allclose(f1(np.array([[3., 1], [0, 0]])), [6 / 7, 0])
    # BH is hand-written, so cross-check against statsmodels.
    from statsmodels.stats.multitest import multipletests
    pr = np.random.default_rng(0).random(24) ** 3
    o = np.argsort(pr, kind="stable")
    q = np.empty(24)
    q[o[::-1]] = np.minimum(1, np.minimum.accumulate(pr[o][::-1] * 24 / np.arange(24, 0, -1)))
    assert np.allclose(q, multipletests(pr, method="fdr_bh")[1])
    print("selfcheck OK")


def main():
    if "--selfcheck" in sys.argv:
        return selfcheck()
    est, t = analyse()
    est.to_csv(FIG / f"{OUT}_estimates.csv", index=False)
    t.to_csv(FIG / f"{OUT}_tests.csv", index=False)
    fig = build(est, t)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{OUT}.{ext}", dpi=600)
    print(t.round(4).to_string())
    print(f"-> {FIG / OUT}.pdf")


if __name__ == "__main__":
    main()
