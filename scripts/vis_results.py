# -*- coding: utf-8 -*-
"""
vis_results — fig/fig2_results.pdf (main text) and fig/figS_models_ablation.pdf (supplement).

Everything comes from a **1,000-replicate participant-level bootstrap**. The direction is fixed to a single external validation
— train on Hospital A (25 participants), test on Hospital B (18 participants).

  (a) **Main result** — comparison of input configurations. Physiology only < calendar rule < combined.
      The margin by which combined beats the rule and its 95 % interval are written on the same panel.

The supplementary figure figS_models_ablation holds the learner comparison and the component ablation.

Data: results/boot_full/boot.csv   (8,000 replicates, eight models)
      results/ablation/abl.csv              (8,000 replicates, component ablation)

  python3 scripts/vis_results.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
FIG = PAPER / "fig"
SRC = PAPER.parent
DEFAULT_BOOT = PAPER.parent / "results" / "boot_full" / "boot.csv"
DPI = 500

# Palette from panel d: late-luteal purple, follicular blue, periovulatory magenta.
CAL, PHYS, COMB = "#84729e", "#a8c4d6", "#bf3566"
INK, MUTED, GRID = "#1a1a1e", "#6b6b73", "#e3e4e7"
GREY = "#b9bcc4"

# Display names = same order and same wording as the paper table
MODELS = [("Yu2022", "Biphasic"), ("Goodale2019", "RF"),
          ("Kilungeja2025", "Windowed RF"), ("LightGBM", "LightGBM"),
          ("OvCast", "OvCast")]
LABELS = ["2-class", "3-class", "4-class", "5-class"]
REF_LABEL = "4-class"                      # pre-specified reference label
ABL = [("no_summary", "Within-cycle\nsummary"),
       ("no_twostage", "Two-stage\nresidual"),
       ("no_prior", "Calendar\nprior"),
       ("argmax", "Ring\ndecision")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def tag(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=INK)


def pairwise_tests(r, pt_csv=None):
    """Participant-level paired Wilcoxon. The sample is the people of the test hospital.

    An earlier version drew p from the bootstrap replicates. Three things were wrong.
      · Null centering used |draws - observed| >= |observed|, but the null is
        "the difference is 0", so the bootstrap **mean** must be subtracted to recenter.
      · Replicates resample the same 43 participants and are not independent observations. Raising
        the replicate count to 10,000 makes p 1,000x smaller — p reflects the number of
        repetitions I chose, not the data.
      · The bootstrap p has a floor of 1/1000, so with Holm over 24 tests the best is 0.024.

    Switching the sample unit to participants removes all three. n = number of test participants,
    independent of the resampling count and of the number of label merges. One participant gives one pair.

    r is unused — it is accepted only so the signature records that the bootstrap used by the bars
    and error bars and the test use different units.
    """
    src = Path(pt_csv or os.environ.get(
        "OVUL_PT_CSV", PAPER.parent / "results" / "patient_level_tests.csv"))
    assert src.is_file(), (
        f"No participant-level test results: {src}\n"
        "  Pass a participant-level test CSV with --sig-tests (output of paper/vis_fig2_testcohort.py).")
    t = pd.read_csv(src)
    t["metric"] = t["metric"]
    t["label"] = t["label"].str.replace(r"^y(\d)_std$", r"\1-class", regex=True)
    t[["first", "second"]] = t["pair"].str.split("-", expand=True)
    out = t[["metric", "label", "first", "second", "n", "median", "mean", "p"]].copy()
    out = out.rename(columns={"p": "p_raw", "median": "median_difference",
                              "mean": "mean_difference", "n": "n_patients"})
    # One family: 3 pairs x 4 class definitions x 2 metrics = 24 tests.
    order = np.argsort(out.p_raw.to_numpy(), kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(
        out.p_raw.to_numpy()[order] * np.arange(len(out), 0, -1)))
    out.loc[out.index[order], "p_holm"] = adjusted
    out["symbol"] = out.p_holm.map(lambda p: "***" if p < .001 else
                                   "**" if p < .01 else "*" if p < .05 else "ns")
    return out.reset_index(drop=True)



SIG = False          # Enabled with --sig. Default is no brackets (the text reports intervals).
# Only one bracket is drawn: **physiology vs combined**. Drawing all three pairs gives three
# brackets per group, which is crowded; the text reports the other two as gains and intervals.
_PAIR_ORDER = [("combined", "physio")]
# With participant-level tests (--sig-tests) all three pairs are drawn. Shortest bracket first:
# physio-calendar, calendar-combined, physio-combined. Pair names follow first/second in the test CSV.
_PAIR_ORDER_ALL = [("physio", "calendar"), ("combined", "calendar"), ("combined", "physio")]
# With --sig-tests the marks are set by the BH-FDR q of the participant-level tests on the test cohort
# (output of vis_fig2_testcohort.py). Bars and error bars remain the 1,000 training resamples.
SIG_TESTS = None


def paired_gain(r, met, lab, a, b):
    """Paired difference within each replicate. Same 1,000 replicates as the bar error bars."""
    s = r[r.label == lab]
    get = lambda w: s[s.arm == ("combined" if w == "calendar" else w)][
        f"cd14_{met}" if w == "calendar" else f"OvCast_{met}"].to_numpy()
    d = get(a) - get(b)
    return d.mean(), *np.quantile(d, [.025, .975])


def draw_sig(ax, r, met, x, offsets, top):
    """Stack the brackets for the three pairs in each group.

    The mark is not a p-value — * if the 95% percentile interval of the paired difference excludes 0,
    otherwise ns. A rank test over replicates would make p depend on the resampling count, so it is not used.
    """
    pairs = _PAIR_ORDER if SIG_TESTS is None else _PAIR_ORDER_ALL
    step = (top - .88) / 1.6 if SIG_TESTS is None else .11   # keep 10 pt text off the bracket above
    for xi, lab in zip(x, LABELS):
        lv = 0                                   # brackets drawn so far = level of the next bracket
        for a, b in pairs:
            _, lo, hi = paired_gain(r, met, lab, a, b)
            if SIG_TESTS is None:
                mark = "*" if (lo > 0 or hi < 0) else "ns"
            else:
                mark = sig_mark(met, lab, a, b)
                if mark == "ns":                 # only significant comparisons are drawn; the caption covers the rest.
                    continue
            y = .90 + lv * step
            lv += 1
            x1, x2 = xi + offsets[a], xi + offsets[b]
            ax.plot([x1, x1, x2, x2], [y - step*.22, y, y, y - step*.22],
                    lw=.8, color=INK, zorder=8, clip_on=False)
            ax.text((x1 + x2) / 2, y + step*.06, mark, ha="center",
                    va="bottom", fontsize=8 if mark == "ns" else 11,
                    color=INK, zorder=8, clip_on=False)


def sig_mark(met, lab, a, b):
    t = SIG_TESTS
    p = t[(t.metric == met) & (t.classes == int(lab[0])) & (t["first"] == a)
          & (t["second"] == b)].p_fdr.item()
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def sig_top():
    """Axis upper limit for --sig-tests, set by the max number of stacked significant brackets in one group."""
    n = max(sum(sig_mark(m, lab, a, b) != "ns" for a, b in _PAIR_ORDER_ALL)
            for m in ("macro", "ov") for lab in LABELS)
    return .90 + (max(n, 1) - 1) * .11 + .12     # one level gives 1.02 — same as the old --sig


def panel_a(ax, r, met="macro", legend=True, style="bar", tests=None):
    """Matched line/bar versions with discrete CIs and a dashed calendar reference."""
    x = np.arange(len(LABELS), dtype=float)
    summaries = {}
    offsets = {"physio": -.24, "calendar": 0, "combined": .24}
    for arm, color, marker, name in [
            ("physio", PHYS, "s", "Physiology signal only"),
            ("calendar", CAL, None, "Calendar rule only (cycle day 14)"),
            ("combined", COMB, "o", "Combined")]:
        means, lows, highs, distributions = [], [], [], []
        for label in LABELS:
            subset = r[(r.arm == ("combined" if arm == "calendar" else arm)) & (r.label == label)]
            values = subset[f"cd14_{met}" if arm == "calendar" else f"OvCast_{met}"]
            distributions.append(values.to_numpy())
            lo, hi = values.quantile([.025, .975])
            mean = values.mean()
            if arm == "calendar":
                lo = hi = mean
            assert 0 <= lo <= mean <= hi <= 1
            means.append(mean); lows.append(lo); highs.append(hi)
        means, lows, highs = map(np.asarray, (means, lows, highs))
        summaries[arm] = (means, lows, highs)
        pos = x + offsets[arm]
        if arm == "calendar":
            if style == "bar":
                ax.bar(pos, means, width=.20, color=color, edgecolor=color,
                       linewidth=.9, label=name, zorder=3)
            elif style == "point_ci":
                # The rule has no interval. Show it as a single short horizontal tick instead of a dot.
                for i, (xx, yy) in enumerate(zip(x, means)):
                    ax.plot([xx-.17, xx+.17], [yy, yy], color=color, lw=3.0,
                            solid_capstyle="butt", zorder=5,
                            label=name if i == 0 else None)
            elif style == "ribbon":
                ax.plot(x, means, color=color, lw=2.4, linestyle=(0, (5, 2.5)),
                        label=name, zorder=6)
            elif style == "line":
                ax.plot(x, means, color=color, lw=1.4, marker="o", ms=3.4,
                        markeredgecolor="white", markeredgewidth=.5, label=name, zorder=5)
            elif met == "ov":
                assert np.allclose(means, means[0])
                ax.plot([-.47, 3.47], [means[0]]*2, color=color, lw=2.2,
                        linestyle=(0, (5, 2.5)), label=name, zorder=7)
            else:
                for i, (xx, yy) in enumerate(zip(x, means)):
                    ax.plot([xx-.43, xx+.43], [yy, yy], color=color, lw=2.2,
                            linestyle=(0, (5, 2.5)), label=name if i == 0 else None, zorder=7)
        elif style == "point_ci":
            # Without bars the axis is not tied to 0 and can be cropped. The same interval
            # then looks three to four times wider than in the bar version — the strongest way to make intervals readable.
            ax.errorbar(pos, means, yerr=[means-lows, highs-means], fmt=marker,
                        ms=7, color=color, mec="white", mew=1.0, ecolor=color,
                        elinewidth=2.0, capsize=5.0, capthick=2.0, label=name,
                        zorder=5)
        elif style == "ribbon":
            ax.fill_between(x, lows, highs, color=color, alpha=.18, lw=0, zorder=2)
            ax.plot(x, lows, color=color, lw=.8, zorder=3)
            ax.plot(x, highs, color=color, lw=.8, zorder=3)
            ax.plot(x, means, color=color, lw=2.4, marker=marker, ms=7,
                    markeredgecolor="white", markeredgewidth=1.0, label=name,
                    zorder=5)
        else:
            if style == "bar":
                ax.bar(pos, means, width=.20, color=color, edgecolor=color,
                       linewidth=.9, label=name, zorder=3)
            else:
                stats = [dict(med=np.median(v), q1=np.quantile(v, .25),
                              q3=np.quantile(v, .75), whislo=lo, whishi=hi)
                         for v, lo, hi in zip(distributions, lows, highs)]
                ax.bxp(stats, positions=pos, widths=.29, showfliers=False,
                       patch_artist=True, manage_ticks=False, zorder=5,
                       boxprops=dict(facecolor=color, edgecolor=INK, linewidth=.6),
                       medianprops=dict(color="white", linewidth=1.1),
                       whiskerprops=dict(color=INK, linewidth=.8),
                       capprops=dict(color=INK, linewidth=.8))
                ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=color, label=name))
            if style == "bar":
                # Separate encodings by role. **Color is the group**, **black is uncertainty**.
                # Using one color in two shades makes them look like stacked quantities and blurs
                # which is the value and which is the interval. Bars are drawn light; only the interval
                # is drawn bold in dark so it stands out on the light background.
                # The interval is not pure black. With darker bars, pure black is too heavy.
                # A dark gray stays readable on the group color while drawing less attention.
                CI = "#464c54"
                ax.errorbar(pos, means, yerr=[means-lows, highs-means], fmt="none",
                            ecolor=CI, elinewidth=1.4, capsize=4.6, capthick=1.4,
                            zorder=5)
    if style == "line":
        # Colour areas connect distribution medians at nominal class positions;
        # they illustrate the gap from the deterministic calendar, not a CI.
        calendar = summaries["calendar"][0]
        for arm, color in (("physio", PHYS), ("combined", COMB)):
            medians = [r[(r.arm == arm) & (r.label == lab)][f"OvCast_{met}"].median()
                       for lab in LABELS]
            ax.fill_between(x, medians, calendar, color=color, alpha=.12, linewidth=0, zorder=2)
    # Significance brackets are not drawn. The pre-specified evidence is interval estimation and the test
    # is secondary, so it belongs to the text, not the figure. tests is kept only as an argument.
    top = (1.02 if SIG_TESTS is None else sig_top()) if SIG else .92
    ax.set(xticks=x, xticklabels=["2", "3", "4", "5"],
           xlim=(-.55, 3.55),
           **(dict(ylim=(.3, .9), yticks=np.linspace(.3, .9, 7))
              if style in ("point_ci", "ribbon")
              else dict(ylim=(0, top), yticks=np.linspace(0, .9, 10))))
    if SIG and style == "bar":
        draw_sig(ax, r, met, x, offsets, top)
    ax.set_xlabel("Class", fontsize=10, labelpad=7)
    ax.set_ylabel("Macro-F1" if met == "macro" else "Periovulatory F1", fontsize=10)
    ax.yaxis.set_label_coords(-.17, .5)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_bounds(.3 if style in ("point_ci", "ribbon") else 0, .9)
    despine(ax)


def build_main_figure(r, style="bar", tests=None):
    """180-mm clinical summary; all text 10 pt except 16-pt panel letters."""
    from matplotlib.text import Text
    from matplotlib.font_manager import FontProperties
    # With several bracket levels, enlarge the figure as much as the axis so bar mm/unit matches the original (66/1.02).
    H = 100 if SIG_TESTS is None else round(34 + sig_top() * 66 / 1.02)
    fig = plt.figure(figsize=(180/25.4, H/25.4))
    # Two equal 90-mm cells, each with a 68 × 66-mm plotting rectangle.
    axes = [fig.add_axes([left/180, 23/H, 68/180, (H - 34)/H]) for left in (16, 106)]
    panel_letters = []
    for i, (ax, met) in enumerate(zip(axes, ("macro", "ov"))):
        panel_a(ax, r, met=met, legend=False, style=style, tests=tests)
        panel_letters.append(fig.text((2.5+i*90)/180, (H - 6)/H, "ab"[i],
                                     fontsize=16, weight="bold", color=INK,
                                     ha="left", va="center"))
    handles, labels = axes[0].get_legend_handles_labels()
    # Legend order matches performance order — physiology < calendar < combined. That order is
    # this paper's conclusion, so reading the legend conveys it once more.
    order = [labels.index(n) for n in ("Physiology signal only",
                                       "Calendar rule only (cycle day 14)",
                                       "Combined")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=3, frameon=False, fontsize=10, handlelength=1.1,
               handletextpad=.5, columnspacing=.9)
    font_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for item in fig.findobj(Text):
        weight = "bold" if item.get_fontweight() == "bold" else "normal"
        item.set_fontproperties(FontProperties(
            fname=font_dir/("Arial_Bold.ttf" if weight=="bold" else "Arial.ttf"),
            size=16 if item in panel_letters else 10, weight=weight))
    # The axis upper limit is set by panel_a. The two panels only need to match — hard-coding
    # a value here would break this line every time the axis is tightened.
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert np.allclose([ax.get_position().width*180 for ax in axes], [68, 68])
    return fig



STRATA = PAPER.parent / "results" / "strata" / "strata.csv"


def regular_frame():
    """Reshape the regular-cycle stratum into the form panel_a reads.

    The strata output has the same replicate and arm structure, only different column names. Matching
    the names lets panel_a be reused as is — no second copy of the plotting code.
    """
    import os
    src = Path(os.environ.get("OVUL_STRATA_CSV", "").strip() or STRATA)
    assert src.is_file(), f"Strata output missing: {src}"
    t = pd.read_csv(src)
    return t.rename(columns={"regular_model_macro": "OvCast_macro",
                             "regular_model_ov": "OvCast_ov",
                             "regular_cal_macro": "cd14_macro",
                             "regular_cal_ov": "cd14_ov"})


def build_four_panel(r, r_reg, style="bar", tests=None):
    """Four panels: top is the full cohort, bottom is regular cycles.

    Putting the same comparison side by side in two groups makes "the intervals overlap in the full cohort but
    separate in regular cycles" readable in one figure. That contrast is the third finding of this
    paper — the gain concentrates where the calendar anchor is already reliable.
    """
    from matplotlib.text import Text
    from matplotlib.font_manager import FontProperties
    H = 190
    fig = plt.figure(figsize=(180/25.4, H/25.4))
    axes, letters = [], []
    for row, (src, tag) in enumerate(((r, "Whole cohort"),
                                      (r_reg, "Regular cycles"))):
        bot = (H - 80 - row*86) / H
        for col, met in enumerate(("macro", "ov")):
            ax = fig.add_axes([(16 + col*90)/180, bot, 68/180, 62/H])
            panel_a(ax, src, met=met, legend=False, style=style, tests=tests)
            axes.append(ax)
            k = row*2 + col
            letters.append(fig.text((2.5 + col*90)/180, bot + 70/H, "abcd"[k],
                                    fontsize=16, weight="bold", color=INK,
                                    ha="left", va="center"))
        fig.text(.5, bot + 70/H, tag, fontsize=10, color=INK,
                 ha="center", va="center")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(n) for n in ("Physiology signal only",
                                       "Calendar rule only (cycle day 14)",
                                       "Combined")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", bbox_to_anchor=(.5, .012),
               ncol=3, frameon=False, fontsize=10, handlelength=1.1,
               handletextpad=.5, columnspacing=.9)
    font_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for item in fig.findobj(Text):
        w = "bold" if item.get_fontweight() == "bold" else "normal"
        item.set_fontproperties(FontProperties(
            fname=font_dir/("Arial_Bold.ttf" if w == "bold" else "Arial.ttf"),
            size=16 if item in letters else 10, weight=w))
    # All four panels need the same y-axis so top and bottom can be compared by eye.
    assert len({ax.get_ylim() for ax in axes}) == 1
    return fig


def panel_models(ax, r):
    """Model comparison with physiology only — the one place where learner differences show."""
    s = r[(r.arm == "physio") & (r.label == REF_LABEL)]
    data = [s[f"{k}_macro"].to_numpy() for k, _ in MODELS]
    col = [GREY] * (len(MODELS) - 1) + [COMB]
    bp = ax.boxplot(data, positions=np.arange(len(MODELS)), widths=0.6,
                    patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                    zorder=3, medianprops=dict(color="white", linewidth=1.0),
                    whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7))
    for b, c in zip(bp["boxes"], col):
        b.set(facecolor=c, edgecolor=tuple(np.array(
            plt.matplotlib.colors.to_rgb(c)) * 0.62), linewidth=0.6)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([n for _, n in MODELS], rotation=38, ha="right")
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylabel("macro-F1")
    ax.set_title("Physiology signal only, four phases", fontsize=7, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)


def panel_c(ax, a):
    """Component ablation: how much does performance drop when it is switched off (reference label, two configurations)."""
    w = 0.38
    x = np.arange(len(ABL))
    for i, (arm, col, nm) in enumerate((("combined", COMB, "Combined"),
                                        ("physio", PHYS, "Physiology signal only"))):
        s = a[(a.arm == arm) & (a.label == REF_LABEL)]
        mu = [(s[f"{k}_macro"] - s.full_macro).mean() for k, _ in ABL]
        lo = [(s[f"{k}_macro"] - s.full_macro).quantile(.025) for k, _ in ABL]
        hi = [(s[f"{k}_macro"] - s.full_macro).quantile(.975) for k, _ in ABL]
        pos = x + (i - 0.5) * w
        ax.bar(pos, mu, width=w * 0.92, color=col, edgecolor="none",
               label=nm, zorder=3)
        ax.vlines(pos, lo, hi, color=INK, lw=0.6, zorder=4)
        for xp, m_, l_ in zip(pos, mu, lo):
            if abs(m_) < 1e-6:
                # In configurations without the calendar this component switches itself off. It is not missing data.
                ax.text(xp, 0.004, "0 by\ndesign", ha="center", va="bottom",
                        fontsize=5.2, color=col, linespacing=0.95)
            else:
                ax.text(xp, min(l_, m_) - 0.004, f"{m_:+.3f}", ha="center",
                        va="top", fontsize=5.4, color=INK)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in ABL], fontsize=6.2)
    ax.set_ylabel("macro-F1 change\nwhen component removed")
    lo_all = min((a[(a.arm == m) & (a.label == REF_LABEL)][f"{k}_macro"]
                  - a[(a.arm == m) & (a.label == REF_LABEL)].full_macro).quantile(.025)
                 for m in ("combined", "physio") for k, _ in ABL)
    ax.set_ylim(lo_all - 0.022, 0.030)      # keep value labels below the bars from being clipped
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", borderpad=0)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-only", action="store_true", help="Export only fig2_results")
    parser.add_argument("--sig", action="store_true",
                        help="mark * when the paired-difference interval excludes 0")
    parser.add_argument("--sig-tests", metavar="CSV",
                        help="mark * from a participant-level test CSV and export only fig2_results_bar_sig")
    args = parser.parse_args()
    global SIG, SIG_TESTS
    SIG = args.sig or bool(args.sig_tests)
    # Data sources can be overridden via environment variables, e.g. to plug in results rerun with fixed preprocessing.
    _b = os.environ.get("OVUL_BOOT_CSV", "").strip()
    _a = os.environ.get("OVUL_ABL_CSV", "").strip()
    r = pd.read_csv(_b if _b else DEFAULT_BOOT)
    if args.sig_tests:
        # Do not touch the manuscript figure (fig2_results.*). Export only the starred version.
        SIG_TESTS = pd.read_csv(args.sig_tests)
        fig = build_main_figure(r, style="bar")
        for ext in ("pdf", "svg", "png"):
            fig.savefig(FIG / f"fig2_results_bar_sig.{ext}", dpi=DPI)
        print(f"-> {FIG / 'fig2_results_bar_sig.pdf'}")
        return

    tests = pairwise_tests(r)
    tests.to_csv(FIG / "fig2_results_pairwise_stats.csv", index=False)
    for style in ("line", "bar", "point_ci", "ribbon"):
        fig = build_main_figure(r, style=style, tests=tests)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(FIG / f"fig2_results_{style}.{ext}", dpi=DPI)
            if style == "bar":
                fig.savefig(FIG / f"fig2_results.{ext}", dpi=DPI)
        plt.close(fig)

    if SIG:
        fig4 = build_four_panel(r, regular_frame(), style="bar", tests=tests)
        for ext in ("pdf", "svg", "png"):
            fig4.savefig(FIG / f"fig2_results_strata.{ext}", dpi=DPI)
        plt.close(fig4)
        print(f"→ {FIG/'fig2_results_strata.pdf'}  (four panels: full + regular cycle)")

    if args.main_only:
        print(f"Saved {FIG / 'fig2_results.pdf'} (PNG and SVG also exported)")
        return

    # Supplementary figure: learner comparison and component ablation.
    a = pd.read_csv(_a if _a else SRC / "results" / "ablation" / "abl.csv")
    fig2, (bx, cx) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                  gridspec_kw=dict(width_ratios=[1.0, 1.55],
                                                   wspace=0.34))
    panel_models(bx, r)
    panel_c(cx, a)
    for axx, t in ((bx, "a"), (cx, "b")):
        tag(axx, t)
    for ext in ("pdf", "png"):
        fig2.savefig(FIG / f"figS_models_ablation.{ext}", bbox_inches="tight",
                     pad_inches=0.03, dpi=DPI)
    plt.close(fig2)

    print(f"→ {FIG/'fig2_results.pdf'}          (main text, main result)")
    print(f"→ {FIG/'figS_models_ablation.pdf'}  (supplement, learners and ablation)")
    for lab in LABELS:
        s_ = r[(r.arm == "combined") & (r.label == lab)]
        d = s_.OvCast_macro - s_.cd14_macro
        ph = r[(r.arm == "physio") & (r.label == lab)]["OvCast_macro"].mean()
        print(f"  {lab}  calendar {s_.cd14_macro.mean():.3f}  physio {ph:.3f}  "
              f"combined {s_.OvCast_macro.mean():.3f}  gain {d.mean():+.3f} "
              f"[{d.quantile(.025):+.3f},{d.quantile(.975):+.3f}]")


if __name__ == "__main__":
    main()
