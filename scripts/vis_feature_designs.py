"""Draw the six feature designs using matplotlib and NumPy only.

Run: python3 vis_feature_designs.py
Synthetic traces illustrate representations, not measured data or exact
implementations of the cited smoothing/baseline algorithms.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

FIG = Path(__file__).resolve().parent / "fig"
DPI = 300
# Palette sampled from fig/experiment_axes_results.png.
INK, MUTED, ARROW = "#252b33", "#6b6b73", "#dce1e6"
BLUE, PALE, PINK = "#467c9e", "#e8edf1", "#bb3269"
WINDOW_COLORS = {1: "#8c6bb1", 2: "#2f8f6b", 3: BLUE, 4: "#e08a1e"}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Nimbus Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def text(ax, x, y, s, size=7, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", zorder=kw.pop("zorder", 6), **kw)


def arrow(ax, start, end, color=ARROW, style="-|>", lw=0.9):
    ax.annotate("", xy=end, xytext=start, zorder=5,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=7, shrinkA=1, shrinkB=1))


def axes(fig, rect, tag=None):
    ax = fig.add_axes(rect, xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    if tag:
        ax.text(-0.01, 1.01, tag, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10.5, fontweight="bold", color=INK,
                zorder=9)
    return ax


TITLES = ["Smoothed nightly level", "Day-over-day change",
          "Within-night block contrast", "Personal baseline deviation",
          "Within-night sequence", "Sequence summary"]


def output_glyph(ax, x, y, kind, ux=1, uy=1, scale=1):
    """Draw output shapes in mm; keep all 24 stack lines distinct."""
    def line(xs, ys, lw=0.7):
        ax.plot(x+np.asarray(xs)*ux*scale, y+np.asarray(ys)*uy*scale,
                color=INK, lw=lw, solid_capstyle="butt", zorder=7, clip_on=False)
    def bar(left, width):
        ax.add_patch(Rectangle((x+left*ux*scale, y-.875*uy*scale),
                              width*ux*scale, 1.75*uy*scale,
                              fc=INK, ec="none", zorder=7, clip_on=False))
    if kind == "one":
        bar(-4.25, 8.5)
    else:
        left, right = (-4.25, 4.25) if kind == "stack" else (-5.75, -.25)
        # Finer strokes are necessary to resolve 24 lines in a miniature glyph.
        for row_y in np.linspace(-2.75, 2.75, 24):
            line([left, right], [row_y, row_y], lw=.3*scale)
        if kind == "fold":
            line([.25, 2.25, .25], [2.75, 0, -2.75], lw=.7*scale)
            line([2.25, 2.75], [0, 0], lw=.7*scale)
            bar(2.75, 3)


def design(ax, i, values):
    # Independent symbol families: coloured numerals and monochrome letters.
    text(ax, 5, 100, TITLES[i], size=7.5, weight="bold", color=INK)
    ax.plot([5, 95], [79, 79], color=ARROW, lw=0.7)
    windows = [(1, 2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4), (3,), (3,)][i]
    w, h = 3.68/55.44*100, 3.68/48*100
    for window in windows:
        cx, cy = 9+(window-1)*10, 87
        selected = i == 1 and window == 4
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                     boxstyle="round,pad=0,rounding_size=0.7",
                     fc=WINDOW_COLORS[window], ec=INK if selected else "none",
                     lw=1.5 if selected else .7, zorder=8, clip_on=False))
        text(ax, cx, cy, str(window), size=8, weight="bold",
             ha="center", color="white", zorder=10)
    arrow(ax, (44, 87), (52, 87), color=MUTED, style="->", lw=.8)
    out_x = 57
    ax.add_patch(Rectangle((out_x-w/2, cy-h/2), w, h,
                          fc="white", ec=INK, lw=.9, zorder=8, clip_on=False))
    text(ax, out_x, cy, "X" if i < 4 else "Z" if i == 4 else "Y",
         size=8, weight="bold", ha="center", color=INK, zorder=10)
    x = np.linspace(8, 70, 24)
    y = 33 + 40 * values
    mean = y.mean()
    # Identical base coordinates and scale in all six cells.
    ax.plot(x, y, color=BLUE, lw=0.7, zorder=3)
    ax.plot(x, y, "o", color=BLUE, alpha=0.45, ms=2.3, zorder=3)
    if i == 0:
        smooth = np.convolve(np.pad(y, (2, 2), mode="edge"), np.ones(5)/5, "valid")
        dense_x = np.linspace(x[0], x[-1], 200)
        smooth = np.polynomial.Polynomial.fit(x, smooth, 5)(dense_x)
        ax.plot(dense_x, smooth, color=INK, lw=1.2, zorder=4)
        # Bracket spans the complete trace; its stem ends at the summary.
        ax.plot([8, 8, 70, 70], [38, 33, 33, 38], color=MUTED, lw=0.8)
        ax.plot([39, 39], [33, 22], color=PINK, lw=1.2)
        ax.plot(39, 22, "o", color=PINK, ms=5)
    elif i == 1:
        previous = y - 15 - np.linspace(0, 4, 24)
        prev_mean = previous.mean()
        ax.plot(x, previous, "o-", color=MUTED, alpha=0.4, lw=0.8, ms=2.3, zorder=1)
        # Differences connect the two actual illustrated nightly levels.
        ax.plot([8, 70], [mean]*2, color=BLUE, lw=1)
        ax.plot([8, 70], [prev_mean]*2, color=MUTED, lw=1, ls=(0, (3, 2)))
        text(ax, 8, 74, "Current", size=6)
        text(ax, 8, 25, "Previous", size=6, color=MUTED)
        ax.plot([70, 74, 74, 70], [prev_mean, prev_mean, mean, mean], color=PINK, lw=1.2)
        text(ax, 78, (prev_mean+mean)/2, "Δ", size=7)
        # Slope has explicit endpoints: preceding and current nightly means.
        ax.plot([85, 96], [prev_mean, mean], "o-", color=PINK, ms=3, lw=1.2)
        ax.plot([70, 85], [prev_mean]*2, color=ARROW, lw=0.6, ls=(0, (2, 2)))
        ax.plot([70, 96], [mean]*2, color=ARROW, lw=0.6, ls=(0, (2, 2)))
    elif i == 2:
        edges = np.linspace(x[0]-1.35, x[-1]+1.35, 5)
        # Exactly six original samples per bounded region. The pink bars
        # are illustrative block levels, not a specification of the algorithm.
        levels = y.reshape(4, 6).mean(axis=1)
        assert all(((x >= edges[j]) & (x < edges[j+1])).sum() == 6 for j in range(4))
        for j in range(4):
            ax.add_patch(Rectangle((edges[j], 37), edges[j+1]-edges[j], 33,
                                   fc=PALE, ec="none", alpha=0.45+0.55*(j%2)))
            ax.plot([edges[j], edges[j+1]], [levels[j]]*2,
                    color=INK, lw=1.2, zorder=4)
        for edge in edges:
            ax.plot([edge, edge], [37, 70], color=ARROW, lw=0.7)
        # Each bracket spans the two block levels being compared.
        for j in range(3):
            edge = edges[j+1]
            ax.plot([edge-1, edge, edge, edge+1],
                    [levels[j], levels[j], levels[j+1], levels[j+1]],
                    color=INK, lw=1.2, zorder=5)
    elif i == 3:
        baseline = 32
        ax.plot([8, 73], [baseline]*2, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.plot([8, 73], [mean]*2, color=INK, lw=1.2)
        text(ax, 8, 25, "Baseline", size=6, color=MUTED)
        ax.plot([73, 77, 77, 73], [baseline, baseline, mean, mean], color=PINK, lw=1.2)
        text(ax, 84, (baseline+mean)/2, "Δ", size=7)
        # A small history trace gives trend an explicit temporal context.
        hx, hy = [81, 88, 95], [60, 63, 69]
        ax.plot(hx, hy, "o-", color=PINK, ms=2.5, lw=1)
        ax.plot([80, 97], [57, 57], color=ARROW, lw=0.6)
    elif i == 4:
        # The whole trace, not its final point, is retained in the matrix.
        ax.plot([8, 8, 70, 70], [38, 32, 32, 38], color=BLUE, lw=0.8)
        ax.plot([39, 39, 78, 78], [32, 22, 22, 47], color=BLUE, lw=0.8)
        arrow(ax, (78, 47), (84, 47), BLUE, lw=0.8)
        # One cell per ordered sample; all 24 cells remain distinct.
        for row in range(24):
            ax.add_patch(Rectangle((86, 22+(23-row)*2.1), 6, 1.7,
                                  ec=BLUE, lw=0.7, fc=PALE))
        text(ax, 89, 77, "24", size=6, ha="center")
    else:
        lo, hi = np.argmin(y), np.argmax(y)
        ax.scatter([x[lo], 39, x[hi]], [y[lo], mean, y[hi]], s=25,
                   color=PINK, edgecolors="white", linewidths=0.5, zorder=5)
        # Each statistic is connected from its source to a same-level output.
        for sx, yy, label in ((x[hi], y[hi], "max"), (39, mean, "mean"),
                              (x[lo], y[lo], "min")):
            ax.plot([8, 82], [yy, yy], color=MUTED, lw=0.8,
                    ls=(0, (2, 2)), zorder=2)
            ax.plot(82, yy, "o", color=PINK, ms=4)
            text(ax, 86, yy, label, size=6, color=INK)


def timeline(fig):
    # Preserve both participant timelines on the common clock axis, in mm.
    ax = fig.add_axes([4.5/180, 124/170, 171/180, 36/170],
                      xlim=(0, 171), ylim=(0, 36))
    ax.axis("off")
    def clock(hour):
        return 45 + (hour-22)*10
    for hour in (26, 30):
        ax.plot([clock(hour)]*2, [6, 33.1], color=MUTED, lw=0.7,
                ls=(0, (3, 2)), zorder=1)
    anchors = {}
    for participant, start, end, top, sleep in [
            (1, 23+10/60, 31, 31.1, "23:10 – 07:00"),
            (2, 25+40/60, 33.5, 16.5, "01:40 – 09:30")]:
        # Separate night blocks while preserving every clock coordinate.
        ax.add_patch(Rectangle((0, top-10.2), 171, 12.5,
                               fc=INK, alpha=.025, ec="none", zorder=0))
        text(ax, 0, top-3, f"Participant {participant}", size=6.5, weight="bold")
        text(ax, 17, top-3, f"sleep {sleep}", size=5.5, color=MUTED)
        mid = (start+end)/2
        spans = [(start+1.5, end-.5), (start+.5, end-.5), (26, 30), (mid-2, mid+2)]
        assert all(a < b for a, b in spans)
        assert spans[2] == (26, 30) and spans[3][1]-spans[3][0] == 4
        ax.plot([clock(start), clock(end)], [top+2]*2, color=ARROW, lw=0.7)
        for number, (left, right) in enumerate(spans, 1):
            y = top-(number-1)*3
            ax.add_patch(FancyBboxPatch((clock(left), y-.9), clock(right)-clock(left), 1.8,
                         boxstyle="round,pad=0,rounding_size=0.7",
                         fc=WINDOW_COLORS[number], ec="none", zorder=3))
            text(ax, clock(left)+1.8, y, str(number), size=5.5, weight="bold",
                 color="white", ha="center")
            if number in (3, 4):
                anchors[participant, number] = (clock(left), y)
                if participant == 1:
                    text(ax, clock(right)+1.5, y,
                         "fixed clock" if number == 3 else "moves with recording",
                         size=5.5, color=WINDOW_COLORS[number])
    for number in (3, 4):
        first, second = anchors[1, number], anchors[2, number]
        ax.plot([first[0], second[0]], [first[1], second[1]],
                color=WINDOW_COLORS[number], lw=.8, zorder=2)
    ax.plot([clock(22), clock(34)], [5]*2, color=MUTED, lw=0.7)
    for hour in range(22, 35):
        ax.plot([clock(hour)]*2, [4.2, 5.8], color=ARROW, lw=0.7)
        if hour % 2 == 0:
            text(ax, clock(hour), 1.5, f"{hour % 24:02d}:00", size=5.8,
                 ha="center", color=MUTED)


def legend_blocks(fig):
    ax = fig.add_axes([4.5/180, 0, 171/180, 1], xlim=(0, 171), ylim=(0, 170))
    ax.axis("off")
    for y, height, heading in [(164, 8, "Analysis window"),
                               (116, 6, "Feature output — set by the design")]:
        ax.add_patch(Rectangle((0, y-height/2), 171, height,
                               fc=INK, alpha=.03, ec="none"))
        text(ax, 2, y, heading, size=7.5, weight="bold")
    slots = [34, 68, 102, 136]
    for x, number, name in zip(slots, (1, 2, 3, 4),
            ("90/30 min", "30/30 min", "02:00–06:00", "Midpoint ±2 h")):
        ax.add_patch(FancyBboxPatch((x, 162.16), 3.68, 3.68,
                     boxstyle="round,pad=0,rounding_size=0.4",
                     fc=WINDOW_COLORS[number], ec="none"))
        text(ax, x+1.84, 164, str(number), size=8, color="white",
             weight="bold", ha="center")
        text(ax, x+5, 164, name, size=6.5, color=WINDOW_COLORS[number])
    for x, letter, kind, name in zip(slots[1:], ("X", "Y", "Z"),
                                     ("one", "fold", "stack"),
                                     ("1 row", "24 → 1 row", "24 rows")):
        ax.add_patch(Rectangle((x, 115.16), 3.68, 3.68, fc="white", ec=INK, lw=.9))
        text(ax, x+1.84, 117, letter, size=8, weight="bold", ha="center")
        output_glyph(ax, x+10, 117, kind, scale=.65)
        text(ax, x+15, 117, name, size=6.5)
    # The brace spans only Y and Z: both require a within-night slot axis.
    # An inset label interrupts the brace so the entire annotation fits in 6 mm.
    ax.plot([103.84, 103.84, 115.8], [114.9, 114.1, 114.1], color=MUTED, lw=.7)
    ax.plot([125.9, 137.84, 137.84], [114.1, 114.1, 114.9], color=MUTED, lw=.7)
    text(ax, 120.84, 114.1, "slot axis", size=5.5, color=MUTED, ha="center")


def build_figure():
    fig = plt.figure(figsize=(180/25.4, 170/25.4), facecolor="white")
    timeline(fig)
    legend_blocks(fig)
    rng = np.random.default_rng(20260930)
    t = np.linspace(0, 1, 24)
    values = 0.48 + 0.23*np.sin(2*np.pi*(t-0.1)) + 0.12*t + rng.normal(0, 0.055, 24)
    assert values.shape == (24,) and np.all((values > 0) & (values < 1))
    for i in range(6):
        ax = axes(fig, [0.025+(i % 3)*0.321, (49-(i//3)*54)/170, 0.308, 48/170],
                  tag="abcdef"[i])
        design(ax, i, values)
    # Explicit TTF paths avoid an OTF alias in the local font cache.
    font_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for item in fig.findobj(Text):
        name = "Arial_Bold.ttf" if item.get_fontweight() == "bold" else "Arial.ttf"
        item.set_fontproperties(FontProperties(fname=font_dir / name,
                                               size=item.get_fontsize()))
    return fig


def main():
    FIG.mkdir(exist_ok=True)
    fig = build_figure()
    # Preserve the exact 180-mm page width; no tight-bbox resizing.
    for suffix in ("pdf", "svg", "png"):
        out = FIG / f"fig2_features.{suffix}"
        fig.savefig(out, dpi=DPI)
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
