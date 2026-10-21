# -*- coding: utf-8 -*-
"""Supplementary table: all models x four granularities x two input sets x two metrics.

Numbers come only from three outputs. Nothing is copied by hand.
  results/boot_full/boot.csv   4 prior + LightGBM + OvCast + calendar
  results/boot_sota/sota.csv   TabPFN v2 · TabICL · TabM · Mamba
  results/boot_sota2/sota.csv  KAN · RealMLP · ExcelFormer

All three draw replicate b with the same seed (SEED+b), so the same b is the same training sample.
Hence an inner merge does not mix rows.

**Two things are excluded from the comparison, for different reasons.**
  · Maijala2019 — a rule that reconstructs phase from cycle day, so the 'physiology only' cell is undefined.
  · TabPFN v2 · TabICL — foundation models **pretrained** on millions of synthetic tables.
    They bring in outside information, breaking the premise "same feature matrix, only the learning algorithm differs".
  Both are excluded by nature, not by performance. Their numbers are kept in the Note.

  python3 scripts/make_suppl_models.py macro   > macro-F1 table body
  python3 scripts/make_suppl_models.py ov      > ovulatory-phase F1 table body
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
LABELS = ["2-class", "3-class", "4-class", "5-class"]
# Rows are grouped by family: prior studies -> off-the-shelf -> 2024~25 -> proposed.
ROWS = [
    ("Yu2022", r"Biphasic switch-point\,\cite{yu2022tracking}"),
    ("Goodale2019", r"Random forest\,\cite{goodale2019wearable}"),
    ("Kilungeja2025", r"Windowed random forest\,\cite{kilungeja2025machine}"),
    ("LightGBM", r"LightGBM\,\cite{ke2017lightgbm}"),
    ("TabM", r"TabM\,\cite{gorishniy2025tabm}"),
    ("Mamba", r"Mamba\,\cite{gu2024mamba}"),
    ("KAN", r"Kolmogorov--Arnold network\,\cite{liu2024kan}"),
    ("RealMLP", r"RealMLP\,\cite{holzmuller2024better}"),
    ("ExcelFormer", r"ExcelFormer\,\cite{chen2024excelformer}"),
    ("OvCast", r"\textbf{OvCast} (this study)"),
]


def load():
    f = pd.read_csv(R / "boot_full" / "boot.csv")
    a = pd.read_csv(R / "boot_sota" / "sota.csv")
    b = pd.read_csv(R / "boot_sota2" / "sota.csv")
    key = ["boot", "label", "arm"]
    d = f.merge(a, on=key, how="inner").merge(b, on=key, how="inner")
    # Precondition: replicates must align across cells so means and intervals share the same basis.
    n = d.groupby(key[1:]).size()
    assert n.nunique() == 1, f"replicate counts differ across cells:\n{n}"
    need = [f"{k}_{m}" for k, _ in ROWS for m in ("macro", "ov")]
    miss = [c for c in need if c not in d.columns]
    assert not miss, f"missing columns: {miss}"
    return d


def cell(s, key, met):
    v = s[f"{key}_{met}"].dropna()
    if len(v) < 2 or v.std() < 1e-9:
        return f"{v.mean():.3f}"
    return "%.3f\\,[%.3f--%.3f]" % (v.mean(), v.quantile(.025), v.quantile(.975))


def body(d, met):
    out = []
    for lab in LABELS:
        s = d[d.label == lab]
        out.append(r"\multicolumn{4}{@{}l}{\textit{%s labels}} \\" % lab)
        # The calendar does not vary by method. Write it once per block, vertically.
        cal = r"\multirow{%d}{*}{%s}" % (
            len(ROWS), cell(s[s.arm == "combined"], "cd14", met))
        for i, (key, name) in enumerate(ROWS):
            cells = [cal if i == 0 else ""]
            cells += [cell(s[s.arm == arm], key, met)
                      for arm in ("physio", "combined")]
            out.append(name + " & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1])


if __name__ == "__main__":
    met = sys.argv[1] if len(sys.argv) > 1 else "macro"
    assert met in ("macro", "ov"), "macro or ov"
    print(body(load(), met))
