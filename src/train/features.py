# -*- coding: utf-8 -*-
"""
features — OvSeq: the proposed classifier, competing on equal footing with LR/DT/SVM/LightGBM.

The four existing models classify each night independently, which discards the structure of the problem.
The peri-ovulatory phase is a **contiguous 6-day block** within one cycle. Classifying each night
separately breaks that block, and scattered false positives erode precision.

OvSeq has two parts.
  ① per-patient normalization  — removes individual baseline heart-rate/temperature levels, keeping only within-person change
  ② segment decoding           — per cycle, pick the contiguous 6-day window with the largest score sum and mark only it as ovulatory

Calendar rules are not used as candidates. Input is X, output is per-night probability/label — exactly the
same interface as the four existing models. It runs on all three input sets (calendar only / physio only / combined).
The decoder uses only the **date order** within a cycle, so it pulls in no calendar-rule information.

  python3 src/train/features.py
Output -> results/features/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import lightgbm as lgb  # noqa: E402

DATA = ROOT / "ovul_data" / "finalfinal" / "interp_huawei24mid" / "locf" / "y4_std.xlsx"
OUT = ROOT / "results" / "features"
LABEL, OV = "y4_std", 2
PHYSIO = ["bbt", "hr", "sdnn", "lfhf", "ln_sdnn", "ln_lfhf", "hr_min", "rmssd",
          "bbt_s3", "hr_s3", "dbbt_1", "dhr_1", "dbbt_3", "dhr_3",
          "bbt_slope5", "hr_slope5", "days_since_bbt_nadir", "rising_streak"]
WIN = 6            # peri-ovulatory phase is ov-4..ov+1 = 6 days
THR = np.linspace(0.05, 0.95, 91)
SEED = 42


def cal_eng(cd):
    return np.column_stack([cd, cd - 1.0, cd - 14.0, cd - 15.0, np.abs(cd - 14.0),
                            ((cd >= 12) & (cd <= 17)).astype(float),
                            np.sin(2 * np.pi * cd / 28), np.cos(2 * np.pi * cd / 28)])


def patnorm(Xp, g):
    """① per-patient z standardization."""
    Z = np.zeros_like(Xp)
    for p in np.unique(g):
        m = g == p
        mu, sd = Xp[m].mean(0), Xp[m].std(0)
        Z[m] = (Xp[m] - mu) / np.where(sd < 1e-8, 1.0, sd)
    return Z


def window_decode(score, cyc_id, day, win=WIN):
    """② per cycle, pick the contiguous win-day window with the maximum score sum.

    Inside the window is ovulatory, outside is not. Uses only date order, so independent of calendar rules.
    Returns each night's 'in-window score' — kept continuous so a threshold can be applied.
    """
    out = np.zeros(len(score))
    for c in np.unique(cyc_id):
        m = np.where(cyc_id == c)[0]
        d, s = day[m], score[m]
        best, best_sum = None, -np.inf
        for start in np.unique(d):
            inw = (d >= start) & (d < start + win)
            if not inw.any():
                continue
            tot = s[inw].sum() - 0.25 * s[~inw].sum()   # penalty for scores outside the window
            if tot > best_sum:
                best_sum, best = tot, inw
        if best is not None:
            out[m[best]] = s[best]
    return out


def ovf1(yb, pb):
    return float(f1_score(yb, pb, labels=[1], average="macro", zero_division=0))


def base_models(seed):
    return {
        "LR": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
        "DT": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=seed),
        "SVM": Pipeline([("s", StandardScaler()),
                         ("m", SVC(probability=True, random_state=seed))]),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                       min_child_samples=20, subsample=0.9, subsample_freq=1,
                                       colsample_bytree=0.8, reg_lambda=1.0,
                                       random_state=seed, n_jobs=1, verbose=-1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(DATA, engine="openpyxl")
    d.columns = [str(c).strip() for c in d.columns]
    d["day_of_cycle"] = pd.to_numeric(d["day_of_cycle"], errors="coerce")
    d = d.dropna(subset=["day_of_cycle", LABEL]).sort_values(
        ["participant_id", "cycle", "day_of_cycle"]).reset_index(drop=True)
    y = pd.to_numeric(d[LABEL]).astype(int).to_numpy() - 1
    yb = (y == OV).astype(int)
    g = d["participant_id"].astype(str).to_numpy()
    cd = d["day_of_cycle"].to_numpy(float)
    cyc = (d["participant_id"].astype(str) + "|" + d["cycle"].astype(str)).to_numpy()
    Xp = np.nan_to_num(d[PHYSIO].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    ARMS = {"combined": lambda Z: np.column_stack([Z, cal_eng(cd)]),
            "physio":   lambda Z: Z,
            "cal":      lambda Z: cal_eng(cd)}
    is_A = np.char.startswith(g.astype(str), "S")
    itr_all, ite = np.where(is_A)[0], np.where(~is_A)[0]
    print(f"train {len(itr_all)} nights / test {len(ite)} nights · {len(np.unique(cyc))} cycles\n")

    rows = []
    for arm, build_X in ARMS.items():
        for seed in range(5):
            rng = np.random.RandomState(SEED + seed)
            pats = np.unique(g[itr_all])
            val = set(rng.choice(pats, max(1, len(pats) // 4), replace=False))
            m_ = np.array([p in val for p in g[itr_all]])
            ifit, ival = itr_all[~m_], itr_all[m_]

            # four existing models — per night, raw features
            X = build_X(Xp)
            for name, mdl in base_models(SEED + seed).items():
                mdl.fit(X[ifit], y[ifit])
                col = list(mdl.classes_).index(OV) if hasattr(mdl, "classes_") \
                    else list(mdl[-1].classes_).index(OV)
                pv, pt = mdl.predict_proba(X[ival])[:, col], mdl.predict_proba(X[ite])[:, col]
                t = max((ovf1(yb[ival], (pv >= q).astype(int)), q) for q in THR)[1]
                rows.append(dict(arm=arm, seed=seed, model=name,
                                 ov_f1=ovf1(yb[ite], (pt >= t).astype(int))))

            # proposed model — ① per-patient normalization + ② segment decoding
            Xs = build_X(patnorm(Xp, g))
            m = GradientBoostingClassifier(n_estimators=120, learning_rate=0.02,
                                           max_depth=3, subsample=0.8,
                                           min_samples_leaf=20, random_state=SEED + seed)
            m.fit(Xs[ifit], y[ifit])
            c = list(m.classes_).index(OV)
            sv, st = m.predict_proba(Xs[ival])[:, c], m.predict_proba(Xs[ite])[:, c]
            dv = window_decode(sv, cyc[ival], cd[ival])
            dt_ = window_decode(st, cyc[ite], cd[ite])
            t = max((ovf1(yb[ival], (dv >= q).astype(int)), q) for q in THR)[1]
            rows.append(dict(arm=arm, seed=seed, model="OvSeq",
                             ov_f1=ovf1(yb[ite], (dt_ >= t).astype(int))))
            # variant without the decoder (① only) — to isolate contributions
            t2 = max((ovf1(yb[ival], (sv >= q).astype(int)), q) for q in THR)[1]
            rows.append(dict(arm=arm, seed=seed, model="OvSeq_nodecode",
                             ov_f1=ovf1(yb[ite], (st >= t2).astype(int))))

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "models_by_arm.csv", index=False)
    piv = r.groupby(["arm", "model"]).ov_f1.agg(["mean", "std"]).round(4)
    order = ["LR", "DT", "SVM", "LightGBM", "OvSeq_nodecode", "OvSeq"]
    print("=== A -> B, 4-class ov-F1 (5 seeds) ===")
    for arm in ("cal", "physio", "combined"):
        print(f"\n[{arm}]")
        print(piv.loc[arm].reindex(order).to_string())
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
