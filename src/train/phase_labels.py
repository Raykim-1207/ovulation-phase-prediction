# -*- coding: utf-8 -*-
"""
phase_labels — participant-level bootstrap of the ovulation_date residual regression model (sg fixed).

ov-F1 is a function of the ovulation_date estimation error. Ceiling curve measured directly on the test set:
    error 0 d -> 1.000 · 1 d -> 0.888 · 1.5 d -> 0.812
    2 d -> 0.753 · 2.5 d -> 0.692 · 3 d -> 0.618
The cd15 rule has a 2.39-day error, so it is capped at 0.664. So instead of classifying the night phase,
we predict **how many days to shift the ovulation date earlier or later**.

Features are OvSeq's causal cumulative history + past mean cycle length + day_of_cycle.
  Regress delta = actual ovulation_date - 15 and recover the night labels from ovulation_date_hat.

  python3 src/train/phase_labels.py --n-boot 1000 --workers 20
Output -> results/phase_labels/
"""
from __future__ import annotations

import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.ovseq import causal_history, build as build_ovseq  # noqa: E402
from features import PHYSIO, cal_eng, base_models, THR, SEED  # noqa: E402
from strict_cal_rule import build_hist_cl_map  # noqa: E402

FINAL = ROOT / "ovul_data" / "finalfinal" / "interp_huawei24mid" / "locf"
OUT = ROOT / "results" / "phase_labels"
ANCHOR, CLIP = 15.0, 6.0
LABELS = ["y2_std", "y3_std", "y4_std", "y5_std"]
OV_CLS = {"y2_std": 0, "y3_std": 1, "y4_std": 2, "y5_std": 2}
_S = {}


def phase5(cd, ov):
    y = np.full(len(cd), 5)
    y[cd <= 5] = 1
    y[(cd >= 6) & (cd <= ov - 5)] = 2
    y[(cd >= ov - 4) & (cd <= ov + 1)] = 3
    y[(cd >= ov + 2) & (cd <= ov + 7)] = 4
    return y


def collapse(p5, lab):
    if lab == "y5_std": return p5 - 1
    if lab == "y4_std": return np.where(p5 >= 4, 3, p5 - 1)
    if lab == "y3_std": return np.where(p5 == 1, 0, np.where(p5 == 3, 1, 2))
    return np.where(p5 == 3, 0, 1)


def f1b(yb, pb):
    return float(f1_score(yb, pb, zero_division=0))


def prep():
    d = pd.read_excel(FINAL / "y4_std.xlsx", engine="openpyxl")
    d.columns = [str(c).strip() for c in d.columns]
    d["day_of_cycle"] = pd.to_numeric(d["day_of_cycle"], errors="coerce")
    d = d.dropna(subset=["day_of_cycle", "y4_std"]).sort_values(
        ["participant_id", "cycle", "day_of_cycle"]).reset_index(drop=True)
    g = d["participant_id"].astype(str).to_numpy()
    cd = d["day_of_cycle"].to_numpy(float)
    ovt = ((pd.to_datetime(d["ovulation_date"]) - pd.to_datetime(d["menses_start"])).dt.days + 1).to_numpy(float)
    H, _ = causal_history(d)
    hist = build_hist_cl_map(g, d["cycle"].to_numpy(), d["menses_start"].to_numpy(),
                             d["next_menses_start"].to_numpy())
    hcl = np.array([hist.get((str(p), int(c)), 28.0) for p, c in
                    zip(g, pd.to_numeric(d["cycle"], errors="coerce").fillna(0))])
    Xd = np.column_stack([H, hcl, cd])
    Xp = np.nan_to_num(d[PHYSIO].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    Xn = np.column_stack([Xp, H, cal_eng(cd)])
    Xb = np.column_stack([Xp, cal_eng(cd)])
    return dict(g=g, cd=cd, ovt=ovt, Xd=Xd, Xn=Xn, Xb=Xb,
                delta=np.clip(ovt - ANCHOR, -CLIP, CLIP),
                is_A=np.char.startswith(g.astype(str), "S"))


def init():
    _S["d"] = prep()


def one(args):
    b, lab = args
    D = _S["d"]
    g, cd, ovt, delta = D["g"], D["cd"], D["ovt"], D["delta"]
    ovc = OV_CLS[lab]
    y = collapse(phase5(cd, ovt), lab)
    yb = (y == ovc).astype(int)
    ite = np.where(~D["is_A"])[0]
    pats = np.unique(g[D["is_A"]])
    rng = np.random.RandomState(SEED + b)
    draw = pats if b == 0 else rng.choice(pats, len(pats), replace=True)
    itr = np.concatenate([np.where(g == p)[0] for p in draw])
    tr_pat = np.concatenate([[p] * int((g == p).sum()) for p in draw])
    uq = np.unique(draw)
    val = set(rng.choice(uq, max(1, len(uq) // 4), replace=False))
    m_ = np.array([p in val for p in tr_pat])
    ifit, ival = itr[~m_], itr[m_]
    if len(ival) < 20:
        return None
    out = dict(boot=b, label=lab,
               cd14=f1b(yb[ite], (collapse(phase5(cd[ite], np.full(len(ite), 14.0)), lab) == ovc).astype(int)),
               cd15=f1b(yb[ite], (collapse(phase5(cd[ite], np.full(len(ite), 15.0)), lab) == ovc).astype(int)),
               mae_rule=float(np.abs(ovt[ite] - 15.0).mean()))
    try:
        m = GradientBoostingRegressor(n_estimators=120, learning_rate=0.03, max_depth=2,
                                      subsample=0.8, min_samples_leaf=30, random_state=SEED + b)
        m.fit(D["Xd"][ifit], delta[ifit])
        dh = np.clip(m.predict(D["Xd"][ite]), -CLIP, CLIP)
        ovh = np.round(ANCHOR + dh)
        out["OvDay"] = f1b(yb[ite], (collapse(phase5(cd[ite], ovh), lab) == ovc).astype(int))
        out["mae_ovday"] = float(np.abs(ovt[ite] - (ANCHOR + dh)).mean())
        for nm, X in (("LightGBM", D["Xb"]), ("OvSeq", D["Xn"])):
            mdl = build_ovseq(SEED + b) if nm == "OvSeq" else base_models(SEED + b)["LightGBM"]
            mdl.fit(X[ifit], y[ifit])
            c = list(mdl.classes_).index(ovc)
            pv, pt = mdl.predict_proba(X[ival])[:, c], mdl.predict_proba(X[ite])[:, c]
            t = max((f1b(yb[ival], (pv >= q).astype(int)), q) for q in THR)[1]
            out[nm] = f1b(yb[ite], (pt >= t).astype(int))
    except Exception as e:
        out["error"] = str(e)[:60]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    for lab in LABELS:
        jobs = [(b, lab) for b in range(a.n_boot)]
        with ProcessPoolExecutor(max_workers=a.workers, initializer=init) as ex:
            got = [r for r in ex.map(one, jobs, chunksize=8) if r]
        rows += got
        df = pd.DataFrame(got)
        print(f"[{time.time()-t0:6.0f}s] {lab} n={len(df):4d}  cd15={df.cd15.mean():.4f} "
              f"LightGBM={df.LightGBM.mean():.4f} OvSeq={df.OvSeq.mean():.4f} "
              f"OvDay={df.OvDay.mean():.4f}  MAE {df.mae_rule.mean():.2f}->{df.mae_ovday.mean():.2f}",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT / "boot_replicates.csv", index=False)
    r = pd.DataFrame(rows)
    agg = []
    for lab, s in r.groupby("label"):
        for m in ("cd14", "cd15", "LightGBM", "OvSeq", "OvDay"):
            d_ = s["OvDay"] - s[m]
            agg.append(dict(label=f"{lab[1]}-class", model=m, mean=round(s[m].mean(), 3),
                            lo=round(s[m].quantile(.025), 3), hi=round(s[m].quantile(.975), 3),
                            diff=round(d_.mean(), 3), diff_lo=round(d_.quantile(.025), 3),
                            diff_hi=round(d_.quantile(.975), 3),
                            wins=round(100 * (d_ > 0).mean(), 1)))
    t = pd.DataFrame(agg)
    t.to_csv(OUT / "boot_summary.csv", index=False)
    print("\n=== Bootstrap 1,000 replicates · ov-F1 ===")
    print(t.pivot(index="label", columns="model", values="mean")
          [["cd14", "cd15", "LightGBM", "OvSeq", "OvDay"]].to_string())
    print("\n=== OvDay - win rate against each model (%) ===")
    print(t[t.model != "OvDay"].pivot(index="label", columns="model", values="wins")
          [["cd14", "cd15", "LightGBM", "OvSeq"]].to_string())
    print(f"\novulation_date error: rule {r.mae_rule.mean():.2f} d -> OvDay {r.mae_ovday.mean():.2f} d")
    print(f"\n-> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
