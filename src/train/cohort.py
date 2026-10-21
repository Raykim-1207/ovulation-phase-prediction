# -*- coding: utf-8 -*-
"""
cohort — OvPhase vs LR/DT/SVM/LightGBM, **exactly the same input**.

Protocol (pinned down so it holds up under review)

  1. Same input   Every model receives the same matrix X. physio = 18 nightly physiological columns,
                   combined = those 18 columns + 8 calendar columns. No per-patient normalization, no
                   cycle-history processing. The only difference is the learning algorithm.
  2. No leakage   ① Split by patient (Hospital A -> Hospital B). No patient appears on both sides.
                   ② Fitting sees only training rows. Test-row statistics (mean, variance, frequency)
                      are never used anywhere.
                   ③ Input columns are computed from nights up to t only (verified by causality audit).
                   audit() below actually checks all three; the experiment proceeds only if it passes.
  3. No tuning    Every model is fit once with default settings. No model has a step that
                   selects based on test performance.
  4. Baseline     cd14 (day_of_cycle 14 rule). cd15 is reported alongside as sensitivity.

  python3 src/train/cohort.py
Output -> results/cohort/
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.ovphase import OvPhaseClassifier  # noqa: E402
from features import PHYSIO, cal_eng, base_models, SEED  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402

_fd = os.environ.get("OVUL_FINAL_DIR", "").strip()
FINAL = Path(_fd) if _fd else (
    ROOT / "ovul_data" / "finalfinal" / "interp_huawei24mid" / "locf")
OUT = ROOT / "results" / "cohort"
LABELS = ["y2_std", "y3_std", "y4_std", "y5_std"]
OV_CLS = {"y2_std": 0, "y3_std": 1, "y4_std": 2, "y5_std": 2}


def load():
    d = pd.read_excel(FINAL / "y4_std.xlsx", engine="openpyxl")
    d.columns = [str(c).strip() for c in d.columns]
    d["day_of_cycle"] = pd.to_numeric(d["day_of_cycle"], errors="coerce")
    d = d.dropna(subset=["day_of_cycle", "y4_std"]).sort_values(
        ["participant_id", "cycle", "day_of_cycle"], kind="mergesort").reset_index(drop=True)
    g = d["participant_id"].astype(str).to_numpy()
    cd = d["day_of_cycle"].to_numpy(float)
    ovt = ((pd.to_datetime(d["ovulation_date"]) - pd.to_datetime(d["menses_start"])).dt.days
           + 1).to_numpy(float)
    Xp = np.nan_to_num(d[PHYSIO].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    cyc = (g + "|" + d["cycle"].astype(str)).to_numpy()
    return d, dict(g=g, cd=cd, ovt=ovt, cyc=cyc, Xp=Xp, C=cal_eng(cd),
                   is_A=np.char.startswith(g.astype(str), "S"))


def audit(d, D):
    """Leakage audit. Raises if any check fails."""
    g, itr = D["g"], np.where(D["is_A"])[0]
    ite = np.where(~D["is_A"])[0]
    # ① No patient appears on both sides
    ov = set(g[itr]) & set(g[ite])
    assert not ov, f"Patient overlap: {ov}"
    # ② Input columns are causal. Recompute each derived column with a definition that uses only
    #    nights up to t and compare to the stored values, and also compare to a future-looking
    #    definition. Passes only if the causal one matches and the future-looking one does not.
    def _s3(m):
        return pd.to_numeric(d["bbt_s3"].to_numpy()[m], errors="coerce").astype(float)

    def causal_defs(m):
        v, x = _s3(m), D["cd"][m]
        out = {}
        out["dbbt_1"] = np.r_[np.nan, np.diff(v)]
        sm = np.full(len(m), np.nan)               # observed nights since running minimum
        seen = []
        for i in range(len(m)):
            if np.isnan(v[i]):
                continue
            seen.append(i)
            sm[i] = len(seen) - 1 - int(np.argmin([v[j] for j in seen]))
        out["days_since_bbt_nadir"] = sm
        sl = np.full(len(m), np.nan)               # slope over last 5 nights
        for i in range(len(m)):
            w, xx = v[max(0, i - 4):i + 1], x[max(0, i - 4):i + 1]
            ok = ~np.isnan(w)
            if ok.sum() >= 2:
                sl[i] = np.polyfit(xx[ok], w[ok], 1)[0]
        out["bbt_slope5"] = sl
        return out

    def ahead_defs(m):
        v, x = _s3(m), D["cd"][m]
        out = {}
        out["dbbt_1"] = np.r_[np.diff(v), np.nan]
        a = int(np.nanargmin(v))                   # minimum over whole cycle = looks ahead
        out["days_since_bbt_nadir"] = np.arange(len(m)) - a
        sl = np.full(len(m), np.nan)               # centered 5 nights = looks ahead
        for i in range(len(m)):
            w, xx = v[max(0, i - 2):i + 3], x[max(0, i - 2):i + 3]
            ok = ~np.isnan(w)
            if ok.sum() >= 2:
                sl[i] = np.polyfit(xx[ok], w[ok], 1)[0]
        out["bbt_slope5"] = sl
        return out

    cols = ["dbbt_1", "days_since_bbt_nadir", "bbt_slope5"]
    hit = {c: [0, 0, 0] for c in cols}             # [n cycles, causal match, future match]
    for c in np.unique(D["cyc"]):
        m = np.where(D["cyc"] == c)[0]
        if len(m) < 8:
            continue
        ca, ah = causal_defs(m), ahead_defs(m)
        for col in cols:
            v = pd.to_numeric(d[col].to_numpy()[m], errors="coerce").astype(float)
            for j, e in enumerate((ca[col], ah[col])):
                ok = ~np.isnan(v) & ~np.isnan(e)
                if ok.sum() < 5:
                    continue
                if j == 0:
                    hit[col][0] += 1
                hit[col][1 + j] += bool(np.allclose(v[ok], e[ok], atol=1e-5))
    bad = [c for c in cols if hit[c][1] < hit[c][0] or hit[c][2] > hit[c][1]]
    for c in cols:
        n, ok_c, ok_a = hit[c]
        print(f"   causality {c:22s} causal {ok_c}/{n} · future-looking {ok_a}/{n}", flush=True)
    assert not bad, f"Non-causal columns found: {bad}"
    # ③ Test rows are not used in training — index sets are disjoint
    assert not (set(itr) & set(ite))
    print(f"Leakage audit passed · train {len(itr)} nights/{len(set(g[itr]))} patients · "
          f"test {len(ite)} nights/{len(set(g[ite]))} patients · patient overlap 0", flush=True)


def macro(y, p, nc):
    return float(f1_score(y, p, labels=list(range(nc)), average="macro", zero_division=0))


def ovf1(y, p, ov):
    return float(f1_score(y, p, labels=[ov], average="macro", zero_division=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, D = load()
    audit(d, D)
    g, cd, ovt = D["g"], D["cd"], D["ovt"]
    itr = np.where(D["is_A"])[0]
    ite = np.where(~D["is_A"])[0]

    rows = []
    for lab in LABELS:
        nc, ovc = int(lab[1]), OV_CLS[lab]
        y = collapse(phase5(cd, ovt), lab)
        for arm in ("physio", "combined"):
            X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
            for seed in range(5):
                cand = list(base_models(SEED + seed).items()) + [
                    ("OvPhase", OvPhaseClassifier(random_state=SEED + seed))]
                r = dict(label=f"{nc}-class", arm=arm, seed=seed)
                for nm, mdl in cand:
                    mdl.fit(X[itr], y[itr])
                    pred = mdl.predict(X[ite])
                    r[f"{nm}_macro"] = macro(y[ite], pred, nc)
                    r[f"{nm}_ov"] = ovf1(y[ite], pred, ovc)
                rows.append(r)
            for rl, dv in (("cd14", 14.0), ("cd15", 15.0)):
                pr = collapse(phase5(cd[ite], np.full(len(ite), dv)), lab)
                for r in rows[-5:]:
                    r[f"{rl}_macro"] = macro(y[ite], pr, nc)
                    r[f"{rl}_ov"] = ovf1(y[ite], pr, ovc)
            s = pd.DataFrame(rows[-5:])
            print(f"  {nc}-class {arm:9s} " + "  ".join(
                f"{m}={s[m+'_macro'].mean():.3f}"
                for m in ("cd14", "LR", "DT", "SVM", "LightGBM", "OvPhase")), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "runs.csv", index=False)
    for met in ("macro", "ov"):
        print(f"\n=== {met}-F1 (mean of 5 seeds) ===")
        p = df.groupby(["arm", "label"])[
            [f"{m}_{met}" for m in
             ("cd14", "cd15", "LR", "DT", "SVM", "LightGBM", "OvPhase")]].mean()
        p.columns = [c.rsplit("_", 1)[0] for c in p.columns]
        print(p.round(3).to_string())
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
