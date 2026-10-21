# -*- coding: utf-8 -*-
"""auroc_ovcast — peri-ovulatory class AUROC·AUPRC of OvCast from the original fit (b=0) (secondary metrics in Methods).

Follows the prediction path of ovulation_day_error exactly (CPU, same seed). The class predictions must equal pred_nights
for the probabilities to come from the same model — that comparison is the self-check. The calendar rule outputs no probabilities, so it is excluded.

  OVUL_DATA_DIR_ROOT=.../ovul_data OVUL_FINAL_DIR=.../locf python3 scripts/auroc_ovcast.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU, same as the original run
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PAPER = Path(__file__).resolve().parent
sys.path[:0] = [str(PAPER.parent / "src" / "train"),
                str(PAPER.parent / "src")]
import ovulation_day_error as E                                    # noqa: E402


def main():
    d, D = E.load()
    rows = []
    for lab in E.LABELS:
        nc, ovc = int(lab[1]), E.OV_CLS[lab]
        y = E.collapse(E.phase5(D["cd"], D["ovt"]), lab)
        ite = np.where(~D["is_A"])[0]
        itr = np.where(D["is_A"])[0]                            # b=0: all training participants
        for arm in ("physio", "combined"):
            X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
            m, rule = E.build_ovcast(arm, lab, D["cd"], E.SEED)
            pred = E.fit_predict(m, rule, X, D["cyc"], D["cd"], y, itr, ite)
            ref = pd.read_csv(PAPER / "pred_nights" / f"{lab}_{arm}.csv")
            assert (ref.y_pred.to_numpy() == pred).all(), f"{lab} {arm}: differs from pred_nights"
            p = np.asarray(m.predict_proba(X[ite]))[:, ovc]
            t = (y[ite] == ovc).astype(int)
            rows.append(dict(label=lab, arm=arm, auroc=roc_auc_score(t, p),
                             auprc=average_precision_score(t, p)))
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PAPER / "fig" / "auroc_ovcast.csv", index=False)
    for arm in ("combined", "physio"):
        s = out[out.arm == arm]
        print(f"{arm}: AUROC {s.auroc.min():.3f}--{s.auroc.max():.3f}, "
              f"AUPRC {s.auprc.min():.3f}--{s.auprc.max():.3f}")


if __name__ == "__main__":
    main()
