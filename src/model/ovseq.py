# -*- coding: utf-8 -*-
"""
OvSeq — proposed model that classifies the nightly phase by accumulating cycle history.

The existing models (LR/DT/SVM/LightGBM) see only that one night. They do have fixed-lag
differences (1 and 3 days earlier), but know nothing about what has happened since the
cycle began.

At night t, OvSeq uses a **cumulative summary of nights 1..t of that cycle** as features.
Days since the temperature minimum, rise above the minimum, rank of the current value so
far — such quantities capture the timing of the temperature rise regardless of cycle length.

  · Causal. Night t's features never include nights > t.
  · Same interface as the existing models (X -> class probabilities). No selection device, no calendar rule.
  · History features are built from physiological signals only, so they exist in the physio-only configuration too.

An early design had a decoder choosing a contiguous 6-day window per cycle, but it was
dropped. Choosing that window required seeing the later part of the cycle, i.e. looking
into the future, and restricting it causally gave exactly zero gain. Before ovulation
there is no way to know where the window is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

HIST_SRC = ["bbt", "hr", "sdnn", "rmssd", "lfhf", "bbt_slope5", "rising_streak"]
PARAMS = dict(n_estimators=120, learning_rate=0.02, max_depth=3,
              subsample=0.8, min_samples_leaf=20)


def causal_history(d: pd.DataFrame,
                   keys=("participant_id", "cycle"), order="day_of_cycle"):
    """Features of night t = cumulative summary of nights 1..t of its cycle. Never looks ahead."""
    rows, names = [], None
    for _, gd in d.groupby(list(keys), sort=False):
        gd = gd.sort_values(order)
        cd = gd[order].to_numpy(float)
        V = np.nan_to_num(gd[HIST_SRC].apply(pd.to_numeric, errors="coerce")
                          .to_numpy(float))
        for t in range(len(gd)):
            w, c = V[: t + 1], cd[: t + 1]
            bbt, slope, streak = w[:, 0], w[:, 5], w[:, 6]
            f = {
                "n_seen": t + 1.0,
                "cd_since_bbt_min": cd[t] - c[int(np.argmin(bbt))],
                "cd_at_bbt_min": c[int(np.argmin(bbt))],
                "bbt_minus_min": bbt[t] - bbt.min(),
                "bbt_minus_runmean": bbt[t] - bbt.mean(),
                "bbt_z_sofar": (bbt[t] - bbt.mean()) / (bbt.std() + 1e-6),
                "bbt_rank_sofar": float((bbt <= bbt[t]).mean()),
                "slope_cummax": slope.max(),
                "cd_at_max_slope": c[int(np.argmax(slope))],
                "streak_cummax": streak.max(),
                "hr_z_sofar": (w[t, 1] - w[:, 1].mean()) / (w[:, 1].std() + 1e-6),
                "sdnn_z_sofar": (w[t, 2] - w[:, 2].mean()) / (w[:, 2].std() + 1e-6),
                "rmssd_z_sofar": (w[t, 3] - w[:, 3].mean()) / (w[:, 3].std() + 1e-6),
                "lfhf_z_sofar": (w[t, 4] - w[:, 4].mean()) / (w[:, 4].std() + 1e-6),
            }
            if names is None:
                names = list(f)
            rows.append([f[k] for k in names])
    return np.asarray(rows, float), names


def build(seed: int = 42) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=seed, **PARAMS)
