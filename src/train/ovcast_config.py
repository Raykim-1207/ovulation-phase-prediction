# -*- coding: utf-8 -*-
"""
ovcast_config — pins the final OvCast configuration in one place.

The ablation and the final comparison must use **the same settings** for the two tables to explain
each other. Defined only here; both import it.

Module configuration (fixed from 1,000-replicate ablation results)

  ④ causal cycle summary (relative)   cycle_summary, summary_mode="relative"
        relative quantities built only from nights <= t within the cycle. **Core module** —
        turning it off drops physio by -0.044 ~ -0.095.
  ⑤ two-stage residual fit           two_stage="auto"
        first fit the calendar baseline prediction; the summary only touches the residual.
        Active only when the calendar block is present. Off: combined -0.005 ~ -0.009.
  ⑦ calendar-rule prior              prior_strength
        lays the calendar rule as the initial margin to set a floor. Injected only with the calendar.
        Off: combined -0.002 ~ -0.005.
  ③ ring Bayes decision              decision="ring"
        contributes more at finer granularity (-0.005 ~ -0.010 at 4/5-class). Minor module.

**Removed** (in the ablation, turning these off was better in all seven cells)

  ① ring-distance label smoothing  sigma=0 by default. On: worse by +0.003 ~ +0.021.
  ② class-balance tilt             beta=0 by default. Difference +-0.002, no contribution.
  ⑥ causal margin smoothing        implemented, but the effect was reversed, so removed from code.

  The switches are kept so that both on and off can be shown in the ablation table.

That ⑤⑦ switch themselves on and off depending on calendar presence is the core of this configuration. It is
not a knob but a rule that follows from the input set — the two-stage fit protects the calendar baseline
prediction, and the rule prior has nothing to inject when there is no calendar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.ovcast import OvCast  # noqa: E402
from cohort import PHYSIO  # noqa: E402
from phase_labels import phase5, collapse  # noqa: E402

N_PHYSIO = len(PHYSIO)
RULE_DAY = 14.0          # ⑦ calendar rule used as the prior
#   Same rule as the prespecified comparator. Giving a better rule (cd15) only to the model
#   while comparing against the weaker rule would attribute part of the gain to the rule swap.
#   Changing it keeps scores within ±0.01 (checked directly).

FINAL = dict(
    n_estimators=300, learning_rate=0.05,
    grow_policy="lossguide", max_leaves=31, max_depth=0,
    sigma=0.0, beta=0.0, decision="ring",                       # ①② removed, ③ kept
    cycle_summary=True, wins=(3, 5, 7), summary_mode="relative",  # ④
    summary_cols=list(range(N_PHYSIO)),
    two_stage="auto", stage2_frac=0.5,                          # ⑤
    prior_strength=2.0,                                          # ⑦
)

# LightGBM baseline: settings chosen on Hospital A validation (per input set)
LGBM = {"combined": dict(num_leaves=7, max_depth=6, learning_rate=0.03),
        "physio": dict(num_leaves=15, max_depth=6, learning_rate=0.08)}


def rule_pred(cd, label):
    """Calendar-rule phase for ⑦. Used only in configurations with calendar input."""
    return collapse(phase5(cd, np.full(len(cd), RULE_DAY)), label)


def build(arm, label, cd, seed, **over):
    """Build the final OvCast and, for configurations with the calendar, prepare the rule prior.

    Returns: (model, rule prediction or None)
    """
    cfg = dict(FINAL)
    cfg.update(over)
    if arm != "combined":
        cfg["prior_strength"] = 0.0      # no injection without the calendar
    m = OvCast(random_state=seed, **cfg)
    rule = rule_pred(cd, label) if cfg["prior_strength"] > 0 else None
    return m, rule


def fit_predict(m, rule, X, cyc, cd, y, itr, ite):
    """Fit and predict, passing structure and prior sliced to the split."""
    m.set_structure(cyc[itr], cd[itr])
    m.set_prior(None if rule is None else rule[itr])
    m.fit(X[itr], y[itr])
    m.set_structure(cyc[ite], cd[ite])
    m.set_prior(None if rule is None else rule[ite])
    return np.asarray(m.predict(X[ite]))
