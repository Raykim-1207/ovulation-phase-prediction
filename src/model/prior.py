# -*- coding: utf-8 -*-
"""
Models from four prior ovulation studies, implemented within our evaluation framework.

  [1] Goodale 2019 (JMIR)          Ava bracelet. Random Forest with 1,000 trees, max_features=3.
                                   3 phases (follicular/fertile/luteal), fertile = ovulation -5..0 days.
  [2] Yu 2022 (Reprod Biol Endo)   Huawei Band 5. Smooths BBT and HR waveforms per cycle,
                                   locates fertile days at the biphasic transition point, then fits
                                   a probability model on the immediately preceding run.
  [3] Kilungeja 2025 (npj Womens H) Wrist device. Fixed-window features -> Random Forest.
                                   Daily 4-phase tracking uses moving windows.
  [4] Maijala 2019 (BMC Womens H)  Oura ring. Training-free rule — ovulation is placed where nightly
                                   skin temperature rises 0.15 degrees above baseline (RISE_0.15).

**Comparison protocol**. The four papers differ in both phase definitions and validation, so their
reported numbers cannot be compared side by side. We therefore **take only the models and use our
own evaluation framework** — same input matrix, Hospital A -> Hospital B external validation, same labels, same metrics.

**Input identity**. Every model receives the same base matrix X. Models that transform their input
by design ([2] smoothing/transition point, [3] moving windows) do so inside the estimator —
the same place where OvCast applies its ring loss internally. All transforms are **causal**:
the value for night t uses only nights <= t of that cycle.

The exact rule of [4] (baseline length, number of consecutive days) could not be obtained from the full text.
We implement the classic three-over-six structure with the reported 0.15 degree threshold and expose the
parameters. They should be aligned once the full text is available.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- shared helpers
def _by_cycle(cyc, day):
    """Return (row indices, sorted by date) for each cycle."""
    for c in np.unique(cyc):
        m = np.where(cyc == c)[0]
        yield c, m[np.argsort(day[m])]


def trailing(X, cyc, day, win):
    """Causal moving-window mean. Night t sees only nights max(0,t-win+1)..t of its cycle."""
    out = np.zeros_like(X, dtype=float)
    for _, m in _by_cycle(cyc, day):
        V = X[m]
        for t in range(len(m)):
            out[m[t]] = V[max(0, t - win + 1): t + 1].mean(0)
    return out


class _Struct:
    """Channel that passes cycle structure (patient, cycle, night number) to fit/predict.

    This is not a measurement but structural information any longitudinal dataset has. It lets
    structure-aware and structure-agnostic models share one interface.
    """

    def set_structure(self, cyc, day):
        self.cyc_, self.day_ = np.asarray(cyc), np.asarray(day, float)
        return self


# ------------------------------------------------ [1] Goodale 2019 — RF
class Goodale2019(_Struct):
    """Random Forest with 1,000 trees. The paper used max_features=3 with 11 features.

    Applying that 3 unchanged to our matrix with a different column count changes the **fraction** of
    features seen per split (3/11 = 27% -> 3/26 = 12%), departing from the paper and unfairly weakening the model.
    So we transfer the fraction 3/11, not the count. The fraction is exposed as a parameter.
    """

    RATIO = 3.0 / 11.0

    def __init__(self, n_estimators=1000, ratio=RATIO, random_state=42):
        self.n_estimators = n_estimators
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        mf = max(1, int(round(self.ratio * X.shape[1])))
        self.m_ = RandomForestClassifier(n_estimators=self.n_estimators,
                                         max_features=mf, n_jobs=1,
                                         random_state=self.random_state)
        self.m_.fit(X, y)
        self.classes_ = self.m_.classes_
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X, float))

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, float))


# --------------------------------- [2] Yu 2022 — biphasic transition point + probability estimate
class Yu2022(_Struct):
    """Per-cycle smoothing -> biphasic transition point -> probability from the preceding run.

    Follows the paper's description ("smooth BBT and HR waveforms per cycle, extract features from the
    biphasic rule so that fertile days fall at the transition point, and estimate whether a day is fertile
    from the run immediately preceding it"). Builds the following, all causal:

      · smoothed mean of the previous LAG nights   (waveform smoothing)
      · current value - running minimum           (rise)
      · nights elapsed since the running minimum   (distance to transition point)
      · slope of the previous LAG nights           (transition direction)

    These go into logistic regression to give per-night phase probabilities.
    """

    def __init__(self, cols=(0, 1), lag=3, random_state=42):
        # The paper mainly used BBT and HR, and also tested ln(SDNN) and ln(LF/HF).
        # Passing all four columns in cols is closest to the paper's intent.
        self.cols = cols
        self.LAG = lag
        self.random_state = random_state

    def _biphasic(self, X):
        cyc, day = self.cyc_, self.day_
        Z = X[:, list(self.cols)]
        sm = trailing(Z, cyc, day, self.LAG)
        rise = np.zeros_like(Z)
        since = np.zeros((len(X), Z.shape[1]))
        slope = np.zeros_like(Z)
        for _, m in _by_cycle(cyc, day):
            V = sm[m]
            for t in range(len(m)):
                w = V[: t + 1]
                mn = w.min(0)
                rise[m[t]] = V[t] - mn
                since[m[t]] = t - w.argmin(0)
                a = max(0, t - self.LAG + 1)
                slope[m[t]] = (V[t] - V[a]) / max(1.0, t - a)
        return np.column_stack([X, sm, rise, since, slope])

    def fit(self, X, y):
        F = self._biphasic(np.asarray(X, float))
        self.m_ = Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=3000,
                                                     random_state=self.random_state))])
        self.m_.fit(F, y)
        self.classes_ = self.m_[-1].classes_
        return self

    def predict(self, X):
        return self.m_.predict(self._biphasic(np.asarray(X, float)))

    def predict_proba(self, X):
        return self.m_.predict_proba(self._biphasic(np.asarray(X, float)))


# ------------------------------- [3] Kilungeja 2025 — moving-window features + RF
class Kilungeja2025(_Struct):
    """Random Forest after appending causal moving-window (1, 3, 5 nights) summaries.

    The paper used fixed-window features for 3 phases and moving windows for daily 4-phase tracking.
    Here we follow the daily tracking variant (moving windows).
    """

    def __init__(self, wins=(1, 3, 5), n_estimators=500, random_state=42):
        # The paper only says "fixed-size window" and "moving window" without giving sizes.
        # To avoid handicapping it with an arbitrary choice, it is a parameter chosen on
        # Hospital A validation — the same procedure as our model.
        self.WINS = tuple(wins)
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _win(self, X):
        return np.column_stack([trailing(X, self.cyc_, self.day_, w)
                                for w in self.WINS])

    def fit(self, X, y):
        self.m_ = RandomForestClassifier(n_estimators=self.n_estimators,
                                         random_state=self.random_state, n_jobs=1)
        self.m_.fit(self._win(np.asarray(X, float)), y)
        self.classes_ = self.m_.classes_
        return self

    def predict(self, X):
        return self.m_.predict(self._win(np.asarray(X, float)))

    def predict_proba(self, X):
        return self.m_.predict_proba(self._win(np.asarray(X, float)))


# --------------------------------- [4] Maijala 2019 — rule RISE_0.15
class Maijala2019(_Struct):
    """Training-free temperature-rise rule. Phases are reconstructed from the estimated ovulation day.

    The paper states the 0.15 degree threshold as a "**between-phase** difference" and does not give
    baseline length or number of consecutive days. Fixing one reading would let that assumption decide the
    prior study's score, so all three readings are implemented and the choice is made within Hospital A.

      mode="phase"     try every split of the cycle into two segments, take the one with the largest
                       **difference in segment means**; if it is >= thr, the day before that point
                       is the ovulation day.  <- closest to the paper's wording ("0.15 degree between phases")
      mode="baseline"  first point exceeding the mean of the previous base nights + thr for consec nights in a row
      mode="cover"     first point exceeding the **maximum** of the previous base nights + thr for consec nights in a row
                       (cover line of the classic three-over-six)

    In every mode, if the condition is not met within the cycle, the ovulation day is treated as not found
    and fallback is used.
    """

    def __init__(self, bbt_col=0, thr=0.15, base=6, consec=3, mode="phase",
                 min_side=4, fallback=15.0):
        self.bbt_col, self.thr = bbt_col, thr
        self.base, self.consec, self.mode = base, consec, mode
        self.min_side, self.fallback = min_side, fallback

    def fit(self, X, y=None):
        self.classes_ = np.unique(y) if y is not None else None
        return self

    def _one_cycle(self, v, d):
        n = len(v)
        if self.mode == "phase":
            best, gap = None, -np.inf
            for t in range(self.min_side, n - self.min_side + 1):
                g = v[t:].mean() - v[:t].mean()
                if g > gap:
                    gap, best = g, t
            return (d[best] - 1.0) if (best is not None and gap >= self.thr) else None
        for t in range(self.base, n - self.consec + 1):
            w = v[t - self.base:t]
            ref = w.max() if self.mode == "cover" else w.mean()
            if np.all(v[t:t + self.consec] > ref + self.thr):
                return d[t] - 1.0
        return None

    def ovulation_day(self, X):
        """Return, per night, the estimated ovulation cycle day for its cycle."""
        b = np.asarray(X, float)[:, self.bbt_col]
        out = np.full(len(b), self.fallback)
        for _, m in _by_cycle(self.cyc_, self.day_):
            hit = self._one_cycle(b[m], self.day_[m])
            out[m] = self.fallback if hit is None else hit
        return out
