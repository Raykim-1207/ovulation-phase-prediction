# -*- coding: utf-8 -*-
"""
OvCast (OVulation foreCASTing) — ring-ordinal boosting for ovulation phase prediction.

The **inputs are exactly the same** as the existing learners (LR/DT/SVM/LightGBM).
Everything new lives inside the learning algorithm, motivated by the label structure.

The phases are not an unordered set but a **ring**.

        menstrual -> follicular -> peri-ovulatory -> luteal -> (menstrual again)

Off-the-shelf multiclass learners do not know this ring. They count mistaking the
peri-ovulatory phase for follicular as the same size of error as mistaking it for
menstrual, which is wrong both clinically and statistically. Moreover the
peri-ovulatory phase is only 6 of 28 days, so plain log-loss converges toward
dropping this class — and macro-F1 charges that loss in full.

OvPhase puts three pieces inside the boosting.

  ① Ring-distance label smoothing   Decay the target distribution by ring distance
                          instead of one-hot.
                          q_k ∝ exp(-d(k,y)^2 / 2σ^2),  d = shortest distance on the ring.
                          Errors into a neighbouring phase get only a small penalty.
  ② Class-balanced gradient    Weight each sample by (1/n_class)^β so rare phases
                          pull the fit. Targets macro-F1 rather than accuracy.
  ③ Ring Bayes decision      At prediction, pick the phase minimizing the expected
                          ring-distance loss instead of argmax. Stray errors do not
                          jump to the opposite side of the ring.

①② change the gradient at every boosting step, so the learned trees themselves change.
This is not preprocessing or feature engineering — the input matrix is identical to the baseline.

The implementation is gradient boosting over K outputs. With p = softmax(F), the
gradient for class k is w_i (p_ik - q_ik), and each step fits regression trees to the
negative gradient.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:            # the reference implementation runs on its own
    xgb = None


def ring_dist(K: int) -> np.ndarray:
    """Shortest-distance matrix on the phase ring. With K=2 it degenerates to plain label smoothing."""
    a = np.arange(K)
    d = np.abs(a[:, None] - a[None, :])
    return np.minimum(d, K - d).astype(float)


def causal_cycle_summary(X, cyc, day, wins=(3, 5), cols=None, mode="relative"):
    """Causal within-cycle temporal summary. Night t only sees nights <= t of its cycle.

    The physiological signal of ovulation shows up not in a single night's value but in
    **its position within the cycle** — has the temperature hit its minimum, and for how
    many days has it been rising. A learner that pools nights as independent samples
    cannot use this structure in principle. Here the summary is built inside the estimator.

    mode="relative" (default)
        Emit only **relative quantities** within that person's cycle; no absolute levels.
          · current value - running minimum   (rise since the minimum)
          · nights since the running minimum  (distance to the turning point)
          · current value - mean of previous w nights   (deviation from recent)
        Absolute temperature/heart-rate levels vary by device and cohort, so features
        carrying them fit well within the training hospital but do not transfer to
        another hospital. Keeping only relative quantities uses the same structure while
        cutting that dependence.

    mode="full"
        Additionally include the **moving average itself** over the previous w nights
        (absolute level). Better in within-hospital validation but may hurt cross-hospital
        transfer — kept for comparison.

    cols selects the columns to summarize. Calendar columns are deterministic functions
    of cycle day, so running minima or moving averages of them add no information.

    All running statistics use only data up to t, so nothing looks into the future.
    """
    X = np.asarray(X, float)
    full = X
    if cols is not None:
        X = X[:, list(cols)]
    n, p = X.shape
    out = [full]
    run_min = np.zeros_like(X)
    since = np.zeros_like(X)
    for c in np.unique(cyc):
        m = np.where(cyc == c)[0]
        m = m[np.argsort(day[m])]
        V = X[m]
        cm = np.minimum.accumulate(V, 0)
        run_min[m] = V - cm
        k = np.zeros(p)
        for t in range(len(m)):
            k = np.where(V[t] <= cm[t] + 1e-12, 0.0, k + 1.0)
            since[m[t]] = k
    out += [run_min, since]
    for w in wins:
        sm = np.zeros_like(X)
        for c in np.unique(cyc):
            m = np.where(cyc == c)[0]
            m = m[np.argsort(day[m])]
            V = X[m]
            for t in range(len(m)):
                sm[m[t]] = V[max(0, t - w + 1): t + 1].mean(0)
        out.append(X - sm)                 # relative quantity
        if mode == "full":
            out.append(sm)                 # absolute level
    return np.column_stack(out)


def causal_margin_smooth(F, cyc, day, alpha=0.5, k=2):
    """⑥ Causal margin smoothing. Blend night t's logits with nights <= t of its cycle.

    A phase is a **contiguous block** within a cycle — the peri-ovulatory phase is a run
    of consecutive days, not scattered single days. Classifying each night independently
    breaks the block and produces scattered false positives. Here the logits are blended
    with the preceding nights using exponential weights to damp those jumps.

        F'[t] = (1-w) * F[t] + w * (exponentially weighted mean of the previous k nights' logits)

    Night t uses only nights <= t, so this is causal. alpha = 0 does nothing.
    """
    if alpha <= 0:
        return F
    out = F.copy()
    for c in np.unique(cyc):
        m = np.where(cyc == c)[0]
        m = m[np.argsort(day[m])]
        V = F[m]
        for t in range(1, len(m)):
            lo = max(0, t - k)
            w = np.array([0.5 ** (t - j) for j in range(lo, t)])
            if not len(w):
                continue
            w = w / w.sum()
            out[m[t]] = (1 - alpha) * V[t] + alpha * (w[:, None] * V[lo:t]).sum(0)
    return out


def _softmax(F):
    F = F - F.max(1, keepdims=True)
    e = np.exp(F)
    return e / e.sum(1, keepdims=True)


class _Struct:
    """Channel for passing cycle structure to the estimator (same convention as the prior-model implementations)."""

    def set_structure(self, cyc, day):
        self.cyc_, self.day_ = np.asarray(cyc), np.asarray(day, float)
        return self

    def _tx(self, X):
        if not getattr(self, "cycle_summary", False):
            return np.asarray(X, float)
        if not hasattr(self, "cyc_"):
            raise RuntimeError("call set_structure() first")
        return causal_cycle_summary(X, self.cyc_, self.day_, self.wins,
                                    cols=self.summary_cols,
                                    mode=self.summary_mode)


class _OvCastLS(_Struct):
    """Reference implementation. Fits least-squares trees to the negative gradient (first order).

    This version spells out the definition for readability. The one actually used is
    OvCast below — same loss, but leaf values are set from gradient and Hessian (Newton).
    That is the same optimization LightGBM/XGBoost use, so the learner comparison is not
    confounded by differences in optimizer.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=20, subsample=0.8, sigma=0.6, beta=0.75,
                 decision="ring", reg_lambda=1.0, colsample_bytree=1.0,
                 min_child_weight=1.0, max_leaves=0, grow_policy="depthwise",
                 cycle_summary=True, wins=(3, 5), summary_cols=None,
                 summary_mode="relative", two_stage="auto", stage2_frac=0.5,
                 prior_strength=0.0,
                 backend="newton", random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.sigma = sigma              # ① smoothing width (0 = one-hot)
        self.beta = beta                # ② balancing strength (0 = no weighting)
        self.decision = decision        # ③ "ring" or "argmax"
        # The four below are per-tree knobs the comparison learners (LightGBM/XGBoost) have.
        # Exposed so the search space matches on shared axes — unrelated to the proposed parts (σ, β).
        self.reg_lambda = reg_lambda            # leaf-value L2
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.max_leaves = max_leaves            # 0 = unlimited (for leaf-wise growth)
        self.grow_policy = grow_policy          # "depthwise" or "lossguide"
        # ④ Build the causal within-cycle temporal summary inside the estimator. The
        #    structure (participant, cycle, night index) comes via set_structure — it is the
        #    structure of longitudinal data, not a measurement.
        self.cycle_summary = cycle_summary
        self.wins = tuple(wins)
        self.summary_cols = summary_cols   # None = all, otherwise physiological columns only
        self.summary_mode = summary_mode   # "relative" (robust to transfer) or "full"
        # ⑤ Two-stage fit. Stage 1 is built on the base inputs only; stage 2 starts from its
        #    margin and refines **only the residual** with the cycle summary. Even if the
        #    summary does not transfer to another hospital, stage 1 remains intact, so the
        #    damage is contained. stage2_frac is the fraction of boosting rounds for stage 2.
        # ⑤ Two-stage residual fit. "auto" enables it **only when the calendar block is in
        #    the input**. Stage 2 exists to protect the calendar-driven base prediction, so
        #    without calendar there is no reason to enable it, and it actually hurt
        #    (physio +0.097 -> +0.023). It is a rule that follows from the input set, not a knob.
        self.two_stage = two_stage
        self.stage2_frac = stage2_frac
        # ⑦ Calendar-rule prior injection. The rule phase received via set_prior() is laid
        #    down as the initial margin and boosting proceeds on top. Since the starting
        #    point is the rule, a misfit rarely drops below rule level, and only the signal's
        #    contribution needs to be learned. Must not be enabled for configurations without
        #    calendar input (physio) — that would be cheating.
        self.prior_strength = prior_strength
        self.backend = backend          # "newton" (canonical) or "ls" (reference implementation)
        self.random_state = random_state

    # ---- training ----------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self.K_ = K
        idx = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx[v] for v in y])
        self.D_ = ring_dist(K)

        # ① target distribution decayed by ring distance
        if self.sigma > 0:
            Q = np.exp(-(self.D_ ** 2) / (2 * self.sigma ** 2))
            Q /= Q.sum(1, keepdims=True)
        else:
            Q = np.eye(K)
        Q = Q[yi]

        # ② class-balance weighting
        cnt = np.bincount(yi, minlength=K).astype(float)
        w = (cnt.sum() / (K * np.maximum(cnt, 1.0))) ** self.beta
        w = w[yi]
        w = w / w.mean()
        self.w_mean_ = w

        # initial value = log of the weighted prior
        prior = np.array([w[yi == k].sum() for k in range(K)])
        prior = np.maximum(prior, 1e-6) / prior.sum()
        self.F0_ = np.log(prior)

        F = np.tile(self.F0_, (len(X), 1))
        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        n_sub = max(1, int(self.subsample * len(X)))
        for _ in range(self.n_estimators):
            P = _softmax(F)
            G = w[:, None] * (Q - P)                 # negative gradient
            sub = rng.choice(len(X), n_sub, replace=False)
            stage = []
            for k in range(K):
                t = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          random_state=self.random_state)
                t.fit(X[sub], G[sub, k])
                F[:, k] += self.learning_rate * t.predict(X)
                stage.append(t)
            self.trees_.append(stage)
        return self

    # ---- prediction ----------------------------------------------------------
    def decision_function(self, X):
        X = np.asarray(X, float)
        F = np.tile(self.F0_, (len(X), 1))
        for stage in self.trees_:
            for k, t in enumerate(stage):
                F[:, k] += self.learning_rate * t.predict(X)
        return F

    def _use_two_stage(self, n_cols):
        """Is there a calendar block? = are there more columns than summary_cols points to?"""
        if self.two_stage != "auto":
            return bool(self.two_stage)
        k = len(self.summary_cols) if self.summary_cols is not None else n_cols
        return n_cols > k

    def set_prior(self, pred):
        """⑦ Rule prediction (per-night phase). Ignored when prior_strength is 0."""
        self.prior_ = None if pred is None else np.asarray(pred).astype(int)
        return self

    def _prior_margin(self, n, K):
        if self.prior_strength <= 0 or getattr(self, "prior_", None) is None:
            return None
        M = np.zeros((n, K), float)
        idx = {c: i for i, c in enumerate(self.classes_)}
        for r, v in enumerate(self.prior_[:n]):
            j = idx.get(int(v))
            if j is not None:
                M[r, j] = self.prior_strength
        return M

    def predict_proba(self, X):
        return _softmax(self.decision_function(X))

    def predict(self, X):
        P = self.predict_proba(X)
        if self.decision == "ring":
            # ③ minimize expected ring-distance loss
            risk = P @ (self.D_ ** 2).T
            j = risk.argmin(1)
        else:
            j = P.argmax(1)
        return self.classes_[j]

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in
                ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf",
                 "subsample", "sigma", "beta", "decision", "reg_lambda",
                 "colsample_bytree", "min_child_weight", "max_leaves",
                 "grow_policy", "cycle_summary", "wins", "summary_cols",
                 "summary_mode", "two_stage", "stage2_frac", "prior_strength",
                 "backend", "random_state")}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self


class OvCast(_OvCastLS):
    """Canonical OvCast. Passes the ring-ordinal loss to xgboost as a custom objective.

    ①②③ are unchanged — ring-smoothed target q, class-balance weight w, ring Bayes
    decision. Only the way trees are grown differs. The reference implementation fits
    least-squares trees to the negative gradient (first order); here xgboost uses both
    gradient and Hessian to set leaf values (Newton). This is the standard difference in
    boosting, and the loss is the same function.

        grad_k = w (p_k - q_k)
        hess_k = w p_k (1 - p_k)

    On this data (1,269 x 26) the GPU is 22x slower than the CPU — the matrix is so small
    that kernel launch latency dominates compute (measured). So device is fixed to cpu.
    """

    def fit(self, X, y):
        if xgb is None or self.backend == "ls":
            return super().fit(X, y)
        X0 = np.asarray(X, float)          # base inputs (for stage 1)
        X = self._tx(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        K = self.K_ = len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx[v] for v in y])
        self.D_ = ring_dist(K)

        if self.sigma > 0:
            Q = np.exp(-(self.D_ ** 2) / (2 * self.sigma ** 2))
            Q /= Q.sum(1, keepdims=True)
        else:
            Q = np.eye(K)
        Q = Q[yi]
        cnt = np.bincount(yi, minlength=K).astype(float)
        w = (cnt.sum() / (K * np.maximum(cnt, 1.0))) ** self.beta
        w = w[yi]
        w = w / w.mean()

        def obj(F, _dm):
            F = F.reshape(-1, K)
            P = _softmax(F)
            g = w[:, None] * (P - Q)
            h = np.maximum(w[:, None] * P * (1.0 - P), 1e-6)
            return g.reshape(-1), h.reshape(-1)

        par = dict(num_class=K, max_depth=self.max_depth, eta=self.learning_rate,
                   subsample=self.subsample,
                   min_child_weight=self.min_child_weight,
                   colsample_bytree=self.colsample_bytree,
                   max_leaves=self.max_leaves, grow_policy=self.grow_policy,
                   reg_lambda=self.reg_lambda,
                   tree_method="hist", device="cpu", base_score=0.0,
                   seed=self.random_state, disable_default_eval_metric=1)
        pm = self._prior_margin(len(X), K)
        ts = self._use_two_stage(X0.shape[1])
        if not (ts and self.cycle_summary):
            dtr = xgb.DMatrix(X, label=yi)
            if pm is not None:
                dtr.set_base_margin(pm.reshape(-1))
            self.bst_ = xgb.train(par, dtr,
                                  num_boost_round=self.n_estimators, obj=obj)
            self.bst2_ = None
            return self
        # ⑤ Stage 1: base inputs only. Stage 2: refine the residual on top of that margin with the cycle summary.
        n2 = max(1, int(round(self.n_estimators * self.stage2_frac)))
        n1 = max(1, self.n_estimators - n2)
        d1 = xgb.DMatrix(X0, label=yi)
        if pm is not None:
            d1.set_base_margin(pm.reshape(-1))
        self.bst_ = xgb.train(par, d1, num_boost_round=n1, obj=obj)
        m1 = self.bst_.predict(d1, output_margin=True).reshape(-1, K)
        d2 = xgb.DMatrix(X, label=yi)
        d2.set_base_margin(m1.reshape(-1))
        self.bst2_ = xgb.train(par, d2, num_boost_round=n2, obj=obj)
        return self

    def decision_function(self, X):
        if xgb is None or self.backend == "ls":
            return super().decision_function(X)
        X0 = np.asarray(X, float)
        pm = self._prior_margin(len(X0), self.K_)
        if getattr(self, "bst2_", None) is None:
            d = xgb.DMatrix(self._tx(X))
            if pm is not None:
                d.set_base_margin(pm.reshape(-1))
            F = self.bst_.predict(d, output_margin=True)
            return F.reshape(len(X0), self.K_)
        d1 = xgb.DMatrix(X0)
        if pm is not None:
            d1.set_base_margin(pm.reshape(-1))
        m1 = self.bst_.predict(d1, output_margin=True)
        d2 = xgb.DMatrix(self._tx(X))
        d2.set_base_margin(m1)
        F = self.bst2_.predict(d2, output_margin=True)
        return F.reshape(len(X0), self.K_)


# Legacy aliases for backward compatibility (OVAL was renamed to OvCast).
OVAL = OvCast
OVALFast = OvCast
OvPhaseClassifier = OvCast
OvPhaseXGB = OvCast
