# -*- coding: utf-8 -*-
"""PDCA protocol helpers — group splits, causal patnorm, cluster bootstrap, LGBM/DL defaults."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# Pre-registered confirmatory cell (edit before peeking at sg).
# Primary metric = ov_f1 (fertile/ovulation class), not macro F1.
PRIMARY: dict[str, str] = {
    "family": "interp_huawei",
    "method": "linear",
    "label": "y5_mod",
    "arm": "combined",
    "split": "sg",
    "model": "OvPhysio",
    "baseline_arm": "cal",
    "baseline_model": "rule_cd14",
    "metric": "ov_f1",
}

LGBM_KW: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "min_child_samples": 20,
    "random_state": 42,  # reproducible combined/physio LGBM (SOTA + proposed)
    # An earlier result could not be reproduced in the current environment (physio ov-F1 0.510 -> 0.474).
    # The cause was not identified. What was confirmed: (1) the calendar rule reproduces exactly, (2) AUROC is
    # nearly the same (0.7534 -> 0.7530), and (3) the only change is the threshold chosen by validation (0.05 -> 0.20).
    # LightGBM versions (4.4/4.5/4.6) and device_type differences were ruled out.
    # The two flags below are LightGBM's documented reproducibility combination; whatever the cause, they remove
    # tree differences due to build and thread count, guaranteeing reproducibility going forward.
    "deterministic": True,
    "force_row_wise": True,
}
# Diagnostic: set OVUL_LGBM_FLAGS=off to run without the two flags and compare speed.
import os as _os
if _os.environ.get("OVUL_LGBM_FLAGS") == "off":
    LGBM_KW.pop("deterministic", None)
    LGBM_KW.pop("force_row_wise", None)
elif _os.environ.get("OVUL_LGBM_FLAGS") == "det_only":
    LGBM_KW.pop("force_row_wise", None)

DL_N_SEEDS = 5
TEST_CLUSTER_BOOT = 200


def split_fit_val_idx(
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    min_n: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-group holdout for threshold / α / blend."""
    n = len(y)
    idx = np.arange(n)
    y = np.asarray(y)
    if n < min_n:
        return idx.copy(), idx.copy()

    if groups is not None and len(groups) == n:
        groups = np.asarray(groups)
        n_pat = len(np.unique(groups))
        if n_pat >= 3:
            try:
                n_splits = max(3, min(5, n_pat // 2))
                sgkf = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state,
                )
                for i_fit, i_val in sgkf.split(idx, y, groups):
                    if len(i_fit) >= 2 and len(i_val) >= 2:
                        return i_fit, i_val
            except ValueError:
                pass
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state,
            )
            i_fit, i_val = next(gss.split(idx, y, groups))
            if len(i_fit) >= 2 and len(i_val) >= 2:
                return i_fit, i_val

    from sklearn.model_selection import train_test_split

    try:
        return train_test_split(
            idx, test_size=test_size, stratify=y, random_state=random_state,
        )
    except ValueError:
        i_val = idx[: max(1, n // 5)]
        i_fit = idx[max(1, n // 5):]
        return i_fit, i_val


def bootstrap_cluster_idx(
    groups: np.ndarray,
    mask: Optional[np.ndarray],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Patient-level resample (train or test rows)."""
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(groups))
    if len(idx) == 0:
        return idx
    pats = np.unique(groups[idx])
    boot_pats = rng.choice(pats, size=len(pats), replace=True)
    by_p = {p: idx[groups[idx] == p] for p in pats}
    return np.concatenate([by_p[p] for p in boot_pats])


def causal_patient_median_norm(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    g_tr: np.ndarray,
    g_te: np.ndarray,
    physio_idx: list[int],
    cycle_tr: np.ndarray,
    cycle_day_tr: np.ndarray,
    cycle_te: np.ndarray,
    cycle_day_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal: row (pid, cycle, day d) minus median physio of same pid+cycle, day < d.
    Test uses train history + earlier test nights in the same cycle.
    """
    if not physio_idx:
        return X_tr, X_te
    X_tr = np.asarray(X_tr, dtype=np.float64).copy()
    X_te = np.asarray(X_te, dtype=np.float64).copy()
    idx = np.asarray(physio_idx, dtype=int)
    g_tr, g_te = np.asarray(g_tr), np.asarray(g_te)
    cyc_tr, cd_tr = np.asarray(cycle_tr), np.asarray(cycle_day_tr, dtype=np.float64)
    cyc_te, cd_te = np.asarray(cycle_te), np.asarray(cycle_day_te, dtype=np.float64)

    hist: dict[tuple[Any, Any], list[tuple[float, np.ndarray]]] = {}
    global_med = np.nanmedian(X_tr[:, idx], axis=0)
    global_med = np.where(np.isfinite(global_med), global_med, 0.0)

    def _med_for(key, pid, d: float) -> np.ndarray:
        past = [v for day, v in hist.get(key, []) if day < d - 1e-9]
        if past:
            med = np.nanmedian(np.stack(past), axis=0)
        else:
            other = [v for (p, _), rows in hist.items() if p == pid for _, v in rows]
            med = np.nanmedian(np.stack(other), axis=0) if other else global_med
        return np.where(np.isfinite(med), med, global_med)

    for i in range(len(X_tr)):
        key = (g_tr[i], cyc_tr[i])
        raw = X_tr[i, idx].copy()
        X_tr[i, idx] = raw - _med_for(key, g_tr[i], float(cd_tr[i]))
        hist.setdefault(key, []).append((float(cd_tr[i]), raw))

    for i in range(len(X_te)):
        key = (g_te[i], cyc_te[i])
        raw = X_te[i, idx].copy()
        X_te[i, idx] = raw - _med_for(key, g_te[i], float(cd_te[i]))
        hist.setdefault(key, []).append((float(cd_te[i]), raw))
    return X_tr, X_te


def cluster_bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = TEST_CLUSTER_BOOT,
    seed: int = 0,
) -> dict[str, float]:
    """Patient-cluster bootstrap CI for macro F1 on the test set."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    pats = np.unique(groups)
    if len(pats) < 2:
        f = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return {"f1_boot_mean": f, "f1_boot_p025": f, "f1_boot_p975": f}
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        boot_pats = rng.choice(pats, size=len(pats), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == p) for p in boot_pats])
        scores.append(float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)))
    arr = np.asarray(scores)
    return {
        "f1_boot_mean": float(arr.mean()),
        "f1_boot_p025": float(np.percentile(arr, 2.5)),
        "f1_boot_p975": float(np.percentile(arr, 97.5)),
    }


def ov_hat_active(ov_hat: np.ndarray) -> bool:
    return float(np.std(np.asarray(ov_hat, dtype=np.float64))) > 1e-8


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    g = np.array(list("AABBCAABBCBC"))
    a, b = split_fit_val_idx(y, g, min_n=4)
    assert not set(g[a]) & set(g[b]), "group leak"
    print("protocol_pdca self-check OK")
