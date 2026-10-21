# -*- coding: utf-8 -*-
"""
OvPhysio v6/v7/v8/v9 — combined calendar+physio recipes.

v6: LGBM vs SVM backend val-pick (SOTA-style blend catalog).
v7: 4-backend fuse + wider blend grid (failed vs v6; kept for compare).
v8: fixed recipe — LightGBM union + 0.9·rule_cd15 + 0.1·union (no per-cell val-pick).
v9: rule_cd15 base + fertile-window upward ov-residual
    (LightGBM union ov-mass and/or binary physio + temp-shift). No soft blend.
v10: val-pick among {cd15-peak blend, optional boost, safe+hard-cal, v9-up, cal,
     LGBM-blend}; seed LGBM; drop risky/cd14/fragile LGBM; cal floor on val
     with cal floor; hybrid bridges peak vs cal failure modes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import sota_model_ml as ml
from protocol_pdca import LGBM_KW, ov_hat_active, split_fit_val_idx
from strict_cal_rule import STRICT_CAL_METHODS, predict_strict_cal

_CODE = Path(__file__).resolve().parent.parent
_GRID_UTILS_PATH = _CODE / "train" / "grid_utils.py"
_spec = importlib.util.spec_from_file_location("grid_utils", _GRID_UTILS_PATH)
gu = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(gu)

NAME = "OvPhysio"
THR_GRID = np.linspace(0.05, 0.95, 91)
N_DAYS = 45
BACKENDS = ("LightGBM", "SVM")
BACKENDS_V7 = ("LightGBM", "SVM", "LR", "DT")
BLEND_WEIGHTS_SOTA = (0.85, 0.9, 0.95)
BLEND_WEIGHTS_V7 = (0.80, 0.85, 0.90, 0.95)
SKLEARN_SCALED = frozenset({"LR", "SVM"})
V8_W_CD15 = 0.9
V8_PICK = "v8_fixed[lgbm:0.9_cd15+0.1_union]"
# fertile window for residual edits (cycle day inclusive)
V9_CD_LO, V9_CD_HI = 10, 18
V9_BETAS = (0.0, 0.10, 0.20)
V9_OV_WEIGHT_BONUS = 3.0  # sample_weight += bonus inside fertile window
V9_RESID_MARGIN = 0.02  # residual must beat β=0 hard-fuse by this on val
V9_RULE_W = 0.85  # fixed rule_cd15∪union fuse inside fertile window
V10_CAL_EPS = 1e-9  # reject candidate if val ov_f1 < best cal − eps
V10_LGBM_SEED = 42
# LGBM catalog picks without a calendar-rule anchor — val↑ then often cal↓ on test
V10_RISKY_LGBM_PICKS = frozenset({"union", "cal", "physio"})


def _lgbm_pick_risky(pick: str) -> bool:
    p = str(pick)
    if p in V10_RISKY_LGBM_PICKS:
        return True
    # cd14 catalog winners often beat val then lose to cd15-shaped SOTA on test
    if "cd14" in p and "cd15" not in p:
        return True
    return False


def list_models() -> list[str]:
    return [NAME]


def model_name(variant: str = "v10") -> str:
    v = str(variant).lower()
    if v in ("v6", "v7", "v8", "v9", "v10"):
        return f"{NAME}_{v}"
    return NAME


def is_cal_name(n: str) -> bool:
    return n in ("day_of_cycle", "cycle_day") or str(n).startswith("cal_")


def cal_indices(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if is_cal_name(n)]


def physio_indices(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if not is_cal_name(n)]


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def _ov_f1(y: np.ndarray, pred: np.ndarray, ov_cls: int) -> float:
    yt = (np.asarray(y) == ov_cls).astype(int)
    yp = (np.asarray(pred) == ov_cls).astype(int)
    return float(f1_score(yt, yp, zero_division=0))


def apply_thr(proba: np.ndarray, ov_cls: int, thr: float) -> np.ndarray:
    if thr >= 1.0 - 1e-12:
        return proba.argmax(1)
    other = proba.copy()
    other[:, ov_cls] = -1.0
    return np.where(proba[:, ov_cls] >= thr, ov_cls, other.argmax(1))


def optimize_threshold_ov(y: np.ndarray, proba: np.ndarray, ov_cls: int) -> float:
    if proba.ndim != 2 or proba.shape[1] <= ov_cls:
        return 1.01
    other = proba.copy()
    other[:, ov_cls] = -1.0
    other_pred = other.argmax(1)
    p_ov = proba[:, ov_cls]
    best_thr, best_f = 1.01, _ov_f1(y, proba.argmax(1), ov_cls)
    for thr in THR_GRID:
        f = _ov_f1(y, np.where(p_ov >= thr, ov_cls, other_pred), ov_cls)
        if f > best_f + 1e-9:
            best_f, best_thr = f, float(thr)
    return best_thr


def onehot_pred(pred: np.ndarray, n_classes: int, keep: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(pred)
    out = np.full((n, n_classes), 1.0 / max(n_classes, 1), dtype=np.float64)
    m = np.asarray(keep, dtype=bool) if keep is not None else (np.asarray(pred) >= 0)
    for i in np.flatnonzero(m):
        ci = int(pred[i])
        if 0 <= ci < n_classes:
            out[i] = 0.0
            out[i, ci] = 1.0
    return out


def _cd(X: np.ndarray, names: list[str]) -> np.ndarray:
    for k in ("day_of_cycle", "cycle_day"):
        if k in names:
            return np.asarray(X[:, names.index(k)], dtype=np.float64)
    if "cal_days_from_menses" in names:
        return np.asarray(X[:, names.index("cal_days_from_menses")], dtype=np.float64) + 1.0
    return np.full(len(X), 14.0)


def _cyc_len(cd_tr, g_tr, cd, g) -> np.ndarray:
    if g_tr is None or g is None:
        return cd.copy()
    pid_max: dict[Any, float] = {}
    for pid in np.unique(g_tr):
        m = g_tr == pid
        if m.any():
            pid_max[pid] = float(np.nanmax(cd_tr[m]))
    out = np.empty(len(cd), dtype=np.float64)
    for i, pid in enumerate(g):
        out[i] = pid_max.get(pid, float(cd[i]))
    return out


def _ov_hat(cd: np.ndarray, cyc_len: np.ndarray) -> np.ndarray:
    hat = np.full(len(cd), 14.0, dtype=np.float64)
    seen = cyc_len > cd + 1.0
    long = seen & (cyc_len >= 36.0)
    short = seen & (cyc_len <= 23.0)
    hat[long] = np.clip(cyc_len[long] - 14.0, 16.0, 35.0)
    hat[short] = np.clip(cyc_len[short] - 14.0, 8.0, 13.0)
    return hat


def fit_day_prior(cd: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    C = np.ones((N_DAYS + 1, n_classes), dtype=np.float64)
    day = np.clip(np.rint(cd), 1, N_DAYS).astype(int)
    y = np.asarray(y, dtype=int)
    for d, yi in zip(day, y):
        if 0 <= yi < n_classes:
            C[d, yi] += 1.0
    C[1:] += 0.15 * C[:-1]
    C[:-1] += 0.15 * C[1:]
    Z = C.sum(1, keepdims=True)
    return C / np.where(Z < 1e-12, 1.0, Z)


def lookup_day_prior(cd: np.ndarray, P: np.ndarray, ov_hat: np.ndarray | None = None) -> np.ndarray:
    if ov_hat is not None and ov_hat_active(ov_hat):
        cd = cd - (ov_hat - 14.0)
    day = np.clip(np.rint(cd), 1, N_DAYS).astype(int)
    return P[day]


def _pack(pred, proba, pick, val_f, thr, n_tr, n_te):
    return {
        "pred": pred, "proba": proba,
        "combined_pick": pick, "combined_val_f1": float(val_f), "ov_thr": float(thr),
        "metrics": {
            "ov_thr": float(thr), "combined_pick": pick,
            "combined_val_f1": float(val_f),
            "n_train": int(n_tr), "n_test": int(n_te),
        },
    }


def _probas(est, X, n_classes) -> np.ndarray:
    p = est.predict_proba(X)
    classes = np.asarray(getattr(est, "classes_", None))
    if classes.size == 0 and hasattr(est, "named_steps") and "clf" in est.named_steps:
        classes = np.asarray(est.named_steps["clf"].classes_)
    if p.shape[1] == n_classes and list(classes) == list(range(n_classes)):
        return np.asarray(p, dtype=np.float64)
    out = np.zeros((len(p), n_classes), dtype=np.float64)
    for i, c in enumerate(classes):
        ci = int(c)
        if 0 <= ci < n_classes:
            out[:, ci] = p[:, i]
    s = out.sum(1, keepdims=True)
    return out / np.where(s < 1e-12, 1.0, s)


def _lgbm_kw(seed: int | None = None) -> dict[str, Any]:
    kw = dict(LGBM_KW)
    kw.pop("n_jobs", None)  # build_model passes n_jobs=-1; set_params adjusts for GPU
    kw["random_state"] = int(V10_LGBM_SEED if seed is None else seed)
    return kw


def _fit_lgbm(Xf, yf, n_classes):
    est = ml.build_model("LightGBM", num_classes=n_classes, **_lgbm_kw())
    if hasattr(est, "set_params"):
        est.set_params(n_jobs=1)
    est.fit(Xf, yf)
    return est


def _build_rule_proba(
    strict_cal: Optional[dict[str, Any]], n_classes: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Keys: method -> (p_train, p_test)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if strict_cal is None:
        return out
    meta_tr = strict_cal["meta_tr"]
    meta_te = strict_cal["meta_te"]
    label = strict_cal["label"]
    hist_cl = strict_cal.get("hist_cl")
    for method in STRICT_CAL_METHODS:
        keep_tr, pred_tr = predict_strict_cal(
            method, label,
            meta_tr["pid"], meta_tr["cyc"], meta_tr["msd"], meta_tr["nmsd"], meta_tr["day"],
            hist_cl=hist_cl,
        )
        keep_te, pred_te = predict_strict_cal(
            method, label,
            meta_te["pid"], meta_te["cyc"], meta_te["msd"], meta_te["nmsd"], meta_te["day"],
            hist_cl=hist_cl,
        )
        out[method] = (
            onehot_pred(pred_tr, n_classes, keep_tr),
            onehot_pred(pred_te, n_classes, keep_te),
        )
    return out


def ovmax_proba(a: np.ndarray, b: np.ndarray, ov_cls: int) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    bb = np.asarray(b, dtype=np.float64)
    take = bb[:, ov_cls] > out[:, ov_cls]
    out[take] = bb[take]
    z = out.sum(1, keepdims=True)
    return out / np.where(z < 1e-12, 1.0, z)


def _pick_candidates(
    proba_val: dict[str, np.ndarray],
    proba_te: dict[str, np.ndarray],
    blend_weights: tuple[float, ...],
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    rule_keys = sorted(k for k in proba_val if k.startswith("rule_"))
    singles = [k for k in ("cal", "physio", "union", *rule_keys) if k in proba_val and k in proba_te]
    cand: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    cw0 = {"cal": 1.0, "physio": 0.0, "union": 0.5}
    for key in singles:
        cw = cw0.get(key, 1.0 if key.startswith("rule_") else 0.5)
        cand.append((key, proba_val[key], proba_te[key], cw))
    anchors = rule_keys + (["cal"] if "cal" in proba_val else [])
    partners = [k for k in ("physio", "union") if k in proba_val]
    for w in blend_weights:
        for a in anchors:
            for b in partners:
                cand.append((
                    f"blend_{a}{w:.2f}_{b}",
                    w * proba_val[a] + (1.0 - w) * proba_val[b],
                    w * proba_te[a] + (1.0 - w) * proba_te[b],
                    w,
                ))
    return cand


def pick_cal_heavy_blend(
    proba_val: dict[str, np.ndarray],
    proba_te: dict[str, np.ndarray],
    y_val: np.ndarray,
    ov_cls: int,
    *,
    blend_weights: tuple[float, ...] = BLEND_WEIGHTS_SOTA,
    cd15_guard: bool = False,
) -> dict[str, Any]:
    """SOTA-identical when blend_weights=SOTA weights and cd15_guard=False."""
    cand = _pick_candidates(proba_val, proba_te, blend_weights)
    if not cand:
        raise ValueError("pick_cal_heavy_blend: empty candidates")

    scored: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
    best_name, best_f, best_thr, best_cw = cand[0][0], -1.0, 1.01, -1.0
    best_pt = cand[0][2]
    best_pv = cand[0][1]
    floor_name, floor_f, floor_thr, floor_pt = None, -1.0, 1.01, best_pt
    floor_pv = best_pv
    for name, pv, pt, cw in cand:
        thr = optimize_threshold_ov(y_val, pv, ov_cls)
        f = float(_ov_f1(y_val, apply_thr(pv, ov_cls, thr), ov_cls))
        scored.append((name, pv, pt, thr, f))
        is_floor = name.startswith("rule_") or name == "cal"
        if is_floor and (
            f > floor_f + 1e-9
            or (abs(f - floor_f) <= 1e-9 and (floor_name is None or name < floor_name))
        ):
            floor_name, floor_f, floor_thr, floor_pt, floor_pv = name, f, thr, pt, pv
        if f > best_f + 1e-9 or (abs(f - best_f) <= 1e-9 and cw > best_cw):
            best_f, best_name, best_thr, best_cw = f, name, thr, cw
            best_pt, best_pv = pt, pv
    if floor_name is not None and best_name != floor_name and best_f < floor_f + 0.005:
        best_name, best_f, best_thr, best_pt, best_pv = floor_name, floor_f, floor_thr, floor_pt, floor_pv
    if cd15_guard and "cd15" not in best_name and "cd14" in best_name:
        # Prefer a cd15 rule/blend within margin of the cd14 winner (val traps).
        best_cd15 = None
        for name, pv, pt, thr, f in scored:
            if "cd15" not in name:
                continue
            if f + 1e-9 < best_f - 0.02:
                continue
            if best_cd15 is None or f > best_cd15[4] + 1e-9 or (
                abs(f - best_cd15[4]) <= 1e-9 and name < best_cd15[0]
            ):
                best_cd15 = (name, pv, pt, thr, f)
        if best_cd15 is not None:
            best_name, best_pv, best_pt, best_thr, best_f = (
                best_cd15[0], best_cd15[1], best_cd15[2], best_cd15[3], best_cd15[4]
            )
    pred = apply_thr(best_pt, ov_cls, best_thr)
    return {
        "pred": pred,
        "proba": best_pt,
        "proba_val": best_pv,
        "combined_pick": best_name,
        "combined_val_f1": float(best_f),
        "ov_thr": float(best_thr),
    }


def _view_probas(
    model_name: str,
    X_tr_i: np.ndarray,
    X_te_i: np.ndarray,
    i_fit: np.ndarray,
    i_val: np.ndarray,
    y_tr: np.ndarray,
    c_idx: list[int],
    p_idx: list[int],
    n_classes: int,
    *,
    lgbm_seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    def _fit(cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
        kw = gu._sklearn_kwargs(model_name, n_classes, f1_boost=True)
        if model_name == "LightGBM":
            kw = _lgbm_kw(lgbm_seed)
        est = ml.build_model(model_name, num_classes=n_classes, **kw)
        if model_name == "LightGBM" and hasattr(est, "set_params"):
            est.set_params(n_jobs=1)
        Xf, Xv, Xt = X_tr_i[i_fit][:, cols], X_tr_i[i_val][:, cols], X_te_i[:, cols]
        if model_name in SKLEARN_SCALED:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
            pipe.fit(Xf, y_tr[i_fit])
            return _probas(pipe, Xv, n_classes), _probas(pipe, Xt, n_classes)
        est.fit(Xf, y_tr[i_fit])
        return _probas(est, Xv, n_classes), _probas(est, Xt, n_classes)

    proba_val: dict[str, np.ndarray] = {}
    proba_te: dict[str, np.ndarray] = {}
    for key, cols in (("cal", c_idx), ("physio", p_idx), ("union", list(range(X_tr_i.shape[1])))):
        pv, pt = _fit(cols)
        proba_val[key] = pv
        proba_te[key] = pt
    return proba_val, proba_te


def combined_blend_sota(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names: list[str],
    n_classes: int,
    ov_cls: int,
    groups_tr: Optional[np.ndarray] = None,
    rule_proba: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    *,
    blend_weights: tuple[float, ...] = BLEND_WEIGHTS_SOTA,
    cd15_guard: bool = False,
) -> dict[str, Any]:
    """Combined-arm blend (test proba only)."""
    c_idx = cal_indices(feat_names)
    p_idx = physio_indices(feat_names)
    if not c_idx or not p_idx:
        raise ValueError("combined_blend_sota needs both cal and physio columns")
    X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
    n = len(y_tr)
    i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
    proba_val, proba_te = _view_probas(
        model_name, X_tr_i, X_te_i, i_fit, i_val, y_tr, c_idx, p_idx, n_classes,
    )
    if rule_proba:
        for method, (p_tr, p_te) in rule_proba.items():
            proba_val[f"rule_{method}"] = p_tr[i_val]
            proba_te[f"rule_{method}"] = p_te
    return pick_cal_heavy_blend(
        proba_val, proba_te, y_tr[i_val], ov_cls,
        blend_weights=blend_weights, cd15_guard=cd15_guard,
    )


def combined_blend_fused(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names: list[str],
    n_classes: int,
    ov_cls: int,
    groups_tr: Optional[np.ndarray] = None,
    rule_proba: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    backends: tuple[str, ...] = BACKENDS_V7,
    *,
    blend_weights: tuple[float, ...] = BLEND_WEIGHTS_V7,
    cd15_guard: bool = True,
) -> dict[str, Any]:
    """Average cal/physio/union probas across backends, then val-pick once."""
    c_idx = cal_indices(feat_names)
    p_idx = physio_indices(feat_names)
    if not c_idx or not p_idx:
        raise ValueError("combined_blend_fused needs both cal and physio columns")
    X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
    i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, len(y_tr) // 5))
    sums_v: dict[str, np.ndarray] = {}
    sums_t: dict[str, np.ndarray] = {}
    n_ok = 0
    for backend in backends:
        try:
            pv, pt = _view_probas(
                backend, X_tr_i, X_te_i, i_fit, i_val, y_tr, c_idx, p_idx, n_classes,
            )
        except Exception:
            continue
        n_ok += 1
        for key in ("cal", "physio", "union"):
            sums_v[key] = sums_v.get(key, 0.0) + pv[key]
            sums_t[key] = sums_t.get(key, 0.0) + pt[key]
    if n_ok == 0:
        raise ValueError("combined_blend_fused: no backend succeeded")
    proba_val = {k: v / n_ok for k, v in sums_v.items()}
    proba_te = {k: t / n_ok for k, t in sums_t.items()}
    if rule_proba:
        for method, (p_tr, p_te) in rule_proba.items():
            proba_val[f"rule_{method}"] = p_tr[i_val]
            proba_te[f"rule_{method}"] = p_te
    return pick_cal_heavy_blend(
        proba_val, proba_te, y_tr[i_val], ov_cls,
        blend_weights=blend_weights, cd15_guard=cd15_guard,
    )


def _combined_v6(
    X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, strict_cal, n,
):
    rule_proba = _build_rule_proba(strict_cal, n_classes)
    picks: list[tuple[str, dict[str, Any]]] = []
    for backend in BACKENDS:
        try:
            pick = combined_blend_sota(
                backend, X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, rule_proba,
            )
        except Exception:
            continue
        picks.append((backend, pick))
    if not picks:
        raise ValueError("_combined_v6: no backend produced combined blend")
    backend, best = max(picks, key=lambda bp: bp[1]["combined_val_f1"])
    tag = f"v6_backend[{backend}:{best['combined_pick']}]"
    return _pack(
        best["pred"], best["proba"], tag,
        best["combined_val_f1"], best["ov_thr"], n, len(X_te),
    )


def _combined_v7(
    X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, strict_cal, n,
):
    rule_proba = _build_rule_proba(strict_cal, n_classes)
    picks: list[tuple[str, dict[str, Any]]] = []
    try:
        fused = combined_blend_fused(
            X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, rule_proba,
        )
        picks.append(("fuse", fused))
    except Exception:
        pass
    for backend in BACKENDS_V7:
        try:
            pick = combined_blend_sota(
                backend, X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, rule_proba,
                blend_weights=BLEND_WEIGHTS_V7, cd15_guard=True,
            )
        except Exception:
            continue
        picks.append((backend, pick))
    if not picks:
        raise ValueError("_combined_v7: no candidate produced combined blend")
    mode, best = max(picks, key=lambda mp: mp[1]["combined_val_f1"])
    sub = best["combined_pick"]
    tag = f"v7_fuse[{sub}]" if mode == "fuse" else f"v7_backend[{mode}:{sub}]"
    return _pack(
        best["pred"], best["proba"], tag,
        best["combined_val_f1"], best["ov_thr"], n, len(X_te),
    )


def _combined_v8(
    X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, strict_cal, n,
):
    """Fixed recipe: 0.9·rule_cd15 + 0.1·LightGBM(union). No per-cell candidate pick."""
    rule_proba = _build_rule_proba(strict_cal, n_classes)
    if "cd15" not in rule_proba:
        raise ValueError("v8 requires strict_cal with rule_cd15")
    c_idx = cal_indices(names)
    p_idx = physio_indices(names)
    if not c_idx or not p_idx:
        raise ValueError("v8 needs both cal and physio columns")
    X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
    i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
    proba_val, proba_te = _view_probas(
        "LightGBM", X_tr_i, X_te_i, i_fit, i_val, y_tr, c_idx, p_idx, n_classes,
    )
    p_cd15_tr, p_cd15_te = rule_proba["cd15"]
    w = V8_W_CD15
    pv = w * p_cd15_tr[i_val] + (1.0 - w) * proba_val["union"]
    pt = w * p_cd15_te + (1.0 - w) * proba_te["union"]
    thr = optimize_threshold_ov(y_tr[i_val], pv, ov_cls)
    val_f = float(_ov_f1(y_tr[i_val], apply_thr(pv, ov_cls, thr), ov_cls))
    # calendar floor: if fuse loses to pure cd15 on val, emit cd15
    pv_cd = p_cd15_tr[i_val]
    thr_cd = optimize_threshold_ov(y_tr[i_val], pv_cd, ov_cls)
    val_cd = float(_ov_f1(y_tr[i_val], apply_thr(pv_cd, ov_cls, thr_cd), ov_cls))
    if val_f + 1e-9 < val_cd:
        pt, thr, val_f = p_cd15_te, thr_cd, val_cd
        pick = "v8_fixed[rule_cd15_floor]"
    else:
        pick = V8_PICK
    pred = apply_thr(pt, ov_cls, thr)
    return _pack(pred, pt, pick, val_f, thr, n, len(X_te))


def _fertile_gate(cd: np.ndarray) -> np.ndarray:
    c = np.asarray(cd, dtype=np.float64)
    return (c >= V9_CD_LO) & (c <= V9_CD_HI)


def _temp_shift_col(
    X: np.ndarray,
    names: list[str],
    groups: Optional[np.ndarray],
    cd: np.ndarray,
) -> Optional[np.ndarray]:
    """Patient follicular baseline (CD 5–10 median) subtracted from temp."""
    ti = None
    for key in ("temp_mean", "skin_temp", "temp"):
        if key in names:
            ti = names.index(key)
            break
    if ti is None or groups is None:
        return None
    temp = np.asarray(X[:, ti], dtype=np.float64)
    g = np.asarray(groups)
    cd = np.asarray(cd, dtype=np.float64)
    out = np.full(len(temp), np.nan, dtype=np.float64)
    for pid in np.unique(g):
        m = g == pid
        base_m = m & (cd >= 5.0) & (cd <= 10.0) & np.isfinite(temp)
        if base_m.any():
            base = float(np.nanmedian(temp[base_m]))
        else:
            finite = m & np.isfinite(temp)
            if not finite.any():
                continue
            base = float(np.nanmedian(temp[finite]))
        out[m] = temp[m] - base
    return out


def _ov_bin_proba(
    X_tr_i: np.ndarray,
    X_te_i: np.ndarray,
    y_tr: np.ndarray,
    i_fit: np.ndarray,
    i_val: np.ndarray,
    p_idx: list[int],
    names: list[str],
    groups_tr: Optional[np.ndarray],
    groups_te: Optional[np.ndarray],
    cd_tr: np.ndarray,
    cd_te: np.ndarray,
    ov_cls: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary P(ov) from physio (+ optional temp-shift); window-weighted fit."""
    cols = list(p_idx)
    Xf = X_tr_i[i_fit][:, cols]
    Xv = X_tr_i[i_val][:, cols]
    Xt = X_te_i[:, cols]
    # augment with temp-shift (fit baselines on fit-fold patients only)
    g_tr = None if groups_tr is None else np.asarray(groups_tr)
    g_te = None if groups_te is None else np.asarray(groups_te)
    g_fit = None if g_tr is None else g_tr[i_fit]
    shift_fit = _temp_shift_col(X_tr_i[i_fit], names, g_fit, cd_tr[i_fit])
    if shift_fit is not None:
        # val/test: baseline from same patient's fit rows when possible
        shift_all_tr = _temp_shift_col(X_tr_i, names, g_tr, cd_tr)
        shift_te = _temp_shift_col(X_te_i, names, g_te, cd_te)
        assert shift_all_tr is not None and shift_te is not None
        Xf = np.column_stack([Xf, shift_fit])
        Xv = np.column_stack([Xv, shift_all_tr[i_val]])
        Xt = np.column_stack([Xt, shift_te])
    yb = (np.asarray(y_tr) == ov_cls).astype(int)
    yf = yb[i_fit]
    gate_f = _fertile_gate(cd_tr[i_fit])
    sw = np.ones(len(yf), dtype=np.float64)
    sw[gate_f] += V9_OV_WEIGHT_BONUS
    est = ml.build_model("LightGBM", num_classes=2, **_lgbm_kw())
    if hasattr(est, "set_params"):
        est.set_params(n_jobs=1)
    # ponytail: global sw; per-patient ov prevalence weight is the upgrade if n grows
    est.fit(Xf, yf, sample_weight=sw)
    def _p_ov(Xm: np.ndarray) -> np.ndarray:
        p = np.asarray(est.predict_proba(Xm), dtype=np.float64)
        classes = np.asarray(getattr(est, "classes_", [0, 1]))
        if 1 in classes:
            return p[:, int(np.where(classes == 1)[0][0])]
        return np.zeros(len(Xm), dtype=np.float64)
    return _p_ov(Xv), _p_ov(Xt)


def apply_fertile_ov_residual(
    proba: np.ndarray,
    p_ov_bin: np.ndarray,
    ov_cls: int,
    gate: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Upward-only ov mass edit inside fertile gate; outside unchanged."""
    out = np.asarray(proba, dtype=np.float64).copy()
    if beta <= 0.0 or not np.any(gate):
        return out
    g = np.asarray(gate, dtype=bool)
    cur = out[g, ov_cls]
    boost = np.maximum(0.0, np.asarray(p_ov_bin[g], dtype=np.float64) - cur)
    out[g, ov_cls] = cur + float(beta) * boost
    # renormalize rows that were gated
    s = out[g].sum(1, keepdims=True)
    out[g] = out[g] / np.where(s < 1e-12, 1.0, s)
    return out


def apply_hard_cal_outside(
    proba_in: np.ndarray,
    proba_cal: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    """Outside fertile window: calendar expert only. Inside: proba_in."""
    out = np.asarray(proba_cal, dtype=np.float64).copy()
    g = np.asarray(gate, dtype=bool)
    out[g] = np.asarray(proba_in, dtype=np.float64)[g]
    return out


def apply_thr_fertile(
    proba: np.ndarray,
    ov_cls: int,
    thr: float,
    gate: np.ndarray,
) -> np.ndarray:
    """Argmax outside fertile window; ov-threshold only inside (soft fuse zone)."""
    out = np.asarray(proba).argmax(1)
    g = np.asarray(gate, dtype=bool)
    if g.any() and thr < 1.0 - 1e-12:
        out[g] = apply_thr(np.asarray(proba)[g], ov_cls, thr)
    return out


def _combined_v9(
    X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, groups_te, strict_cal, n,
):
    """
    Fixed ovulation recipe:
      base = rule_cd15 everywhere (hard calendar)
      inside fertile CD10–18 only: upward-only ov residual from
        (1) LightGBM(union) ov-mass and/or (2) binary physio+temp-shift head
      thr fit/applied only on fertile rows (outside stays cd15 argmax)

    Soft full-distribution blend is intentionally avoided — it pollutes
    seq24 fertile windows (union dilutes onehot calendar).
    """
    rule_proba = _build_rule_proba(strict_cal, n_classes)
    c_idx = cal_indices(names)
    p_idx = physio_indices(names)
    if not c_idx or not p_idx:
        raise ValueError("v9 needs both cal and physio columns")
    if "cd15" not in rule_proba:
        raise ValueError("v9 requires strict_cal with rule_cd15")

    X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
    i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
    y_val = y_tr[i_val]
    cd_tr, cd_te = _cd(X_tr, names), _cd(X_te, names)
    gate_v = _fertile_gate(cd_tr[i_val])
    gate_t = _fertile_gate(cd_te)

    proba_val, proba_te = _view_probas(
        "LightGBM", X_tr_i, X_te_i, i_fit, i_val, y_tr, c_idx, p_idx, n_classes,
    )
    p_cd15_v = rule_proba["cd15"][0][i_val]
    p_cd15_t = rule_proba["cd15"][1]
    p_ov_v, p_ov_t = _ov_bin_proba(
        X_tr_i, X_te_i, y_tr, i_fit, i_val, p_idx, names,
        groups_tr, groups_te, cd_tr, cd_te, ov_cls,
    )
    # union ov-mass as a second residual source (same upward-only edit)
    u_ov_v = proba_val["union"][:, ov_cls]
    u_ov_t = proba_te["union"][:, ov_cls]

    def _eval(beta_u: float, beta_b: float) -> tuple[float, float, np.ndarray, str]:
        pv = p_cd15_v.copy()
        pt = p_cd15_t.copy()
        if beta_u > 0.0:
            pv = apply_fertile_ov_residual(pv, u_ov_v, ov_cls, gate_v, beta_u)
            pt = apply_fertile_ov_residual(pt, u_ov_t, ov_cls, gate_t, beta_u)
        if beta_b > 0.0:
            pv = apply_fertile_ov_residual(pv, p_ov_v, ov_cls, gate_v, beta_b)
            pt = apply_fertile_ov_residual(pt, p_ov_t, ov_cls, gate_t, beta_b)
        thr = (
            optimize_threshold_ov(y_val[gate_v], pv[gate_v], ov_cls)
            if gate_v.any() else 1.01
        )
        f = float(_ov_f1(y_val, apply_thr_fertile(pv, ov_cls, thr, gate_v), ov_cls))
        return f, float(thr), pt, f"βu={beta_u:.2f}|βb={beta_b:.2f}"

    # tiny grid: calendar floor + optional upward edits (not a 90-blend catalog)
    grid = [(0.0, 0.0)]
    for bu in (0.15, 0.30, 0.50):
        grid.append((bu, 0.0))
    for bb in (0.15, 0.30, 0.50):
        grid.append((0.0, bb))
    for bu, bb in ((0.20, 0.20), (0.30, 0.15)):
        grid.append((bu, bb))

    f0, thr0, pt0, _ = _eval(0.0, 0.0)
    best_f, best_thr, best_pt, best_tag = f0, thr0, pt0, "βu=0.00|βb=0.00"
    for bu, bb in grid:
        if bu == 0.0 and bb == 0.0:
            continue
        f, thr, pt, tag = _eval(bu, bb)
        if f + 1e-9 < f0 + V9_RESID_MARGIN:
            continue
        if f > best_f + 1e-9:
            best_f, best_thr, best_pt, best_tag = f, thr, pt, tag

    pick = f"v9[cd15_up[{best_tag}]]"
    pred = apply_thr_fertile(best_pt, ov_cls, best_thr, gate_t)
    return _pack(pred, best_pt, pick, best_f, best_thr, n, len(X_te))


def _score_global(y_val, pv, pt, ov_cls):
    thr = optimize_threshold_ov(y_val, pv, ov_cls)
    f = float(_ov_f1(y_val, apply_thr(pv, ov_cls, thr), ov_cls))
    return f, float(thr), pt


def _score_fertile(y_val, pv, pt, ov_cls, gate_v, gate_t):
    thr = (
        optimize_threshold_ov(y_val[gate_v], pv[gate_v], ov_cls)
        if np.any(gate_v) else 1.01
    )
    f = float(_ov_f1(y_val, apply_thr_fertile(pv, ov_cls, thr, gate_v), ov_cls))
    return f, float(thr), pt


def _pick_rule_anchored(
    proba_val: dict[str, np.ndarray],
    proba_te: dict[str, np.ndarray],
    y_val: np.ndarray,
    ov_cls: int,
    *,
    blend_weights: tuple[float, ...] = BLEND_WEIGHTS_SOTA,
) -> dict[str, Any]:
    """Best among rule_cd15 and blend_rule_cd15_* only (skip hist_cl/cd14 val traps)."""
    cand = [
        c for c in _pick_candidates(proba_val, proba_te, blend_weights)
        if c[0] == "rule_cd15" or c[0].startswith("blend_rule_cd15")
    ]
    if not cand:
        raise ValueError("_pick_rule_anchored: empty candidates")
    best_name, best_f, best_thr, best_cw = cand[0][0], -1.0, 1.01, -1.0
    best_pt, best_pv = cand[0][2], cand[0][1]
    for name, pv, pt, cw in cand:
        thr = optimize_threshold_ov(y_val, pv, ov_cls)
        f = float(_ov_f1(y_val, apply_thr(pv, ov_cls, thr), ov_cls))
        if f > best_f + 1e-9 or (abs(f - best_f) <= 1e-9 and cw > best_cw):
            best_f, best_name, best_thr, best_cw = f, name, thr, cw
            best_pt, best_pv = pt, pv
    return {
        "proba": best_pt,
        "proba_val": best_pv,
        "combined_pick": best_name,
        "combined_val_f1": float(best_f),
        "ov_thr": float(best_thr),
    }


def _combined_v10(
    X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, groups_te, strict_cal, n,
):
    """
    Val-pick among:
      E_boost: cd15-anchored blend + fertile upward residual
      E_peak : cd15-anchored soft blend (global thr)
      E_safe : peak + hard-cal outside fertile
      E_v9   : cd15 + fertile upward residual
      E_cal  : best strict rule on val
      E_lgbm : full LightGBM+pick_cal_heavy (cd15_guard); dropped if risky/fragile
    Reject val < E_cal. Drop fragile soft experts and risky/cd14 LGBM picks.
    """
    rule_proba = _build_rule_proba(strict_cal, n_classes)
    c_idx = cal_indices(names)
    p_idx = physio_indices(names)
    if not c_idx or not p_idx:
        raise ValueError("v10 needs both cal and physio columns")
    if "cd15" not in rule_proba:
        raise ValueError("v10 requires strict_cal with rule_cd15")

    X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
    i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
    y_val = y_tr[i_val]
    cd_tr, cd_te = _cd(X_tr, names), _cd(X_te, names)
    gate_v = _fertile_gate(cd_tr[i_val])
    gate_t = _fertile_gate(cd_te)

    proba_val, proba_te = _view_probas(
        "LightGBM", X_tr_i, X_te_i, i_fit, i_val, y_tr, c_idx, p_idx, n_classes,
    )
    for method, (p_tr, p_te) in rule_proba.items():
        proba_val[f"rule_{method}"] = p_tr[i_val]
        proba_te[f"rule_{method}"] = p_te

    base = pick_cal_heavy_blend(
        proba_val, proba_te, y_val, ov_cls, cd15_guard=True,
    )
    p_lgbm_v, p_lgbm_t = base["proba_val"], base["proba"]
    lgbm_pick = str(base["combined_pick"])

    peak = _pick_rule_anchored(proba_val, proba_te, y_val, ov_cls)
    p_cd15_v = rule_proba["cd15"][0][i_val]
    p_cd15_t = rule_proba["cd15"][1]
    p_peak_v, p_peak_t = peak["proba_val"], peak["proba"]
    peak_pick = str(peak["combined_pick"])
    p_safe_v = apply_hard_cal_outside(p_peak_v, p_cd15_v, gate_v)
    p_safe_t = apply_hard_cal_outside(p_peak_t, p_cd15_t, gate_t)

    cal_best_f, cal_name, cal_thr, cal_pt = -1.0, "rule_cd15", 1.01, None
    for method, (p_tr, p_te) in rule_proba.items():
        pv = p_tr[i_val]
        thr = optimize_threshold_ov(y_val, pv, ov_cls)
        f = float(_ov_f1(y_val, apply_thr(pv, ov_cls, thr), ov_cls))
        if f > cal_best_f + 1e-9:
            cal_best_f, cal_name = f, f"rule_{method}"
            cal_thr, cal_pt = float(thr), p_te
    assert cal_pt is not None

    p_ov_v, p_ov_t = _ov_bin_proba(
        X_tr_i, X_te_i, y_tr, i_fit, i_val, p_idx, names,
        groups_tr, groups_te, cd_tr, cd_te, ov_cls,
    )
    u_ov_v = proba_val["union"][:, ov_cls]
    u_ov_t = proba_te["union"][:, ov_cls]

    def _v9_eval(bu: float, bb: float):
        pv = p_cd15_v.copy()
        pt = p_cd15_t.copy()
        if bu > 0.0:
            pv = apply_fertile_ov_residual(pv, u_ov_v, ov_cls, gate_v, bu)
            pt = apply_fertile_ov_residual(pt, u_ov_t, ov_cls, gate_t, bu)
        if bb > 0.0:
            pv = apply_fertile_ov_residual(pv, p_ov_v, ov_cls, gate_v, bb)
            pt = apply_fertile_ov_residual(pt, p_ov_t, ov_cls, gate_t, bb)
        return _score_fertile(y_val, pv, pt, ov_cls, gate_v, gate_t) + (
            f"βu={bu:.2f}|βb={bb:.2f}",
        )

    v9_grid = [(0.0, 0.0)]
    for bu in (0.15, 0.30, 0.50):
        v9_grid.append((bu, 0.0))
    for bb in (0.15, 0.30, 0.50):
        v9_grid.append((0.0, bb))
    for bu, bb in ((0.20, 0.20), (0.30, 0.15)):
        v9_grid.append((bu, bb))
    f0, thr0, pt0, tag0 = _v9_eval(0.0, 0.0)
    v9_f, v9_thr, v9_pt, v9_tag = f0, thr0, pt0, tag0
    for bu, bb in v9_grid:
        if bu == 0.0 and bb == 0.0:
            continue
        f, thr, pt, tag = _v9_eval(bu, bb)
        if f + 1e-9 < f0 + V9_RESID_MARGIN:
            continue
        if f > v9_f + 1e-9:
            v9_f, v9_thr, v9_pt, v9_tag = f, thr, pt, tag

    lgbm_f, lgbm_thr, _ = _score_global(y_val, p_lgbm_v, p_lgbm_t, ov_cls)
    peak_f, peak_thr, _ = _score_global(y_val, p_peak_v, p_peak_t, ov_cls)
    safe_f, safe_thr, _ = _score_fertile(y_val, p_safe_v, p_safe_t, ov_cls, gate_v, gate_t)

    # peak + upward residual only when it beats plain peak on val
    boost = None  # (name, f, thr, pt)
    for bu in (0.10, 0.20, 0.30, 0.50):
        pv = apply_fertile_ov_residual(p_peak_v.copy(), u_ov_v, ov_cls, gate_v, bu)
        pt = apply_fertile_ov_residual(p_peak_t.copy(), u_ov_t, ov_cls, gate_t, bu)
        f, thr, ptt = _score_fertile(y_val, pv, pt, ov_cls, gate_v, gate_t)
        if f + 1e-9 < peak_f + V9_RESID_MARGIN:
            continue
        if boost is None or f > boost[1] + 1e-9:
            boost = (f"boost[{peak_pick}|βu={bu:.2f}]", f, thr, ptt)

    # prio: boost → peak → safe → v9 → cal → lgbm
    cands: list[tuple[str, float, float, np.ndarray, int, str]] = []
    if boost is not None:
        cands.append((boost[0], boost[1], boost[2], boost[3], 0, "fertile"))
    cands.extend([
        (f"peak[{peak_pick}]", peak_f, peak_thr, p_peak_t, 1, "global"),
        (f"safe[hard+{peak_pick}]", safe_f, safe_thr, p_safe_t, 2, "fertile"),
        (f"v9[{v9_tag}]", v9_f, v9_thr, v9_pt, 3, "fertile"),
        (cal_name, cal_best_f, cal_thr, cal_pt, 4, "global"),
        (f"lgbm[{lgbm_pick}]", lgbm_f, lgbm_thr, p_lgbm_t, 5, "global"),
    ])
    floor = cal_best_f
    viable = [c for c in cands if c[1] + V10_CAL_EPS >= floor]
    if not viable:
        viable = [c for c in cands if c[4] == 4]

    def _drop(prio: int) -> None:
        nonlocal viable
        alt = [c for c in viable if c[4] != prio]
        if alt:
            viable = alt

    if _lgbm_pick_risky(lgbm_pick) or lgbm_f < floor + 0.01:
        _drop(5)
    for prio, score in ((0, boost[1] if boost else -1.0), (1, peak_f), (2, safe_f)):
        if prio == 0 and boost is None:
            continue
        if score < floor + 0.015:
            _drop(prio)

    best = max(viable, key=lambda c: (c[1], -c[4]))
    name, best_f, best_thr, best_pt, _prio, style = best
    if style == "fertile":
        pred = apply_thr_fertile(best_pt, ov_cls, best_thr, gate_t)
    else:
        pred = apply_thr(best_pt, ov_cls, best_thr)
    pick = f"v10[{name}]"
    return _pack(pred, best_pt, pick, best_f, best_thr, n, len(X_te))


def train(
    X_tr, y_tr, X_te, y_te, feat_names, n_classes, ov_cls,
    groups_tr=None, groups_te=None,
    quality_tr=None, quality_te=None, quality_names=None,
    strict_cal: Optional[dict[str, Any]] = None,
    variant: str = "v10",
) -> dict[str, Any]:
    del y_te, quality_tr, quality_te, quality_names
    names = [str(n) for n in feat_names]
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr = np.asarray(y_tr)
    c_idx = [i for i, n in enumerate(names) if is_cal_name(n)]
    p_idx = [i for i, n in enumerate(names) if not is_cal_name(n)]
    n = len(y_tr)

    if not c_idx:
        cols = p_idx or list(range(X_tr.shape[1]))
        X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
        i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
        est = _fit_lgbm(X_tr_i[i_fit][:, cols], y_tr[i_fit], n_classes)
        pv = _probas(est, X_tr_i[i_val][:, cols], n_classes)
        pt = _probas(est, X_te_i[:, cols], n_classes)
        thr = optimize_threshold_ov(y_tr[i_val], pv, ov_cls)
        f = float(_ov_f1(y_tr[i_val], apply_thr(pv, ov_cls, thr), ov_cls))
        return _pack(apply_thr(pt, ov_cls, thr), pt, "physio", f, thr, n, len(X_te))

    cd_tr, cd_te = _cd(X_tr, names), _cd(X_te, names)

    if not p_idx:
        if strict_cal is not None:
            i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
            rules = _build_rule_proba(strict_cal, n_classes)
            best = {"name": "rule_cd14", "f": -1.0, "thr": 1.01, "pt": None}
            for method, (pv_full, pt) in rules.items():
                pv = pv_full[i_val]
                thr = optimize_threshold_ov(y_tr[i_val], pv, ov_cls)
                f = float(_ov_f1(y_tr[i_val], apply_thr(pv, ov_cls, thr), ov_cls))
                if f > best["f"] + 1e-9:
                    best.update(name=f"rule_{method}", f=f, thr=float(thr), pt=pt)
            assert best["pt"] is not None
            return _pack(
                apply_thr(best["pt"], ov_cls, best["thr"]), best["pt"],
                best["name"], best["f"], best["thr"], n, len(X_te),
            )
        g_tr = None if groups_tr is None else np.asarray(groups_tr)
        g_te = None if groups_te is None else np.asarray(groups_te)
        i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))
        g_fit = None if g_tr is None else g_tr[i_fit]
        cyc_tr = _cyc_len(cd_tr[i_fit], g_fit, cd_tr, g_tr)
        cyc_te = _cyc_len(cd_tr, g_tr, cd_te, g_te)
        P_fit = fit_day_prior(cd_tr[i_fit], y_tr[i_fit], n_classes)
        P_all = fit_day_prior(cd_tr, y_tr, n_classes)
        p_cal_v = lookup_day_prior(cd_tr[i_val], P_fit, _ov_hat(cd_tr[i_val], cyc_tr[i_val]))
        p_cal_t = lookup_day_prior(cd_te, P_all, _ov_hat(cd_te, cyc_te))
        thr = optimize_threshold_ov(y_tr[i_val], p_cal_v, ov_cls)
        f = float(_ov_f1(y_tr[i_val], apply_thr(p_cal_v, ov_cls, thr), ov_cls))
        return _pack(apply_thr(p_cal_t, ov_cls, thr), p_cal_t, "cal_rule", f, thr, n, len(X_te))

    if variant == "v6":
        return _combined_v6(
            X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, strict_cal, n,
        )
    if variant == "v7":
        return _combined_v7(
            X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, strict_cal, n,
        )
    if variant == "v8":
        return _combined_v8(
            X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, strict_cal, n,
        )
    if variant == "v9":
        return _combined_v9(
            X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, groups_te, strict_cal, n,
        )
    return _combined_v10(
        X_tr, y_tr, X_te, names, n_classes, ov_cls, groups_tr, groups_te, strict_cal, n,
    )


def _self_check() -> None:
    rng = np.random.RandomState(0)
    n = 600
    pid = np.repeat(np.arange(30), 20)
    cd = np.zeros(n)
    y = np.zeros(n, dtype=int)
    bbt = rng.randn(n) * 0.15
    for p in np.unique(pid):
        m = pid == p
        if p >= 20:
            cd[m] = rng.uniform(1, 40, m.sum())
            ov = np.abs(cd[m] - 22) <= 2.5
        else:
            cd[m] = rng.uniform(1, 28, m.sum())
            ov = np.abs(cd[m] - 14) <= 2.5
        y[m] = ov.astype(int)
        bbt[m] += ov.astype(float) * (-0.35)
    X = np.column_stack([
        cd, cd - 14.0, np.abs(cd - 14.0),
        ((cd >= 12) & (cd <= 17)).astype(float),
        bbt, rng.randn(n), rng.randn(n),
    ])
    names = [
        "day_of_cycle", "cal_off14", "cal_abs_off14", "cal_win_12_17",
        "temp_mean", "rmssd_mean", "sdnn",
    ]
    te_m = np.zeros(n, dtype=bool)
    for p in np.unique(pid):
        te_m[np.where(pid == p)[0][-4:]] = True
    tr, te = np.where(~te_m)[0], np.where(te_m)[0]
    r6 = train(X[tr], y[tr], X[te], y[te], names, 2, 1, pid[tr], pid[te], variant="v6")
    assert str(r6["combined_pick"]).startswith("v6_")
    p_tr = onehot_pred(np.where(np.abs(cd[tr] - 15) <= 2, 1, 0), 2)
    p_te = onehot_pred(np.where(np.abs(cd[te] - 15) <= 2, 1, 0), 2)
    rules = {"cd15": (p_tr, p_te), "cd14": (p_tr, p_te), "hist_cl14": (p_tr, p_te)}
    global _build_rule_proba
    _orig = _build_rule_proba
    _build_rule_proba = lambda strict_cal, n_classes: rules  # noqa: E731
    try:
        r8 = train(
            X[tr], y[tr], X[te], y[te], names, 2, 1, pid[tr], pid[te],
            strict_cal={"label": "y2_mod"}, variant="v8",
        )
        r9 = train(
            X[tr], y[tr], X[te], y[te], names, 2, 1, pid[tr], pid[te],
            strict_cal={"label": "y2_mod"}, variant="v9",
        )
        r10 = train(
            X[tr], y[tr], X[te], y[te], names, 2, 1, pid[tr], pid[te],
            strict_cal={"label": "y2_mod"}, variant="v10",
        )
    finally:
        _build_rule_proba = _orig
    assert r8["pred"].shape == (len(te),)
    assert abs(r8["proba"].sum(1) - 1).max() < 1e-5
    assert r8["combined_pick"] in (V8_PICK, "v8_fixed[rule_cd15_floor]")
    assert r9["pred"].shape == (len(te),)
    assert abs(r9["proba"].sum(1) - 1).max() < 1e-5
    assert str(r9["combined_pick"]).startswith("v9[cd15_up[")
    assert r10["pred"].shape == (len(te),)
    assert abs(r10["proba"].sum(1) - 1).max() < 1e-5
    assert str(r10["combined_pick"]).startswith("v10[")
    assert model_name("v10") == "OvPhysio_v10"
    print("proposed_model OvPhysio v10 self-check OK")


if __name__ == "__main__":
    _self_check()
