# -*- coding: utf-8 -*-
"""Helpers shared by the five-axis grid (grid.py) and the OvPhysio baseline: participant split,
imputation, scikit-learn settings, the torch training loop and periovulatory scoring."""
from __future__ import annotations
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
from model import sota_model_dl as dl  # noqa: E402
from model import sota_model_ml as ml  # noqa: E402
from protocol_pdca import split_fit_val_idx  # noqa: E402


def _is_g(pid: str) -> bool:
    return str(pid).strip().upper().startswith("G")


def split_gs_sg(
    groups: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    is_g = np.array([_is_g(p) for p in groups])
    if mode == "gs":
        tr, te = is_g, ~is_g
    elif mode == "sg":
        tr, te = ~is_g, is_g
    else:
        raise ValueError(mode)
    if tr.sum() == 0 or te.sum() == 0:
        raise RuntimeError(f"split {mode}: empty side (G={is_g.sum()}, S={(~is_g).sum()})")
    return tr, te
DEPLOY_MIN_OV_PRECISION = 0.25
DEPLOY_THR_GRID = np.linspace(0.05, 0.90, 86)


def ovulation_class_id(label: str, n_classes: int) -> int:
    stem = str(label).strip().lower()
    if stem.startswith("y2") or n_classes == 2:
        return 0
    if stem.startswith("y3") or n_classes == 3:
        return 1
    return min(2, n_classes - 1)  # y4 / y5


def _ov_binary_scores(
    y_true: np.ndarray, y_pred: np.ndarray, ov_cls: int
) -> tuple[float, float, float]:
    yt = (y_true == ov_cls).astype(int)
    yp = (y_pred == ov_cls).astype(int)
    return (
        float(precision_score(yt, yp, zero_division=0)),
        float(recall_score(yt, yp, zero_division=0)),
        float(f1_score(yt, yp, zero_division=0)),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
    ov_cls: int = 2,
) -> dict[str, float]:
    """Macro + ovulation (one-vs-rest) deployment metrics."""
    average = "binary" if n_classes == 2 else "macro"
    ov_p, ov_r, ov_f = _ov_binary_scores(y_true, y_pred, ov_cls)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "ov_precision": ov_p,
        "ov_recall": ov_r,
        "ov_f1": ov_f,
        "ov_auroc": float("nan"),
        "ov_auprc": float("nan"),
        "ov_support": int((y_true == ov_cls).sum()),
        "ov_cls": int(ov_cls),
    }
    if y_proba is None or len(np.unique(y_true)) < 2:
        return out
    try:
        yt_ov = (y_true == ov_cls).astype(int)
        if y_proba.ndim == 2 and y_proba.shape[1] > ov_cls:
            pos_ov = y_proba[:, ov_cls]
        else:
            pos_ov = y_proba.reshape(-1)
        if yt_ov.sum() > 0 and yt_ov.sum() < len(yt_ov):
            out["ov_auroc"] = float(roc_auc_score(yt_ov, pos_ov))
            out["ov_auprc"] = float(average_precision_score(yt_ov, pos_ov))

        if n_classes == 2:
            # deployment: fertile (ov) one-vs-rest
            out["auroc"] = float(roc_auc_score(yt_ov, pos_ov))
            out["auprc"] = float(average_precision_score(yt_ov, pos_ov))
        else:
            classes = np.arange(n_classes)
            y_bin = label_binarize(y_true, classes=classes)
            if y_proba.shape[1] != n_classes:
                return out
            out["auroc"] = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
            aps = []
            for k in range(n_classes):
                if y_bin[:, k].sum() == 0:
                    continue
                aps.append(average_precision_score(y_bin[:, k], y_proba[:, k]))
            out["auprc"] = float(np.mean(aps)) if aps else float("nan")
    except Exception:
        pass
    return out


def _deploy_score(ov_p: float, ov_r: float, ov_f: float) -> float:
    """
    Deployment objective: recall first if the precision floor is met, else ov_f1.
    (minimize missed fertile windows — high-recall philosophy)
    """
    if ov_p >= DEPLOY_MIN_OV_PRECISION:
        return 1.0 + ov_r + 0.25 * ov_f  # candidates passing the floor always win
    return ov_f


def optimize_ovulation_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    ov_cls: int,
) -> float:
    """
    Search the P(ov) threshold on validation.
    p >= thr → ovulation, otherwise argmax over the remaining classes.
    """
    if proba is None or proba.ndim != 2 or proba.shape[1] <= ov_cls:
        return 0.5
    p_ov = proba[:, ov_cls]
    other = proba.copy()
    other[:, ov_cls] = -1.0
    other_pred = other.argmax(1)

    # baseline: plain argmax
    base_pred = proba.argmax(1)
    bp, br, bf = _ov_binary_scores(y_true, base_pred, ov_cls)
    best_thr = 0.0  # 0 is not quite the same as always-argmax; thr is searched below
    best_score = _deploy_score(bp, br, bf)
    best_thr = float("nan")  # nan = use argmax only

    for thr in DEPLOY_THR_GRID:
        pred = np.where(p_ov >= thr, ov_cls, other_pred)
        op, orec, of1 = _ov_binary_scores(y_true, pred, ov_cls)
        s = _deploy_score(op, orec, of1)
        if s > best_score + 1e-9:
            best_score, best_thr = s, float(thr)

    if np.isnan(best_thr):
        # argmax is better → a very high thr means almost no override
        # use thr=1.01 (never fires) instead of a separate flag
        return 1.01
    return best_thr


def apply_ovulation_threshold(
    proba: np.ndarray, ov_cls: int, thr: float
) -> np.ndarray:
    if proba is None:
        return np.array([])
    if thr >= 1.0 - 1e-12:
        return proba.argmax(1)
    p_ov = proba[:, ov_cls]
    other = proba.copy()
    other[:, ov_cls] = -1.0
    other_pred = other.argmax(1)
    return np.where(p_ov >= thr, ov_cls, other_pred)


def _ov_f1(y_true: np.ndarray, y_pred: np.ndarray, ov_cls: int) -> float:
    return _ov_binary_scores(y_true, y_pred, ov_cls)[2]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _split_fit_mon_idx(
    y_tr: np.ndarray,
    groups_tr: Optional[np.ndarray] = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    min_n: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-safe fit/val split (StratifiedGroupKFold). See protocol_pdca."""
    return split_fit_val_idx(
        y_tr, groups_tr,
        test_size=test_size, random_state=random_state, min_n=min_n,
    )


def _impute_fit_transform(
    X_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_tr), imp.transform(X_te)


def _sklearn_kwargs(model_name: str, n_classes: int, f1_boost: bool) -> dict[str, Any]:
    if not f1_boost:
        return {}
    if model_name == "LightGBM":
        return {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "min_child_samples": 20,
        }
    if model_name == "LR":
        return {"C": 2.0, "max_iter": 2000}
    if model_name == "DT":
        return {"max_depth": 12}
    return {}


def train_torch(
    model_name: str,
    group: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    ov_cls: int = 2,
    patience: int = 15,
    min_epochs: int = 10,
    f1_boost: bool = True,
    groups_tr: Optional[np.ndarray] = None,
    early_stop: str = "ov",
    random_state: int = 42,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if X_tr.ndim == 2:
        X_tr = X_tr[:, None, :]
        X_te = X_te[:, None, :]
    input_dim = int(X_tr.shape[-1])
    seq_len = int(X_tr.shape[1])

    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    if group == "ml":
        kw: dict[str, Any] = {}
        if model_name == "ME-LSTM":
            kw["num_patients"] = 0
        model = ml.build_model(model_name, input_dim=input_dim, num_classes=n_classes, **kw)
    else:
        kw = {"seq_len": seq_len}
        if model_name == "PatchTST":
            kw["patch_len"] = min(4, seq_len)
            kw["stride"] = max(1, min(2, seq_len))
        model = dl.build_model(model_name, input_dim=input_dim, num_classes=n_classes, **kw)

    if not isinstance(model, nn.Module):
        raise TypeError(f"{model_name} is not an nn.Module")

    device_t = torch.device(device)
    model = model.to(device_t)

    # deployment: boost ovulation class weight (inverse-freq × ov multiplier)
    cnt = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1.0)
    if f1_boost and 0 <= ov_cls < n_classes:
        w[ov_cls] *= 3.0
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32, device=device_t)

    i_fit, i_mon = _split_fit_mon_idx(y_tr, groups_tr)
    x_fit, x_mon = X_tr[i_fit], X_tr[i_mon]
    y_fit, y_mon = y_tr[i_fit], y_tr[i_mon]

    ds = TensorDataset(
        torch.tensor(x_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.long),
    )
    # BatchNorm (TABNET/CLOCS) needs B>1 — drop incomplete last batch
    bs = max(1, min(batch_size, len(ds)))
    dl_tr = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(ds) > bs))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable if trainable else model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=wt)

    x_mon_t = torch.tensor(x_mon, dtype=torch.float32)
    x_te_t = torch.tensor(X_te, dtype=torch.float32)

    best_ov = -1.0
    best_state = None
    bad = 0
    t0 = time.perf_counter()
    ep_run = 0
    for ep in range(epochs):
        ep_run = ep + 1
        model.train()
        for xb, yb in dl_tr:
            if xb.size(0) < 2:
                continue  # BatchNorm: single-sample batch
            xb, yb = xb.to(device_t), yb.to(device_t)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if hasattr(model, "aux_forecast_loss"):
                loss = loss + model.aux_forecast_loss(xb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits_m = model(x_mon_t.to(device_t))
            pred_m = logits_m.argmax(1).cpu().numpy()
            if early_stop == "macro":
                mon_score = _macro_f1(y_mon, pred_m)
            else:
                mon_score = _ov_f1(y_mon, pred_m, ov_cls)
        if mon_score > best_ov + 1e-4:
            best_ov = mon_score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith(("pipeline.", "backbone."))
            }
            bad = 0
        else:
            bad += 1
            if ep_run >= min_epochs and bad >= patience:
                break
    train_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    model.eval()
    with torch.no_grad():
        logits = model(x_te_t.to(device_t))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        logits_m = model(x_mon_t.to(device_t))
        proba_m = torch.softmax(logits_m, dim=1).cpu().numpy()

    ov_thr = 1.01
    if f1_boost and (y_mon == ov_cls).sum() >= 2:
        ov_thr = optimize_ovulation_threshold(y_mon, proba_m, ov_cls)
        pred = apply_ovulation_threshold(proba, ov_cls, ov_thr)
    else:
        pred = proba.argmax(1)

    metrics = compute_metrics(y_te, pred, proba, n_classes, ov_cls=ov_cls)
    metrics["train_sec"] = float(train_sec)
    metrics["epochs_run"] = int(ep_run)
    metrics["n_train"] = int(len(y_tr))
    metrics["n_test"] = int(len(y_te))
    metrics["ov_thr"] = float(ov_thr)
    return {
        "metrics": metrics,
        "pred": pred,
        "proba": proba,
        "proba_mon": proba_m,
        "y_mon": y_mon,
    }


def resolve_device(cuda_device: int) -> str:
    import torch

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        d = cuda_device if 0 <= cuda_device < n else 0
        return f"cuda:{d}"
    return "cpu"
