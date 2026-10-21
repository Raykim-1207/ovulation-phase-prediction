# -*- coding: utf-8 -*-
"""
grid — five-axis experimental grid, sklearn models, incremental logging

Same 3-arm protocol (cal / physio / combined, primary = ov_f1). Defaults to
LightGBM, SVM, LR, DT only (no DL, no OvPhysio).

  python3 .../grid.py --cuda-device 3
  # defaults: models-ml=top6  models-dl=none  out=results/grid_std
  # writes summary.csv + train.log after every run (--resume skips done keys)
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import csv
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
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

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE                 # src/train
CODE_DIR = TRAIN_DIR.parent      # src
ROOT = CODE_DIR.parent           # repository root
sys.path.insert(0, str(CODE_DIR))

from model import proposed_model as proposed  # noqa: E402
from model import sota_model_dl as dl  # noqa: E402
from model import sota_model_ml as ml  # noqa: E402
from protocol_pdca import (  # noqa: E402
    DL_N_SEEDS,
    LGBM_KW,
    PRIMARY,
    causal_patient_median_norm,
    cluster_bootstrap_f1,
    split_fit_val_idx,
)
from strict_cal_rule import (  # noqa: E402
    STRICT_CAL_METHODS,
    build_hist_cl_map,
    calendar_meta_from_df,
    predict_strict_cal,
)

import grid_utils as gu  # noqa: E402

DATA_ROOT = ROOT / "ovul_data" / "finalfinal"
OUT_ROOT = ROOT / "results" / "grid_std"

# ponytail: sklearn-only preset; override via --models-ml/dl
TOP6_ML = ["LightGBM", "SVM", "LR", "DT"]
TOP6_DL: list[str] = []
RESULT_KEY_COLS = (
    "family", "method", "label", "arm", "split", "group", "model",
)
# fixed schema so append doesn't change column count (cal vs combined).
SUMMARY_COLS = (
    "arm", "family", "method", "label", "split", "group", "model",
    "n_feats", "feats", "patnorm",
    "acc", "precision", "recall", "f1", "auroc", "auprc",
    "ov_precision", "ov_recall", "ov_f1", "ov_auroc", "ov_auprc",
    "ov_support", "ov_cls",
    "n_regular", "f1_regular", "ov_f1_regular",
    "n_mild_irregular", "f1_mild_irregular", "ov_f1_mild_irregular",
    "n_high_irregular", "f1_high_irregular", "ov_f1_high_irregular",
    "f1_boot_mean", "f1_boot_p025", "f1_boot_p975",
    "ov_thr", "combined_pick", "combined_val_f1",
)
_CORE_N = SUMMARY_COLS.index("ov_thr")  # 35: through f1_boot_p975
_SUMMARY_STR_COLS = frozenset({
    "arm", "family", "method", "label", "split", "group", "model",
    "feats", "combined_pick",
})


def _looks_like_pick(s: str) -> bool:
    t = (s or "").strip()
    return t.startswith(("rule_", "blend_", "multipath", "gate_")) or t in (
        "cal", "physio", "union", "lgbm_union",
    )


def _row_from_ragged(header: list[str], fields: list[str]) -> dict[str, Any]:
    """Map a CSV row that may have 36/37/38 fields onto SUMMARY_COLS."""
    if header == list(SUMMARY_COLS) and len(fields) == len(SUMMARY_COLS):
        return dict(zip(SUMMARY_COLS, fields))
    core_hdr = list(SUMMARY_COLS[:_CORE_N])
    n_core = min(_CORE_N, len(fields), len(header) if header[:_CORE_N] == core_hdr else _CORE_N)
    if header[:_CORE_N] == core_hdr:
        out = {SUMMARY_COLS[i]: fields[i] if i < len(fields) else "" for i in range(_CORE_N)}
        rest = fields[_CORE_N:]
    else:
        # unknown header: zip what we can
        out = {c: "" for c in SUMMARY_COLS}
        for k, v in zip(header, fields):
            if k in out:
                out[k] = v
        return out
    ov_thr = pick = val = ""
    if len(rest) == 1:
        ov_thr = rest[0]
    elif len(rest) == 2:
        a, b = rest
        if _looks_like_pick(a):
            pick, ov_thr = a, b
        else:
            ov_thr, pick = a, b
    elif len(rest) >= 3:
        a, b, c = rest[0], rest[1], rest[2]
        if _looks_like_pick(a):
            pick, val, ov_thr = a, b, c
        elif _looks_like_pick(b):
            ov_thr, pick, val = a, b, c
        else:
            ov_thr, pick, val = a, b, c
    out["ov_thr"] = ov_thr
    out["combined_pick"] = pick
    out["combined_val_f1"] = val
    return out


def _coerce_summary_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.reindex(columns=list(SUMMARY_COLS)).copy()
    for c in SUMMARY_COLS:
        if c in _SUMMARY_STR_COLS:
            out[c] = out[c].fillna("").astype(str)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(SUMMARY_COLS))
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return pd.DataFrame(columns=list(SUMMARY_COLS))
        rows = [_row_from_ragged(header, r) for r in reader if r]
    return _coerce_summary_dtypes(pd.DataFrame(rows, columns=list(SUMMARY_COLS)))


def rewrite_summary_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(".csv.tmp")
    _coerce_summary_dtypes(df).to_csv(tmp, index=False)
    tmp.replace(path)

LABEL_FILES = ("y2_std", "y3_std", "y4_std", "y5_std")   # the paper uses only the std definition
DEFAULT_LABELS = ("y5_std", "y4_std", "y3_std", "y2_std")
DEFAULT_FAMILIES = ("all",)  # every FAMILY_META folder under finalfinal
DEFAULT_METHODS = ("all",)  # night linear|locf + 24 {day}_{slot}
NIGHT_KEYS = ("participant_id", "file_name_x")
SEQ_LEN = 24
SKLEARN_ML = frozenset(ml.SKLEARN_BUILDERS.keys())


def resolve_ml_models(spec: str) -> list[str]:
    s = (spec or "").strip()
    if s.lower() in ("top6", "top10"):
        return list(TOP6_ML)
    if s.lower() in ("proposed", "only-proposed"):
        return [proposed.NAME]
    if not s or s.lower() == "all":
        out = list(ml.list_models())
        if proposed.NAME not in out:
            out.append(proposed.NAME)
        return out
    return [x.strip() for x in s.split(",") if x.strip()]


def resolve_dl_models(spec: str) -> list[str]:
    s = (spec or "").strip()
    if s.lower() in ("top6", "top10"):
        return list(TOP6_DL)
    if not s or s.lower() == "all":
        return list(dl.list_models())
    if s.lower() in ("none", "-", "off"):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


class RunLogger:
    """Append each result row to summary.csv and mirror prints to train.log."""

    def __init__(self, out_root: Path, resume: bool) -> None:
        self.out_root = out_root
        self.csv_path = out_root / "summary.csv"
        self.log_path = out_root / "train.log"
        self.done: set[tuple[Any, ...]] = set()
        prev = read_summary_csv(self.csv_path)
        if not prev.empty:
            # rewrite once so later pandas/append see a stable header
            rewrite_summary_csv(self.csv_path, prev)
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if resume and not prev.empty:
            for _, r in prev.iterrows():
                self.done.add(tuple(r[c] for c in RESULT_KEY_COLS))

    def result_key(
        self,
        family: str,
        method: str,
        label: str,
        arm: str,
        split: str,
        group: str,
        model: str,
    ) -> tuple[Any, ...]:
        return (family, method, label, arm, split, group, model)

    def is_done(self, key: tuple[Any, ...]) -> bool:
        return key in self.done

    def say(self, msg: str) -> None:
        print(msg, flush=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    def append(self, row: dict[str, Any]) -> None:
        key = tuple(row[c] for c in RESULT_KEY_COLS)
        if key in self.done:
            return
        self.done.add(key)
        stable = {c: row.get(c, "") for c in SUMMARY_COLS}
        pd.DataFrame([stable], columns=list(SUMMARY_COLS)).to_csv(
            self.csv_path,
            mode="a",
            header=not self._csv_header_written,
            index=False,
        )
        self._csv_header_written = True

    def load_summary(self) -> pd.DataFrame:
        return read_summary_csv(self.csv_path)

# ---------------------------------------------------------------------------
# family → kind + default feature whitelist ( ∩ present cols )
# ---------------------------------------------------------------------------
HRV_24 = [
    "avgHR", "maxNN", "avgNN", "sdnn", "nn50", "pnn50", "rmssd",
    "tp", "vlf", "lf", "hf", "lfhf",
]
# mask/dt = observed flags (GRU-D channels). Not mixed with interpolated values as tabular features.
SEQ24_COLS = HRV_24 + ["skin_temp"]

AGG_BASE = [
    "avgHR_mean", "avgHR_min", "avgHR_max",
    "maxNN_mean", "maxNN_min", "maxNN_max",
    "avgNN_mean", "avgNN_min", "avgNN_max",
    "sdnn_mean", "sdnn_min", "sdnn_max",
    "nn50_mean", "nn50_min", "nn50_max",
    "pnn50_mean", "pnn50_min", "pnn50_max",
    "rmssd_mean", "rmssd_min", "rmssd_max",
    "tp_mean", "tp_min", "tp_max",
    "vlf_mean", "vlf_min", "vlf_max",
    "lf_mean", "lf_min", "lf_max",
    "hf_mean", "hf_min", "hf_max",
    "lfhf_mean", "lfhf_min", "lfhf_max",
    "temp_mean", "temp_min", "temp_max", "temp_range",
]

AVA_COLS = list(AGG_BASE)

HUAWEI_COLS = [
    "bbt", "hr", "sdnn", "lfhf", "ln_sdnn", "ln_lfhf", "hr_min", "rmssd",
    "bbt_s3", "hr_s3", "dbbt_1", "dhr_1", "dbbt_3", "dhr_3",
    "bbt_slope5", "hr_slope5", "days_since_bbt_nadir", "rising_streak",
]

NPJ_COLS = [
    "temp_mean", "temp_std", "temp_min", "temp_max",
    "temp_b0_mean", "temp_b0_std", "temp_b1_mean", "temp_b1_std",
    "temp_b2_mean", "temp_b2_std", "temp_b3_mean", "temp_b3_std",
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "hr_b0_mean", "hr_b0_std", "hr_b1_mean", "hr_b1_std",
    "hr_b2_mean", "hr_b2_std", "hr_b3_mean", "hr_b3_std",
    "avgNN_mean", "avgNN_std", "avgNN_min", "avgNN_max",
    "avgNN_b0_mean", "avgNN_b0_std", "avgNN_b1_mean", "avgNN_b1_std",
    "avgNN_b2_mean", "avgNN_b2_std", "avgNN_b3_mean", "avgNN_b3_std",
    "hr_roll_late_minus_early", "temp_roll_late_minus_early",
    "avgNN_roll_late_minus_early",
    "d_temp_mean_1", "d_hr_mean_1", "d_temp_mean_2", "d_hr_mean_2",
]

OURA_COLS = [
    "temp_mean", "temp_min", "temp_max", "hr_mean", "hr_min",
    "rmssd_mean", "sdnn_mean",
    "temp_dev", "hr_dev", "rmssd_dev", "temp_roll3", "hr_roll3",
    "temp_slope3", "temp_rise_from_nadir", "cycle_day",
]

# folder name → (kind, style_key for WL)
FAMILY_META: dict[str, tuple[str, str]] = {
    "interp_24": ("seq24", "seq24"),
    "interp_24agg": ("night", "24agg"),
    "interp_ava": ("night", "ava"),
    "interp_ava24": ("night", "ava"),
    "interp_ava24mid": ("night", "ava"),
    "interp_ava30": ("night", "ava"),
    "interp_huawei": ("night", "huawei"),
    "interp_huawei24": ("night", "huawei"),
    "interp_huawei24mid": ("night", "huawei"),
    "interp_npj": ("night", "npj"),
    "interp_npj24": ("night", "npj"),
    "interp_npj24mid": ("night", "npj"),
    "interp_oura": ("night", "oura"),
    "interp_oura24": ("night", "oura"),
    "interp_oura24mid": ("night", "oura"),
}

STYLE_WL: dict[str, list[str]] = {
    "seq24": SEQ24_COLS,
    "24agg": list(AGG_BASE),
    "ava": AVA_COLS,
    "huawei": HUAWEI_COLS,
    "npj": NPJ_COLS,
    "oura": OURA_COLS,
}

# short alias → folder names
ALIAS: dict[str, tuple[str, ...]] = {
    "24": ("interp_24",),
    "24agg": ("interp_24agg",),
    "ava": ("interp_ava",),
    "ava24": ("interp_ava24",),
    "ava24mid": ("interp_ava24mid",),
    "ava30": ("interp_ava30",),
    "huawei": ("interp_huawei",),
    "huawei24": ("interp_huawei24",),
    "huawei24mid": ("interp_huawei24mid",),
    "npj": ("interp_npj",),
    "npj24": ("interp_npj24",),
    "npj24mid": ("interp_npj24mid",),
    "oura": ("interp_oura",),
    "oura24": ("interp_oura24",),
    "oura24mid": ("interp_oura24mid",),
    "night": ("interp_ava", "interp_huawei", "interp_npj", "interp_oura"),
    "night24": (
        "interp_ava24", "interp_huawei24", "interp_npj24", "interp_oura24",
    ),
}


def expand_families(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return sorted(FAMILY_META.keys())
    out: list[str] = []
    for tok in arg.split(","):
        t = tok.strip()
        if not t:
            continue
        if t in FAMILY_META:
            out.append(t)
        elif t in ALIAS:
            out.extend(ALIAS[t])
        elif f"interp_{t}" in FAMILY_META:
            out.append(f"interp_{t}")
        else:
            raise SystemExit(f"unknown family alias: {t!r}")
    # unique keep order
    seen: set[str] = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def discover(
    data_root: Path,
    families: list[str],
    methods: Optional[list[str]],
    labels: list[str],
) -> list[dict[str, Any]]:
    """One entry per (family, method, label) that exists on disk."""
    rows: list[dict[str, Any]] = []
    for fam in families:
        if fam not in FAMILY_META:
            continue
        fam_dir = data_root / fam
        if not fam_dir.is_dir():
            continue
        kind, style = FAMILY_META[fam]
        method_dirs = sorted(p for p in fam_dir.iterdir() if p.is_dir())
        for md in method_dirs:
            if methods is not None and md.name not in methods:
                continue
            for lab in labels:
                path = md / f"{lab}.xlsx"
                if path.is_file():
                    rows.append({
                        "family": fam,
                        "method": md.name,
                        "label": lab,
                        "path": path,
                        "kind": kind,
                        "style": style,
                        "key": f"{fam}/{md.name}/{lab}",
                    })
    return rows


def read_columns(path: Path) -> list[str]:
    df = pd.read_excel(path, nrows=0, engine="openpyxl")
    return [str(c).strip() for c in df.columns]


def propose_features(
    style: str,
    present: list[str],
    override: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    wl = override if override is not None else STYLE_WL[style]
    hit = [c for c in wl if c in present]
    miss = [c for c in wl if c not in present]
    return hit, miss


# cal = menses start → cycle day only; never feed raw date cols
LEAK_COLS = frozenset({
    "ovulation_date", "label_date", "menses_start", "next_menses_start",
})
# never model features (m_cycle uses next menses → prospective leak)
META_COLS = frozenset({"m_cycle", "m_cycle_group"})
# coverage / interpolation flags — never classifier X. Only coverage counts are used, as the OvPhysio gate.
QUALITY_COLS = frozenset({
    "n_hrv_win", "n_temp_pts", "n_slots", "used_min", "record_min",
    "hrv_mask", "temp_mask", "hrv_dt", "temp_dt",
    "temp_slot_imputed", "temp_day_imputed",
})
QUALITY_COV = ("n_hrv_win", "n_temp_pts", "n_slots", "used_min", "record_min")
SAFE_CAL_COLS = ("day_of_cycle", "cycle_day")  # cycle day = f(menses start, measure date)
COMPETE_ARMS = ("cal", "physio", "combined")
M_CYCLE_GROUPS = ("regular", "mild_irregular", "high_irregular")
THR_GRID = np.linspace(0.05, 0.95, 91)


def m_cycle_group_of(length: float) -> str:
    """Measured cycle length → regularity subgroup (JMIR-style bins)."""
    if length is None or (isinstance(length, float) and not np.isfinite(length)):
        return "unknown"
    n = int(length)
    if 27 <= n <= 32:
        return "regular"
    if 24 <= n <= 26 or 33 <= n <= 35:
        return "mild_irregular"
    if n <= 23 or n >= 36:
        return "high_irregular"
    return "unknown"


def read_m_cycle_group(df: pd.DataFrame) -> np.ndarray:
    """Prefer stored m_cycle_group; else derive from dates or m_cycle."""
    cols = {str(c).strip(): c for c in df.columns}
    if "m_cycle_group" in cols:
        return df[cols["m_cycle_group"]].astype(str).to_numpy()
    if "m_cycle" in cols:
        return np.array(
            [m_cycle_group_of(x) for x in pd.to_numeric(df[cols["m_cycle"]], errors="coerce")],
            dtype=object,
        )
    if "menses_start" in cols and "next_menses_start" in cols:
        msd = pd.to_datetime(df[cols["menses_start"]], errors="coerce")
        nmsd = pd.to_datetime(df[cols["next_menses_start"]], errors="coerce")
        length = (nmsd - msd).dt.days
        return np.array([m_cycle_group_of(x) for x in length], dtype=object)
    return np.array(["unknown"] * len(df), dtype=object)


def propose_cal_physio(
    style: str,
    present: list[str],
    override: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    hit, miss = propose_features(style, present, override)
    for c in SAFE_CAL_COLS:
        if c in present and c not in hit and c not in LEAK_COLS:
            hit.append(c)
    hit = [
        c for c in hit
        if c not in LEAK_COLS and c not in META_COLS and c not in QUALITY_COLS
    ]
    return hit, miss


def propose_cal_only(present: list[str]) -> tuple[list[str], list[str]]:
    hit = [c for c in SAFE_CAL_COLS if c in present]
    miss = [c for c in SAFE_CAL_COLS if c not in present]
    return hit, miss


def propose_physio_only(
    style: str,
    present: list[str],
    override: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    hit, miss = propose_features(style, present, override)
    hit = [
        c for c in hit
        if c not in LEAK_COLS and c not in SAFE_CAL_COLS
        and c not in META_COLS and c not in QUALITY_COLS
    ]
    return hit, miss


def append_cal_eng(
    X: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """day_of_cycle/cycle_day → off14/window/sin28 (no date leaks)."""
    src = next((k for k in SAFE_CAL_COLS if k in names), None)
    if src is None:
        return X, names
    cd = X[:, names.index(src)].astype(np.float64)
    eng = {
        "cal_days_from_menses": cd - 1.0,
        "cal_off14": cd - 14.0,
        "cal_off15": cd - 15.0,
        "cal_abs_off14": np.abs(cd - 14.0),
        "cal_win_12_17": ((cd >= 12.0) & (cd <= 17.0)).astype(np.float64),
        "cal_sin28": np.sin(2.0 * np.pi * cd / 28.0),
        "cal_cos28": np.cos(2.0 * np.pi * cd / 28.0),
    }
    cols, new_names = [], list(names)
    for n, v in eng.items():
        if n in new_names:
            continue
        cols.append(v)
        new_names.append(n)
    if not cols:
        return X, names
    return np.column_stack([X, *cols]), new_names


def is_cal_name(n: str) -> bool:
    return n in SAFE_CAL_COLS or str(n).startswith("cal_")


def physio_indices(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if not is_cal_name(n)]


def cal_indices(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if is_cal_name(n)]


def arm_col_indices(names: list[str], arm: str) -> list[int]:
    if arm == "cal":
        return cal_indices(names)
    if arm == "physio":
        return physio_indices(names)
    if arm == "combined":
        return list(range(len(names)))
    raise ValueError(arm)


def patient_median_norm(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    g_tr: np.ndarray,
    g_te: np.ndarray,
    physio_idx: list[int],
    cycle_tr: Optional[np.ndarray] = None,
    cycle_day_tr: Optional[np.ndarray] = None,
    cycle_te: Optional[np.ndarray] = None,
    cycle_day_te: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal within-cycle median centering (deployable; no test-fit stats)."""
    if not physio_idx:
        return X_tr, X_te
    if cycle_tr is None or cycle_day_tr is None:
        cycle_tr = np.zeros(len(g_tr))
        cycle_day_tr = np.arange(len(g_tr), dtype=np.float64)
    if cycle_te is None or cycle_day_te is None:
        cycle_te = np.zeros(len(g_te))
        cycle_day_te = np.arange(len(g_te), dtype=np.float64)
    return causal_patient_median_norm(
        X_tr, X_te, g_tr, g_te, physio_idx,
        cycle_tr, cycle_day_tr, cycle_te, cycle_day_te,
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _ov_f1(y_true: np.ndarray, y_pred: np.ndarray, ov_cls: int) -> float:
    yt = (np.asarray(y_true) == ov_cls).astype(int)
    yp = (np.asarray(y_pred) == ov_cls).astype(int)
    return float(f1_score(yt, yp, zero_division=0))


def optimize_threshold_macro(
    y_true: np.ndarray, proba: np.ndarray, ov_cls: int
) -> float:
    if proba is None or proba.ndim != 2 or proba.shape[1] <= ov_cls:
        return 1.01
    p_ov = proba[:, ov_cls]
    other = proba.copy()
    other[:, ov_cls] = -1.0
    other_pred = other.argmax(1)
    best_thr, best_f = 1.01, _macro_f1(y_true, proba.argmax(1))
    for thr in THR_GRID:
        pred = np.where(p_ov >= thr, ov_cls, other_pred)
        f = _macro_f1(y_true, pred)
        if f > best_f + 1e-9:
            best_f, best_thr = f, float(thr)
    return best_thr


def optimize_threshold_ov(
    y_true: np.ndarray, proba: np.ndarray, ov_cls: int
) -> float:
    if proba is None or proba.ndim != 2 or proba.shape[1] <= ov_cls:
        return 1.01
    p_ov = proba[:, ov_cls]
    other = proba.copy()
    other[:, ov_cls] = -1.0
    other_pred = other.argmax(1)
    best_thr, best_f = 1.01, _ov_f1(y_true, proba.argmax(1), ov_cls)
    for thr in THR_GRID:
        pred = np.where(p_ov >= thr, ov_cls, other_pred)
        f = _ov_f1(y_true, pred, ov_cls)
        if f > best_f + 1e-9:
            best_f, best_thr = f, float(thr)
    return best_thr


def onehot_pred(
    pred: np.ndarray, n_classes: int, keep: Optional[np.ndarray] = None
) -> np.ndarray:
    n = len(pred)
    out = np.full((n, n_classes), 1.0 / max(n_classes, 1), dtype=np.float64)
    m = np.asarray(keep, dtype=bool) if keep is not None else (np.asarray(pred) >= 0)
    for i in np.flatnonzero(m):
        ci = int(pred[i])
        if 0 <= ci < n_classes:
            out[i] = 0.0
            out[i, ci] = 1.0
    return out


def build_strict_rule_probas(
    cal_meta: dict[str, np.ndarray],
    label: str,
    n_classes: int,
    hist_cl=None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """method -> (keep, onehot_proba) aligned to full dataset rows."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in STRICT_CAL_METHODS:
        keep, pred = predict_strict_cal(
            method, label,
            cal_meta["pid"], cal_meta["cyc"],
            cal_meta["msd"], cal_meta["nmsd"], cal_meta["day"],
            hist_cl=hist_cl,
        )
        out[method] = (keep, onehot_pred(pred, n_classes, keep))
    return out

def apply_thr(proba: np.ndarray, ov_cls: int, thr: float) -> np.ndarray:
    if thr >= 1.0 - 1e-12:
        return proba.argmax(1)
    p_ov = proba[:, ov_cls]
    other = proba.copy()
    other[:, ov_cls] = -1.0
    return np.where(p_ov >= thr, ov_cls, other.argmax(1))


def compute_metrics_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
    ov_cls: int,
) -> dict[str, float]:
    """Always macro for P/R/F1 (binary included)."""
    ov_p, ov_r, ov_f = gu._ov_binary_scores(y_true, y_pred, ov_cls)
    out: dict[str, float] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1": _macro_f1(y_true, y_pred),
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
        pos_ov = y_proba[:, ov_cls] if y_proba.ndim == 2 else y_proba.reshape(-1)
        if yt_ov.sum() > 0 and yt_ov.sum() < len(yt_ov):
            out["ov_auroc"] = float(roc_auc_score(yt_ov, pos_ov))
            out["ov_auprc"] = float(average_precision_score(yt_ov, pos_ov))
        if n_classes == 2:
            out["auroc"] = out["ov_auroc"]
            out["auprc"] = out["ov_auprc"]
        elif y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            out["auroc"] = float(
                roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            )
            aps = [
                average_precision_score(y_bin[:, k], y_proba[:, k])
                for k in range(n_classes)
                if y_bin[:, k].sum() > 0
            ]
            out["auprc"] = float(np.mean(aps)) if aps else float("nan")
    except Exception:
        pass
    return out


def metrics_with_m_cycle_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
    ov_cls: int,
    m_cycle_group: np.ndarray,
) -> dict[str, float]:
    """Overall macro metrics + per m_cycle_group test macro/ov F1."""
    out = compute_metrics_macro(y_true, y_pred, y_proba, n_classes, ov_cls)
    mcg = np.asarray(m_cycle_group).astype(str)
    for gname in M_CYCLE_GROUPS:
        m = mcg == gname
        out[f"n_{gname}"] = int(m.sum())
        if m.sum() < 2 or len(np.unique(y_true[m])) < 1:
            out[f"f1_{gname}"] = float("nan")
            out[f"ov_f1_{gname}"] = float("nan")
            continue
        yp = y_proba[m] if y_proba is not None else None
        sub = compute_metrics_macro(y_true[m], y_pred[m], yp, n_classes, ov_cls)
        out[f"f1_{gname}"] = sub["f1"]
        out[f"ov_f1_{gname}"] = sub["ov_f1"]
    return out


def train_sklearn_macro(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
    ov_cls: int,
    groups_tr: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Impute → fit → thr maximizing macro F1 on group-stratified holdout."""
    X_tr, X_te = gu._impute_fit_transform(X_tr, X_te)
    kw = gu._sklearn_kwargs(model_name, n_classes, f1_boost=True)
    if model_name == "LightGBM":
        kw = dict(LGBM_KW)
    est = ml.build_model(model_name, num_classes=n_classes, **kw)
    if model_name == "LightGBM" and hasattr(est, "set_params"):
        est.set_params(n_jobs=1)

    n = len(y_tr)
    use_cal = n >= 40 and (y_tr == ov_cls).sum() >= 3
    if use_cal:
        i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=40)
        X_fit, y_fit = X_tr[i_fit], y_tr[i_fit]
        X_val, y_val = X_tr[i_val], y_tr[i_val]
    else:
        X_fit, y_fit = X_tr, y_tr
        X_val, y_val = X_tr, y_tr

    t0 = time.perf_counter()
    if model_name in ("LR", "SVM"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
        pipe.fit(X_fit, y_fit)
        proba_val = pipe.predict_proba(X_val) if hasattr(pipe, "predict_proba") else None
        proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
        fallback_pred = pipe.predict(X_te)
    else:
        est.fit(X_fit, y_fit)
        proba_val = est.predict_proba(X_val) if hasattr(est, "predict_proba") else None
        proba = est.predict_proba(X_te) if hasattr(est, "predict_proba") else None
        fallback_pred = est.predict(X_te)
    train_sec = time.perf_counter() - t0

    ov_thr = 1.01
    if use_cal and proba_val is not None and proba is not None:
        ov_thr = optimize_threshold_macro(y_val, proba_val, ov_cls)
        pred = apply_thr(proba, ov_cls, ov_thr)
    else:
        pred = proba.argmax(1) if proba is not None else fallback_pred

    metrics = compute_metrics_macro(y_te, pred, proba, n_classes, ov_cls)
    metrics["train_sec"] = float(train_sec)
    metrics["n_train"] = int(len(y_tr))
    metrics["n_test"] = int(len(y_te))
    metrics["ov_thr"] = float(ov_thr)
    return {"metrics": metrics, "pred": pred, "proba": proba}


def train_combined_blend(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    feat_names: list[str],
    n_classes: int,
    ov_cls: int,
    groups_tr: Optional[np.ndarray] = None,
    rule_proba: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
) -> dict[str, Any]:
    """
    combined = val-picked among {strict rules, cal, physio, union, soft blends}.
    Pick metric = ov_f1; rule floor (+0.005) so combined ≥ best strict rule on val.
    """
    c_idx = cal_indices(feat_names)
    p_idx = physio_indices(feat_names)
    if not c_idx or not p_idx:
        return train_sklearn_macro(
            model_name, X_tr, y_tr, X_te, y_te, n_classes, ov_cls, groups_tr,
        )

    X_tr_i, X_te_i = gu._impute_fit_transform(X_tr, X_te)
    n = len(y_tr)
    i_fit, i_val = split_fit_val_idx(y_tr, groups_tr, min_n=max(1, n // 5))

    views = {
        "cal": (c_idx, False),
        "physio": (p_idx, True),
        "union": (list(range(X_tr_i.shape[1])), True),
    }
    # physio/union: patnorm already applied upstream on physio cols of combined matrix

    def _fit_proba(cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
        kw = gu._sklearn_kwargs(model_name, n_classes, f1_boost=True)
        if model_name == "LightGBM":
            kw = dict(LGBM_KW)
        est = ml.build_model(model_name, num_classes=n_classes, **kw)
        if model_name == "LightGBM" and hasattr(est, "set_params"):
            est.set_params(n_jobs=1)
        Xf, Xv, Xt = X_tr_i[i_fit][:, cols], X_tr_i[i_val][:, cols], X_te_i[:, cols]
        yf = y_tr[i_fit]
        if model_name in ("LR", "SVM"):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
            pipe.fit(Xf, yf)
            pv = pipe.predict_proba(Xv)
            pt = pipe.predict_proba(Xt)
        else:
            est.fit(Xf, yf)
            pv = est.predict_proba(Xv)
            pt = est.predict_proba(Xt)
        return pv, pt

    t0 = time.perf_counter()
    proba_val: dict[str, np.ndarray] = {}
    proba_te: dict[str, np.ndarray] = {}
    for key, (cols, _) in views.items():
        pv, pt = _fit_proba(cols)
        proba_val[key] = pv
        proba_te[key] = pt
    if rule_proba:
        for method, (p_tr, p_te) in rule_proba.items():
            proba_val[f"rule_{method}"] = p_tr[i_val]
            proba_te[f"rule_{method}"] = p_te

    y_val = y_tr[i_val]
    pick = pick_cal_heavy_blend(proba_val, proba_te, y_val, ov_cls)
    best_name, best_f, best_thr, best_pt = (
        pick["combined_pick"], pick["combined_val_f1"], pick["ov_thr"], pick["proba"]
    )
    pred = pick["pred"]
    train_sec = time.perf_counter() - t0
    metrics = compute_metrics_macro(y_te, pred, best_pt, n_classes, ov_cls)
    metrics["train_sec"] = float(train_sec)
    metrics["n_train"] = int(len(y_tr))
    metrics["n_test"] = int(len(y_te))
    metrics["ov_thr"] = float(best_thr)
    metrics["combined_pick"] = best_name
    metrics["combined_val_f1"] = float(best_f)
    return {"metrics": metrics, "pred": pred, "proba": best_pt}


def pick_cal_heavy_blend(
    proba_val: dict[str, np.ndarray],
    proba_te: dict[str, np.ndarray],
    y_val: np.ndarray,
    ov_cls: int,
) -> dict[str, Any]:
    """Val-pick by ov_f1; floor to best rule_*/cal unless challenger ≥ floor+0.005."""
    rule_keys = sorted(k for k in proba_val if k.startswith("rule_"))
    singles = [k for k in ("cal", "physio", "union", *rule_keys) if k in proba_val and k in proba_te]
    cand: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    cw0 = {"cal": 1.0, "physio": 0.0, "union": 0.5}
    for key in singles:
        cw = cw0.get(key, 1.0 if key.startswith("rule_") else 0.5)
        cand.append((key, proba_val[key], proba_te[key], cw))
    anchors = rule_keys + (["cal"] if "cal" in proba_val else [])
    partners = [k for k in ("physio", "union") if k in proba_val]
    for w in (0.85, 0.9, 0.95):
        for a in anchors:
            for b in partners:
                cand.append((
                    f"blend_{a}{w:.2f}_{b}",
                    w * proba_val[a] + (1.0 - w) * proba_val[b],
                    w * proba_te[a] + (1.0 - w) * proba_te[b],
                    w,
                ))
    if not cand:
        raise ValueError("pick_cal_heavy_blend: empty candidates")

    best_name, best_f, best_thr, best_cw = cand[0][0], -1.0, 1.01, -1.0
    best_pt = cand[0][2]
    floor_name, floor_f, floor_thr, floor_pt = None, -1.0, 1.01, best_pt
    for name, pv, pt, cw in cand:
        thr = optimize_threshold_ov(y_val, pv, ov_cls)
        f = _ov_f1(y_val, apply_thr(pv, ov_cls, thr), ov_cls)
        is_floor = name.startswith("rule_") or name == "cal"
        if is_floor and (
            f > floor_f + 1e-9
            or (abs(f - floor_f) <= 1e-9 and (floor_name is None or name < floor_name))
        ):
            floor_name, floor_f, floor_thr, floor_pt = name, f, thr, pt
        if f > best_f + 1e-9 or (abs(f - best_f) <= 1e-9 and cw > best_cw):
            best_f, best_name, best_thr, best_cw = f, name, thr, cw
            best_pt = pt
    if floor_name is not None and best_name != floor_name and best_f < floor_f + 0.005:
        best_name, best_f, best_thr, best_pt = floor_name, floor_f, floor_thr, floor_pt
    pred = apply_thr(best_pt, ov_cls, best_thr)
    return {
        "pred": pred,
        "proba": best_pt,
        "combined_pick": best_name,
        "combined_val_f1": float(best_f),
        "ov_thr": float(best_thr),
    }


def _quality_mat(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in QUALITY_COV if c in df.columns]
    if not names:
        return np.zeros((len(df), 0), dtype=np.float64), []
    Q = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return Q, names


def load_night(
    path: Path, label_col: str, feat_cols: list[str]
) -> tuple:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if label_col not in df.columns:
        raise KeyError(f"{path.name}: missing {label_col}")
    use = [
        c for c in feat_cols
        if c in df.columns and c not in LEAK_COLS
        and c not in META_COLS and c not in QUALITY_COLS
    ]
    if not use:
        raise KeyError(f"{path.name}: no feature cols")
    df = df.dropna(subset=[label_col]).copy()
    y_raw = pd.to_numeric(df[label_col], errors="coerce")
    df = df.loc[y_raw.notna()].copy()
    y = y_raw.loc[df.index].astype(int).to_numpy() - 1
    X = df[use].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    groups = df["participant_id"].astype(str).to_numpy()
    mcg = read_m_cycle_group(df)
    Q, qn = _quality_mat(df)
    cyc = (
        pd.to_numeric(df["cycle"], errors="coerce").fillna(0).to_numpy()
        if "cycle" in df.columns else np.zeros(len(df), dtype=np.float64)
    )
    cd = (
        pd.to_numeric(df["day_of_cycle"], errors="coerce").fillna(14).to_numpy()
        if "day_of_cycle" in df.columns else np.full(len(df), 14.0)
    )
    meta = calendar_meta_from_df(df)
    return X, y, groups, use, mcg, Q, qn, cyc, cd, meta


def load_seq24(
    path: Path, label_col: str, feat_cols: list[str]
) -> tuple:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    use = [
        c for c in feat_cols
        if c in df.columns and c not in LEAK_COLS
        and c not in META_COLS and c not in QUALITY_COLS
    ]
    if not use:
        raise KeyError(f"{path}: no feature cols")
    keys = [k for k in NIGHT_KEYS if k in df.columns]
    if "window" not in df.columns or len(keys) < 2:
        raise KeyError(f"{path}: need window + {NIGHT_KEYS}")
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    df["window"] = pd.to_numeric(df["window"], errors="coerce")

    slot_cols = [c for c in use if c not in SAFE_CAL_COLS]
    cal_cols = [c for c in use if c in SAFE_CAL_COLS]
    qn = [c for c in ("hrv_mask", "temp_mask") if c in df.columns]
    q_map = {"hrv_mask": "n_hrv_win", "temp_mask": "n_temp_pts"}
    seqs, cals, ys, groups, mcgs, q_rows, cycles, cds = [], [], [], [], [], [], [], []
    meta_rows: list[dict[str, Any]] = []
    for (pid, _), g in df.groupby(keys, sort=False):
        g = g.sort_values("window")
        mat = np.full((SEQ_LEN, len(slot_cols)), np.nan, dtype=np.float64)
        if slot_cols:
            for _, row in g.iterrows():
                w = int(row["window"])
                if 1 <= w <= SEQ_LEN:
                    mat[w - 1] = pd.to_numeric(
                        row[slot_cols], errors="coerce"
                    ).to_numpy(dtype=np.float64)
        seqs.append(mat)
        if cal_cols:
            cals.append(
                pd.to_numeric(g.iloc[0][cal_cols], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            )
        ys.append(int(g[label_col].iloc[0]) - 1)
        groups.append(str(pid))
        mcgs.append(str(read_m_cycle_group(g)[0]))
        cycles.append(
            float(pd.to_numeric(g.iloc[0].get("cycle", 0), errors="coerce") or 0)
        )
        cds.append(
            float(pd.to_numeric(g.iloc[0].get("day_of_cycle", 14), errors="coerce") or 14)
        )
        if qn:
            q_rows.append([
                float(pd.to_numeric(g[c], errors="coerce").fillna(0).sum()) for c in qn
            ])
        r0 = g.iloc[0]
        meta_rows.append({
            "pid": str(pid),
            "cyc": r0.get("cycle", np.nan),
            "msd": r0.get("menses_start", None),
            "nmsd": r0.get("next_menses_start", None),
            "day": r0.get("measure_date", None),
        })
    X_seq = np.stack(seqs, axis=0)
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    names = [f"{c}_w{w}" for w in range(1, SEQ_LEN + 1) for c in slot_cols]
    if cal_cols:
        X_cal = np.vstack(cals)
        X_flat = np.column_stack([X_flat, X_cal]) if X_flat.size else X_cal
        names = names + cal_cols
    Q = np.asarray(q_rows, dtype=np.float64) if q_rows else np.zeros((len(ys), 0))
    qnames = [q_map[c] for c in qn]
    meta_df = pd.DataFrame(meta_rows)
    from strict_cal_rule import to_date as _to_date
    from datetime import timedelta as _td
    days = []
    for i, row in meta_df.iterrows():
        d = _to_date(row["day"])
        if d is None:
            m = _to_date(row["msd"])
            d = None if m is None else m + _td(days=int(cds[i]) - 1)
        days.append(d)
    meta = {
        "pid": meta_df["pid"].astype(str).to_numpy(),
        "cyc": pd.to_numeric(meta_df["cyc"], errors="coerce").to_numpy(),
        "msd": meta_df["msd"].to_numpy(),
        "nmsd": meta_df["nmsd"].to_numpy(),
        "day": np.array(days, dtype=object),
    }
    return (
        X_seq,
        X_flat,
        np.asarray(ys, dtype=np.int64),
        np.asarray(groups),
        names,
        np.asarray(mcgs, dtype=object),
        Q,
        qnames,
        np.asarray(cycles, dtype=np.float64),
        np.asarray(cds, dtype=np.float64),
        meta,
    )


def print_feature_plan(src: dict[str, Any], feats: list[str], miss: list[str]) -> None:
    print(f"\n=== {src['key']}  kind={src['kind']} style={src['style']} ===")
    print(f"  path: {src['path']}")
    print(f"  proposed ({len(feats)}): {', '.join(feats)}")
    if miss:
        print(f"  missing vs WL ({len(miss)}): {', '.join(miss)}")
    else:
        print("  missing vs WL: (none)")


def confirm_features(src: dict[str, Any], feats: list[str], miss: list[str]) -> bool:
    print_feature_plan(src, feats, miss)
    try:
        ans = input("  Proceed with these features? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def load_features_file(path: Path) -> dict[str, list[str]]:
    """JSON: {\"interp_ava\": [\"temp_mean\", ...], \"ava\": [...]} style or family keys."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("--features-file must be a JSON object")
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            raise SystemExit(f"features-file[{k!r}] must be a list")
        out[str(k)] = [str(x) for x in v]
    return out


def override_for(src: dict[str, Any], fmap: dict[str, list[str]]) -> Optional[list[str]]:
    for key in (src["key"], f"{src['family']}/{src['method']}", src["family"], src["style"]):
        if key in fmap:
            return fmap[key]
    return None


def write_report_pivots(big: pd.DataFrame, out_root: Path, arms: list[str]) -> None:
    """
    EXPLORATORY ONLY — idxmax over grid; do not use for confirmatory claims.
    Ranking metric follows PRIMARY['metric'] (default ov_f1).
    See primary_confirmatory.csv (+ selection_bias_report.json for bootstrap).
    """
    metric = PRIMARY.get("metric", "ov_f1")
    if metric not in big.columns:
        metric = "f1" if "f1" in big.columns else None
    if big.empty or metric is None:
        return

    # exploratory: best model within each arm×family×label×split
    idx = big.groupby(["arm", "label", "split", "family"], dropna=False)[metric].idxmax()
    best_fam = big.loc[idx].copy()
    idx2 = best_fam.groupby(["arm", "label", "split"], dropna=False)[metric].idxmax()
    best = best_fam.loc[idx2].copy()

    pivot = best.pivot_table(
        index=["label", "split"], columns="arm", values=metric, aggfunc="max"
    )
    for a in arms:
        if a not in pivot.columns:
            pivot[a] = float("nan")
    pivot = pivot.reindex(columns=[a for a in arms if a in pivot.columns])
    if set(arms) >= {"cal", "physio", "combined"} and "combined" in pivot.columns:
        others = [a for a in ("cal", "physio") if a in pivot.columns]
        if others:
            pivot["combined_wins"] = pivot["combined"] >= pivot[others].max(axis=1)
    pivot_path = out_root / "exploratory_pivot_arm_label_split.csv"
    pivot.to_csv(pivot_path)
    print(f"→ {pivot_path}  (EXPLORATORY — metric={metric}, not confirmatory)")

    gcols = [
        c for c in (
            "arm", "label", "split", "family", "model", "f1", "ov_f1",
            "f1_regular", "f1_mild_irregular", "f1_high_irregular",
            "ov_f1_regular", "ov_f1_mild_irregular", "ov_f1_high_irregular",
            "n_regular", "n_mild_irregular", "n_high_irregular",
            "f1_boot_mean", "f1_boot_p025", "f1_boot_p975",
        )
        if c in best.columns
    ]
    gpath = out_root / "exploratory_pivot_m_cycle_groups.csv"
    best[gcols].sort_values(["label", "split", "arm"]).to_csv(gpath, index=False)
    print(f"→ {gpath}  (EXPLORATORY)")

    rows = []
    for (lab, split), sub in best.groupby(["label", "split"], dropna=False):
        d = {r["arm"]: float(r[metric]) for _, r in sub.iterrows()}
        c = d.get("combined", float("nan"))
        cal = d.get("cal", float("nan"))
        phy = d.get("physio", float("nan"))
        wins = (
            np.isfinite(c)
            and all(c >= v - 1e-12 for v in (cal, phy) if np.isfinite(v))
            and any(np.isfinite(v) for v in (cal, phy))
        )
        rows.append({
            "label": lab, "split": split, "metric": metric,
            "cal": cal, "physio": phy, "combined": c,
            "combined_wins": wins,
        })
    bylab = out_root / "exploratory_arm_compete_by_label.csv"
    pd.DataFrame(rows).to_csv(bylab, index=False)
    print(f"→ {bylab}  (EXPLORATORY)")
    n_win = sum(1 for r in rows if r["combined_wins"])
    print(f"  combined_wins on {n_win}/{len(rows)} label×split cells  (EXPLORATORY, {metric})")


def write_primary_confirmatory(df: pd.DataFrame, out_root: Path) -> None:
    """Pre-registered PRIMARY cell only (Cawley & Talbot — one confirmatory contrast)."""
    if df.empty:
        return
    spec = PRIMARY
    metric = spec.get("metric", "ov_f1")
    m = df[
        (df["family"] == spec["family"])
        & (df["method"] == spec["method"])
        & (df["label"] == spec["label"])
        & (df["arm"] == spec["arm"])
        & (df["split"] == spec["split"])
        & (df["model"] == spec["model"])
    ]
    b = df[
        (df["family"] == spec["family"])
        & (df["method"] == spec["method"])
        & (df["label"] == spec["label"])
        & (df["arm"] == spec["baseline_arm"])
        & (df["split"] == spec["split"])
        & (df["model"] == spec["baseline_model"])
    ]
    rows = []
    if len(m):
        rows.append({"role": "treatment", **m.iloc[0].to_dict()})
    if len(b):
        rows.append({"role": "baseline", **b.iloc[0].to_dict()})
    if rows:
        path = out_root / "primary_confirmatory.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"→ {path}  (CONFIRMATORY — pre-registered, metric={metric})")
        if len(m) and len(b) and metric in m.columns and metric in b.columns:
            diff = float(m.iloc[0][metric]) - float(b.iloc[0][metric])
            print(
                f"  PRIMARY Δ{metric}: {diff:+.4f}  "
                f"(treat={float(m.iloc[0][metric]):.4f} vs base={float(b.iloc[0][metric]):.4f})"
            )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="finalfinal train with per-family feature confirmation"
    )
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--list-datasets", action="store_true")
    p.add_argument("--list-features", action="store_true",
                   help="print proposed features per dataset and exit")
    p.add_argument("--confirm-features", action="store_true",
                   help="interactive y/N per family+method before train")
    p.add_argument("--yes", action="store_true",
                   help="noop; default already starts train (kept for old cmds)")
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                   help="aliases: ava,huawei,npj,oura,24,24agg,night,night24,all")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help="all | night linear,locf | 24/24agg e.g. linear_forward")
    p.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    p.add_argument("--features-file", type=Path, default=None,
                   help="JSON overrides: family/style → [cols]")
    p.add_argument(
        "--models-ml",
        default="top6",
        help="top6 = LightGBM,SVM,LR,DT (default) | all | proposed | comma names",
    )
    p.add_argument("--models-dl", default="none",
                   help="none (default) | top6 | all | comma names")
    p.add_argument("--splits", default="gs,sg")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cuda-device", type=int, default=3)
    p.add_argument("--quick", action="store_true",
                   help="ava+huawei, y2_mod, LightGBM, gs, methods=linear")
    p.add_argument(
        "--arms",
        default=",".join(COMPETE_ARMS),
        help="cal|physio|combined (default: all three compete)",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip (family,method,label,arm,split,group,model) already in summary.csv",
    )
    p.add_argument(
        "--cal-rule-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cal arm = strict rules cd14/cd15/hist_cl14 only (default: on). "
             "Use --no-cal-rule-only to restore ML/DL on calendar features.",
    )
    p.add_argument("--self-check", action="store_true")
    p.add_argument(
        "--finalize-only",
        action="store_true",
        help="skip training; write pivots/primary from existing summary.csv",
    )
    p.add_argument("--dump-features", type=Path, default=None,
                   help="write confirmed feature map JSON and exit (with list/confirm)")
    return p.parse_args()


def _self_check(data_root: Path) -> None:
    assert data_root.is_dir(), data_root
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    m = compute_metrics_macro(yt, yp, None, 2, ov_cls=0)
    assert abs(m["f1"] - _macro_f1(yt, yp)) < 1e-9
    Xtr = np.array([[10.0, 14.0], [12.0, 14.0], [20.0, 15.0]], dtype=np.float64)
    Xte = np.array([[11.0, 14.0], [99.0, 10.0]], dtype=np.float64)
    gtr = np.array(["A", "A", "B"])
    gte = np.array(["A", "Z"])
    Xtr2, Xte2 = patient_median_norm(
        Xtr, Xte, gtr, gte, [0],
        np.array([1, 1, 1]), np.array([5.0, 6.0, 20.0]),
        np.array([1, 2]), np.array([7.0, 1.0]),
    )
    assert abs(Xtr2[0, 1] - 14.0) < 1e-9
    assert abs(Xtr2[1, 0] - 2.0) < 1e-9
    assert abs(Xte2[0, 0]) < 1e-9
    # arm partition: cal ∩ physio = ∅, combined = union
    names = ["bbt", "day_of_cycle", "cal_off14", "rmssd"]
    c_i, p_i = set(cal_indices(names)), set(physio_indices(names))
    assert c_i.isdisjoint(p_i)
    assert m_cycle_group_of(29) == "regular"
    assert m_cycle_group_of(25) == "mild_irregular"
    assert m_cycle_group_of(37) == "high_irregular"
    assert set(arm_col_indices(names, "combined")) == c_i | p_i
    assert "menses_start" not in names  # raw menses date never a feat name here
    # combined blend: cal signal + physio noise → should pick cal (or cal-heavy)
    rng = np.random.RandomState(0)
    n = 240
    cal = rng.randn(n, 2)
    phy = rng.randn(n, 4)
    yb = (cal[:, 0] + 0.3 * cal[:, 1] > 0).astype(int)
    Xb = np.hstack([cal, phy])
    nb = ["day_of_cycle", "cal_off14", "bbt", "hr", "rmssd", "sdnn"]
    tr, te = np.arange(180), np.arange(180, 240)
    rb = train_combined_blend(
        "LR", Xb[tr], yb[tr], Xb[te], yb[te], nb, 2, ov_cls=1
    )
    assert rb["metrics"]["combined_pick"].startswith(("cal", "blend_cal"))
    for fam, (kind, style) in FAMILY_META.items():
        d = data_root / fam
        if not d.is_dir():
            continue
        files = list(d.rglob("y2_mod.xlsx"))
        if not files:
            continue
        cols = read_columns(files[0])
        hit, _ = propose_cal_physio(style, cols)
        assert hit, f"{fam}: empty feature hit"
        assert not (LEAK_COLS & set(hit)), f"{fam}: leak in features"
        cal_h, _ = propose_cal_only(cols)
        phy_h, _ = propose_physio_only(style, cols)
        assert not (set(cal_h) & set(phy_h))
        assert not (LEAK_COLS & set(cal_h)) and not (LEAK_COLS & set(phy_h))
        assert "menses_start" not in cal_h and "menses_start" not in phy_h
        if style == "npj":
            assert any(c.startswith("avgNN") for c in phy_h)
            assert not any(c.startswith("ibi") for c in phy_h)
        if style == "seq24":
            assert "hrv_mask" not in phy_h and "temp_mask" not in phy_h
        assert not (QUALITY_COLS & set(hit)), f"{fam}: quality in features"
        assert not (QUALITY_COLS & set(phy_h)), f"{fam}: quality in physio"
        assert kind in ("seq24", "night")
    found = discover(data_root, expand_families("all"), None, ["y2_mod"])
    methods = {s["method"] for s in found}
    families = {s["family"] for s in found}
    assert "locf" in methods, methods
    assert any(m.startswith(("linear_", "locf_")) for m in methods), methods
    assert "interp_24" in families and "interp_24agg" in families
    proposed._self_check()
    assert resolve_ml_models("top6") == TOP6_ML
    assert resolve_dl_models("top6") == TOP6_DL
    assert proposed.NAME in resolve_ml_models("")
    assert proposed.NAME in resolve_ml_models("all")
    assert resolve_dl_models("") == dl.list_models()
    from strict_cal_rule import estimate_ovulation, phase5, collapse_from_y5
    from datetime import date
    ov = estimate_ovulation("cd14", date(2023, 1, 1), "A", 1, {})
    assert ov == date(2023, 1, 14)
    assert phase5(date(2023, 1, 14), date(2023, 1, 1), date(2023, 1, 29), ov, "mod") == 3
    assert collapse_from_y5(3, "y2_mod") == 1
    # ragged summary.csv (cal vs combined keys) must still load
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "summary.csv"
    log = RunLogger(tmp.parent, resume=False)
    log.csv_path = tmp
    log.append({
        "arm": "cal", "family": "f", "method": "m", "label": "y5_mod",
        "split": "gs", "group": "rule", "model": "rule_cd14",
        "n_feats": 0, "feats": "", "patnorm": 0, "f1": 0.7, "ov_f1": 0.5,
        "ov_thr": "", "combined_pick": "rule_cd14",
    })
    log.append({
        "arm": "combined", "family": "f", "method": "m", "label": "y5_mod",
        "split": "sg", "group": "ml", "model": "DT",
        "n_feats": 2, "feats": "a,b", "patnorm": 1, "f1": 0.75, "ov_f1": 0.62,
        "ov_thr": 0.85, "combined_pick": "blend_cal0.85_union",
        "combined_val_f1": 0.66,
    })
    got = log.load_summary()
    assert len(got) == 2
    assert list(got.columns) == list(SUMMARY_COLS)
    assert got.iloc[1]["combined_pick"] == "blend_cal0.85_union"
    assert pd.api.types.is_float_dtype(got["f1"])
    print("grid self-check OK (sklearn top4 + incremental log + ov_f1 primary)")


def safe_train_torch(
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
    ov_cls: int,
    groups_tr: Optional[np.ndarray] = None,
) -> Optional[dict[str, Any]]:
    lr_use = dl.FOUNDATION_LR if model_name in dl.FOUNDATION_DL else lr
    early_stop = "macro" if model_name in dl.FOUNDATION_DL else "ov"
    seeds = range(DL_N_SEEDS) if group == "dl" else (0,)
    f1s: list[float] = []
    last: Optional[dict[str, Any]] = None
    try:
        for seed in seeds:
            res = gu.train_torch(
                model_name, group, X_tr, y_tr, X_te, y_te, n_classes,
                epochs=epochs, batch_size=batch_size, lr=lr_use, device=device,
                ov_cls=ov_cls, patience=max(20, epochs // 5), min_epochs=15,
                f1_boost=True, groups_tr=groups_tr, early_stop=early_stop,
                random_state=42 + int(seed),
            )
            f1s.append(float(res["metrics"]["f1"]))
            last = res
        if last is None:
            return None
        if len(f1s) > 1:
            last["metrics"]["f1"] = float(np.mean(f1s))
            last["metrics"]["f1_seed_std"] = float(np.std(f1s))
        return last
    except Exception as e:
        print(f"  [skip {group}/{model_name}] {e}")
        return None


def write_final_reports(
    runlog: RunLogger, out_root: Path, arms: list[str],
) -> None:
    big = runlog.load_summary()
    if big.empty:
        runlog.say("\n(no result rows)")
        return
    runlog.say(f"\n→ {runlog.csv_path}  ({len(big)} rows)")

    board_rows = []
    best_by_arm: dict[str, float] = {}
    for arm in arms:
        sub = big[big["arm"] == arm]
        if sub.empty:
            continue
        b = sub.loc[sub["f1"].idxmax()]
        best_by_arm[arm] = float(b["f1"])
        board_rows.append({
            "arm": arm,
            "best_macro_f1": float(b["f1"]),
            "family": b["family"],
            "model": b["model"],
            "split": b["split"],
        })
        runlog.say(
            f"  ARM {arm:8s} best macroF1={float(b['f1']):.4f}  "
            f"{b['family']}/{b['model']}/{b['split']}  "
            f"(exploratory — see primary_confirmatory.csv)"
        )
    board = out_root / "exploratory_arm_compete_board.csv"
    pd.DataFrame(board_rows).to_csv(board, index=False)
    runlog.say(f"→ {board}  (EXPLORATORY — idxmax over grid)")
    write_report_pivots(big, out_root, arms)
    write_primary_confirmatory(big, out_root)

    if "combined" in best_by_arm:
        c = best_by_arm["combined"]
        others = {a: v for a, v in best_by_arm.items() if a != "combined"}
        wins = all(c + 1e-12 >= v for v in others.values()) if others else True
        oth = {k: round(v, 4) for k, v in others.items()}
        runlog.say(
            f"\nEXPLORATORY combined≥others (global idxmax)? "
            f"{'YES' if wins else 'NO'}  "
            f"(combined={c:.4f}  others={oth})"
        )
    runlog.say(f"≥0.8 any? {'YES' if big['f1'].max() >= 0.8 else 'NO'}")


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check(args.data_root)
        return

    if args.finalize_only:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        out_root: Path = args.out_root
        out_root.mkdir(parents=True, exist_ok=True)
        runlog = RunLogger(out_root, resume=False)
        runlog.say("finalize-only: writing reports from existing summary.csv")
        write_final_reports(runlog, out_root, arms)
        return

    families = expand_families(args.families)
    if args.methods.strip().lower() == "all":
        methods = None
    else:
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if args.labels.strip().lower() == "all":
        labels = list(LABEL_FILES)
    else:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    if args.quick:
        families = expand_families("ava,huawei")
        methods = ["linear"]
        labels = ["y2_mod"]
        ml_models = ["LightGBM"]
        dl_models: list[str] = []
        splits = ["gs"]
        epochs = 5
        args.yes = True
    else:
        ml_models = resolve_ml_models(args.models_ml)
        dl_models = resolve_dl_models(args.models_dl)
        splits = [x.strip() for x in args.splits.split(",") if x.strip()]
        epochs = args.epochs


    skip_dl: list[tuple[str, str]] = []
    ok_dl: list[str] = []
    for n in dl_models:
        avail, why = dl.model_available(n)
        if avail:
            ok_dl.append(n)
        else:
            skip_dl.append((n, why))
    dl_models = ok_dl
    if skip_dl:
        print("DL models skipped (missing deps):")
        for n, why in skip_dl:
            print(f"  {n}: {why}")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in COMPETE_ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {COMPETE_ARMS}")

    fmap = load_features_file(args.features_file) if args.features_file else {}
    sources = discover(args.data_root, families, methods, labels)
    if not sources:
        raise SystemExit(
            f"no datasets under {args.data_root} "
            f"families={families} methods={methods} labels={labels}"
        )

    if args.list_datasets:
        print(f"data-root: {args.data_root}")
        for s in sources:
            print(f"  {s['key']}")
        print(f"total={len(sources)}")
        return

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in sources:
        groups.setdefault((s["family"], s["method"]), []).append(s)

    confirmed: dict[tuple[str, str], list[str]] = {}
    dump: dict[str, list[str]] = {}

    for (fam, method), items in groups.items():
        probe = items[0]
        present = read_columns(probe["path"])
        ov = override_for(probe, fmap)
        feats, miss = propose_cal_physio(probe["style"], present, ov)
        if not feats:
            raise SystemExit(f"{probe['key']}: no features after ∩ present")
        assert not (LEAK_COLS & set(feats)), LEAK_COLS & set(feats)
        cal_f, _ = propose_cal_only(present)
        phy_f, _ = propose_physio_only(probe["style"], present, ov)

        if args.list_features or args.dump_features:
            print_feature_plan(probe, feats, miss)
            print(f"  arm/cal:      {cal_f or '(none — need day_of_cycle)'}")
            print(f"  arm/physio:   n={len(phy_f)}")
            print(f"  arm/combined: physio∪cal n={len(feats)} before eng")
            print("  note: raw menses_start NOT a feature; day_of_cycle = f(menses_start)")
            dump[f"{fam}/{method}"] = feats
            confirmed[(fam, method)] = feats
            continue

        if args.confirm_features and not args.yes:
            print_feature_plan(probe, feats, miss)
            print(f"  arms={arms}  cal={cal_f}  physio_n={len(phy_f)}")
            if not confirm_features(probe, feats, miss):
                print("  → skipped")
                continue
        else:
            print_feature_plan(probe, feats, miss)
            print(f"  arms={arms}  cal={cal_f or '(none)'}  physio_n={len(phy_f)}")
            print("  → accepted")
        confirmed[(fam, method)] = feats
        dump[f"{fam}/{method}"] = feats

    if args.dump_features:
        args.dump_features.parent.mkdir(parents=True, exist_ok=True)
        args.dump_features.write_text(
            json.dumps(dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.dump_features}")
        return

    if args.list_features:
        return

    if not confirmed:
        raise SystemExit("no datasets confirmed")

    sources = [s for s in sources if (s["family"], s["method"]) in confirmed]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    device = gu.resolve_device(0)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    runlog = RunLogger(out_root, resume=args.resume)

    runlog.say("=" * 70)
    runlog.say("  grid  |  ML=LightGBM,SVM,LR,DT  |  ov_f1 primary  |  → " + str(out_root))
    runlog.say(f"  arms={arms}  (goal: combined ≥ best rule on ov_f1)")
    runlog.say(f"  datasets={len(sources)}  ML={ml_models}  DL={dl_models}  splits={splits}")
    if args.resume and runlog.done:
        runlog.say(f"  resume: skipping {len(runlog.done)} completed runs")
    if args.cal_rule_only:
        runlog.say("  cal mode: STRICT rules cd14/cd15/hist_cl14 (no ML / no cal_* / no day-prior)")
    else:
        runlog.say("  cal mode: ML/DL on calendar features (legacy)")
    runlog.say(f"  device={device}  epochs={epochs}")
    runlog.say("=" * 70)

    for s in sources:
        feats = confirmed[(s["family"], s["method"])]
        lab = s["label"]
        path = s["path"]
        runlog.say(f"\n== {s['key']} load_feats={len(feats)} ==")

        if s["kind"] == "seq24":
            _, X_tab, y, groups_a, used, mcg, Q, qn, cyc_a, cd_a, cal_meta = load_seq24(
                path, lab, feats
            )
        else:
            X_tab, y, groups_a, used, mcg, Q, qn, cyc_a, cd_a, cal_meta = load_night(
                path, lab, feats
            )
        X_tab, used = append_cal_eng(X_tab, used)
        assert not (LEAK_COLS & set(used)), LEAK_COLS & set(used)
        assert not (META_COLS & set(used)), META_COLS & set(used)
        assert not (QUALITY_COLS & set(used)), QUALITY_COLS & set(used)
        assert len(Q) == len(y)
        assert not (LEAK_COLS & set(used))

        n_classes = int(y.max() + 1)
        ov_cls = gu.ovulation_class_id(lab, n_classes)
        vc = pd.Series(mcg).value_counts().to_dict()
        runlog.say(
            f"  N={len(y)} classes={n_classes} ov_cls={ov_cls} after_eng={len(used)}  "
            f"m_cycle_group={vc}  quality_gate={qn or '∅'}"
        )
        # reuse cal/physio torch mon+te probas for combined late fusion
        torch_stash: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        hist_cl = build_hist_cl_map(
            cal_meta["pid"], cal_meta["cyc"], cal_meta["msd"], cal_meta["nmsd"]
        )
        rule_all = (
            build_strict_rule_probas(cal_meta, lab, n_classes, hist_cl)
            if args.cal_rule_only else {}
        )

        for arm in arms:
            if arm == "cal" and args.cal_rule_only:
                runlog.say("  [cal_rule_only] strict cd14/cd15/hist_cl14 (no ML/DL)")
                for method in STRICT_CAL_METHODS:
                    for split in splits:
                        rk = runlog.result_key(
                            s["family"], s["method"], lab, arm, split,
                            "rule", f"rule_{method}",
                        )
                        if runlog.is_done(rk):
                            continue
                        try:
                            tr, te = gu.split_gs_sg(groups_a, split)
                        except RuntimeError as e:
                            runlog.say(f"  [skip split {split}] {e}")
                            continue
                        keep, pred_all = predict_strict_cal(
                            method, lab,
                            cal_meta["pid"], cal_meta["cyc"],
                            cal_meta["msd"], cal_meta["nmsd"], cal_meta["day"],
                            hist_cl=hist_cl,
                        )
                        te_m = te & keep
                        if te_m.sum() < 2:
                            runlog.say(f"  [skip cal/{method} {split}] too few scored nights")
                            continue
                        yte = y[te_m]
                        pred = pred_all[te_m]
                        gte = groups_a[te_m]
                        mcg_te = mcg[te_m]
                        m = metrics_with_m_cycle_subgroups(
                            yte, pred, None, n_classes, ov_cls, mcg_te
                        )
                        m.update(cluster_bootstrap_f1(yte, pred, gte))
                        m["ov_thr"] = float("nan")
                        m["combined_pick"] = f"rule_{method}"
                        runlog.append({
                            "arm": arm,
                            "family": s["family"], "method": s["method"], "label": lab,
                            "split": split, "group": "rule", "model": f"rule_{method}",
                            "n_feats": 0, "feats": "",
                            "patnorm": 0,
                            **{k: m[k] for k in m if k != "confusion"},
                        })
                        runlog.say(
                            f"  [{arm:8s}] rule/{method:10s} {split}  "
                            f"macroF1={m.get('f1', float('nan')):.4f}  "
                            f"ov_f1={m.get('ov_f1', float('nan')):.3f}  "
                            f"n_te={int(te_m.sum())}"
                        )
                continue

            idxs = arm_col_indices(used, arm)
            if not idxs:
                print(f"  [skip arm={arm}] no columns")
                continue
            arm_names = [used[i] for i in idxs]
            X_arm = X_tab[:, idxs]
            if arm in ("physio", "combined"):
                local_physio = [j for j, n in enumerate(arm_names) if not is_cal_name(n)]
            else:
                local_physio = []
            dl_active = [
                n for n in dl_models
                if not (arm == "cal" and n in dl.FOUNDATION_DL)
            ]
            if arm == "cal":
                skip_f = [n for n in dl_models if n in dl.FOUNDATION_DL]
                if skip_f:
                    print(
                        f"  [skip cal/dl] foundation TS (calendar tabular): {skip_f}"
                    )

            for split in splits:
                try:
                    tr, te = gu.split_gs_sg(groups_a, split)
                except RuntimeError as e:
                    print(f"  [skip split {split}] {e}")
                    continue
                Xtr, Xte = X_arm[tr], X_arm[te]
                ytr, yte = y[tr], y[te]
                gtr, gte = groups_a[tr], groups_a[te]
                cyc_tr, cyc_te = cyc_a[tr], cyc_a[te]
                cd_tr, cd_te = cd_a[tr], cd_a[te]
                if local_physio:
                    Xtr, Xte = patient_median_norm(
                        Xtr, Xte, gtr, gte, local_physio,
                        cyc_tr, cd_tr, cyc_te, cd_te,
                    )
                Xtr_t, Xte_t = Xtr[:, None, :], Xte[:, None, :]
                rule_proba = {
                    method: (p[tr], p[te]) for method, (_, p) in rule_all.items()
                } if rule_all else None
                strict_kw = None
                if args.cal_rule_only and arm == "combined":
                    keys = ("pid", "cyc", "msd", "nmsd", "day")
                    strict_kw = {
                        "label": lab,
                        "meta_tr": {k: cal_meta[k][tr] for k in keys},
                        "meta_te": {k: cal_meta[k][te] for k in keys},
                        "hist_cl": hist_cl,
                    }

                for name in ml_models:
                    rk = runlog.result_key(
                        s["family"], s["method"], lab, arm, split, "ml", name,
                    )
                    if runlog.is_done(rk):
                        continue
                    if name == proposed.NAME:
                        res = proposed.train(
                            Xtr, ytr, Xte, yte, arm_names, n_classes, ov_cls,
                            groups_tr=gtr, groups_te=gte,
                            quality_tr=Q[tr], quality_te=Q[te], quality_names=qn,
                            strict_cal=strict_kw,
                        )
                        pred, proba = res["pred"], res.get("proba")
                    elif name in SKLEARN_ML:
                        if arm == "combined":
                            res = train_combined_blend(
                                name, Xtr, ytr, Xte, yte, arm_names,
                                n_classes, ov_cls, groups_tr=gtr,
                                rule_proba=rule_proba,
                            )
                        else:
                            res = train_sklearn_macro(
                                name, Xtr, ytr, Xte, yte, n_classes, ov_cls,
                                groups_tr=gtr,
                            )
                        pred, proba = res["pred"], res.get("proba")
                    else:
                        # torch ML (LSTM…): stash + late-fuse on combined
                        res = safe_train_torch(
                            name, "ml", Xtr_t, ytr, Xte_t, yte, n_classes,
                            epochs, args.batch_size, args.lr, device, ov_cls,
                            groups_tr=gtr,
                        )
                        if res is None:
                            continue
                        pred = res.get("pred")
                        proba = res.get("proba")
                        if proba is not None and res.get("proba_mon") is not None:
                            thr = optimize_threshold_ov(
                                res["y_mon"], res["proba_mon"], ov_cls
                            )
                            pred = apply_thr(proba, ov_cls, thr)
                            res["metrics"]["ov_thr"] = float(thr)
                            torch_stash[(split, "ml", name, arm)] = {
                                "proba": proba,
                                "proba_mon": res["proba_mon"],
                                "y_mon": res["y_mon"],
                            }
                        if arm == "combined" and proba is not None:
                            cal_b = torch_stash.get((split, "ml", name, "cal"))
                            phy_b = torch_stash.get((split, "ml", name, "physio"))
                            if cal_b and phy_b and res.get("proba_mon") is not None:
                                fused = pick_cal_heavy_blend(
                                    {
                                        "cal": cal_b["proba_mon"],
                                        "physio": phy_b["proba_mon"],
                                        "union": res["proba_mon"],
                                    },
                                    {
                                        "cal": cal_b["proba"],
                                        "physio": phy_b["proba"],
                                        "union": proba,
                                    },
                                    cal_b["y_mon"],
                                    ov_cls,
                                )
                                pred, proba = fused["pred"], fused["proba"]
                                res["metrics"].update({
                                    "combined_pick": fused["combined_pick"],
                                    "combined_val_f1": fused["combined_val_f1"],
                                    "ov_thr": fused["ov_thr"],
                                })
                        if pred is None and proba is not None:
                            pred = proba.argmax(1)
                    mcg_te = mcg[te]
                    if pred is not None:
                        m = metrics_with_m_cycle_subgroups(
                            yte, pred, proba, n_classes, ov_cls, mcg_te
                        )
                        m.update(cluster_bootstrap_f1(yte, pred, gte))
                        # keep blend diagnostics from res.metrics
                        for k in ("combined_pick", "combined_val_f1", "ov_thr"):
                            if k in res["metrics"]:
                                m[k] = res["metrics"][k]
                    else:
                        m = res["metrics"]
                    runlog.append({
                        "arm": arm,
                        "family": s["family"], "method": s["method"], "label": lab,
                        "split": split, "group": "ml", "model": name,
                        "n_feats": len(arm_names), "feats": ",".join(arm_names),
                        "patnorm": int(bool(local_physio)),
                        **{k: m[k] for k in m if k != "confusion"},
                    })
                    pick = m.get("combined_pick", "")
                    pick_s = f"  pick={pick}" if pick else ""
                    runlog.say(
                        f"  [{arm:8s}] ml/{name:12s} {split}  "
                        f"macroF1={m.get('f1', float('nan')):.4f}  "
                        f"ov_f1={m.get('ov_f1', float('nan')):.3f}  "
                        f"f1_reg={m.get('f1_regular', float('nan')):.3f}  "
                        f"f1_mild={m.get('f1_mild_irregular', float('nan')):.3f}  "
                        f"f1_high={m.get('f1_high_irregular', float('nan')):.3f}"
                        f"{pick_s}"
                    )

                for name in dl_active:
                    rk = runlog.result_key(
                        s["family"], s["method"], lab, arm, split, "dl", name,
                    )
                    if runlog.is_done(rk):
                        continue
                    res = safe_train_torch(
                        name, "dl", Xtr_t, ytr, Xte_t, yte, n_classes,
                        epochs, args.batch_size, args.lr, device, ov_cls,
                        groups_tr=gtr,
                    )
                    if res is None:
                        continue
                    pred = res.get("pred")
                    proba = res.get("proba")
                    if proba is not None and res.get("proba_mon") is not None:
                        thr = optimize_threshold_ov(
                            res["y_mon"], res["proba_mon"], ov_cls
                        )
                        pred = apply_thr(proba, ov_cls, thr)
                        res["metrics"]["ov_thr"] = float(thr)
                        torch_stash[(split, "dl", name, arm)] = {
                            "proba": proba,
                            "proba_mon": res["proba_mon"],
                            "y_mon": res["y_mon"],
                        }
                    if arm == "combined" and proba is not None:
                        cal_b = torch_stash.get((split, "dl", name, "cal"))
                        phy_b = torch_stash.get((split, "dl", name, "physio"))
                        if cal_b and phy_b and res.get("proba_mon") is not None:
                            fused = pick_cal_heavy_blend(
                                {
                                    "cal": cal_b["proba_mon"],
                                    "physio": phy_b["proba_mon"],
                                    "union": res["proba_mon"],
                                },
                                {
                                    "cal": cal_b["proba"],
                                    "physio": phy_b["proba"],
                                    "union": proba,
                                },
                                cal_b["y_mon"],
                                ov_cls,
                            )
                            pred, proba = fused["pred"], fused["proba"]
                            res["metrics"].update({
                                "combined_pick": fused["combined_pick"],
                                "combined_val_f1": fused["combined_val_f1"],
                                "ov_thr": fused["ov_thr"],
                            })
                    if pred is None and proba is not None:
                        pred = proba.argmax(1)
                    mcg_te = mcg[te]
                    m = (
                        metrics_with_m_cycle_subgroups(
                            yte, pred, proba, n_classes, ov_cls, mcg_te
                        )
                        if pred is not None
                        else res["metrics"]
                    )
                    if pred is not None:
                        m.update(cluster_bootstrap_f1(yte, pred, gte))
                    for k in ("combined_pick", "combined_val_f1", "ov_thr"):
                        if k in res["metrics"]:
                            m[k] = res["metrics"][k]
                    runlog.append({
                        "arm": arm,
                        "family": s["family"], "method": s["method"], "label": lab,
                        "split": split, "group": "dl", "model": name,
                        "n_feats": len(arm_names), "feats": ",".join(arm_names),
                        "patnorm": int(bool(local_physio)),
                        **{k: m[k] for k in m if k != "confusion"},
                    })
                    pick = m.get("combined_pick", "")
                    pick_s = f"  pick={pick}" if pick else ""
                    runlog.say(
                        f"  [{arm:8s}] dl/{name:12s} {split}  "
                        f"macroF1={m.get('f1', float('nan')):.4f}  "
                        f"ov_f1={m.get('ov_f1', float('nan')):.3f}"
                        f"{pick_s}"
                    )

    write_final_reports(runlog, out_root, arms)


if __name__ == "__main__":
    main()
