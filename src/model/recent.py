# -*- coding: utf-8 -*-
"""
Puts four recent models proposed in 2024--2025 on the same evaluation framework.

Why these four. The existing comparators (4 prior ovulation studies + LightGBM) all predate 2025,
and the only real learners among them are random forest and GBDT. To answer "how does it compare
with recent models" we need **recent baselines for this data shape**. The data are
1,269 training nights x 18~26 columns — a small tabular dataset that is also a short time series
of nights lined up within each cycle. So we pick three tabular models + one sequence model.

  [A] TabPFN v2   Hollmann et al., Nature 637:319 (2025). Pretrained transformer for tabular
                  data. Reported to beat tuned GBDT below 10k rows, which is exactly
                  where this dataset sits. Having no hyperparameters is by design.
  [B] TabICL      Qu et al., ICML 2025. Same in-context family, but designed to exceed
                  TabPFN's row/column limits. Keeping both shows the family's performance
                  rather than the quirks of one implementation.
  [C] TabM        Gorishniy et al., ICLR 2025. Parameter-sharing MLP ensemble.
                  The 2025 reference point where tabular deep learning reached GBDT.
  [D] Mamba(S6)   Gu & Dao 2024. Selective state space. The most widely used new architecture
                  in 2024--25 biosignal (ECG/EEG/PPG) literature. Takes the preceding nights
                  within the cycle as a window and classifies — tests whether the model can
                  learn by itself the within-cycle summaries OvCast builds by hand.

**The comparison protocol is unchanged.** Same input matrix, same labels, Hospital A -> Hospital B external validation,
same metrics. Models that build windows do that transform **causally** inside the estimator —
night t only sees nights <= t of its cycle. Same role as Kilungeja2025 in prior.py.

**GPU**. [A][B][C][D] all run on GPU. On CPU, TabPFN refuses more than 1,000 rows
and TabM takes 60 s per fit, making 8,000 bootstrap replicates infeasible.
Each worker selects its device via OVUL_GPU.
"""
from __future__ import annotations

import os
import numpy as np

from .prior import _by_cycle, _Struct


def _device():
    """Device for this worker. The runner sets the index in OVUL_GPU."""
    g = os.environ.get("OVUL_GPU", "").strip()
    return f"cuda:{g}" if g else ("cuda" if _torch().cuda.is_available() else "cpu")


def _torch():
    import torch
    return torch


def causal_windows(X, cyc, day, L):
    """For each night t, stack nights max(0,t-L+1)..t of its cycle as (L, F).

    If the left side is short, pad with the first night of the cycle — padding with 0 would make the model learn
    "early cycle" as a missingness pattern instead of a feature. Future nights are never seen.
    """
    X = np.asarray(X, float)
    out = np.zeros((len(X), L, X.shape[1]), dtype=np.float32)
    for _, m in _by_cycle(np.asarray(cyc), np.asarray(day, float)):
        V = X[m]
        for t in range(len(m)):
            w = V[max(0, t - L + 1): t + 1]
            if len(w) < L:
                w = np.vstack([np.repeat(w[:1], L - len(w), axis=0), w])
            out[m[t]] = w
    return out


class _InContext(_Struct):
    """Shared by TabPFN / TabICL. Fitting is just loading context, so there are no hyperparameters."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def _make(self):                                  # implemented by subclasses
        raise NotImplementedError

    def fit(self, X, y):
        X, y = np.asarray(X, np.float32), np.asarray(y)
        self.m_ = self._make()
        self.m_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X, np.float32))

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, np.float32))


class TabPFNv2(_InContext):
    """[A] Hollmann et al., Nature 2025."""

    def _make(self):
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=_device(), random_state=self.random_state,
                                ignore_pretraining_limits=True)


class TabICL(_InContext):
    """[B] Qu et al., ICML 2025."""

    def _make(self):
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=_device(), random_state=self.random_state)


class _TorchClf(_Struct):
    """Shared training loop for TabM / Mamba.

    Standardization is fit inside the estimator on training data only — the outer pipeline is fit
    to the training partition, but the training sample changes with every bootstrap replicate, so
    refitting within each replicate leaves no path for leakage.
    """

    def __init__(self, epochs=100, lr=2e-3, wd=3e-4, batch=256, random_state=42):
        self.epochs, self.lr, self.wd = epochs, lr, wd
        self.batch, self.random_state = batch, random_state

    def _net(self, n_feat, n_cls):                    # implemented by subclasses
        raise NotImplementedError

    def _prep(self, X, fit=False):                    # implemented by subclasses
        raise NotImplementedError

    def fit(self, X, y):
        torch = _torch()
        torch.manual_seed(self.random_state)
        A = self._prep(X, fit=True)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        idx = {c: i for i, c in enumerate(self.classes_)}
        yt = torch.tensor([idx[v] for v in y], dtype=torch.long)
        dev = _device()
        self.net_ = self._net(A.shape[-1], len(self.classes_)).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.wd)
        At, yt = torch.tensor(A).to(dev), yt.to(dev)
        n = len(At)
        g = torch.Generator(device="cpu").manual_seed(self.random_state)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                opt.zero_grad()
                o = self.net_(At[b])
                loss = torch.nn.functional.cross_entropy(o, yt[b])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        torch = _torch()
        A = torch.tensor(self._prep(X)).to(_device())
        self.net_.eval()
        with torch.no_grad():
            p = torch.softmax(self.net_(A), -1).cpu().numpy()
        return p

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class TabMClf(_TorchClf):
    """[C] Gorishniy et al., ICLR 2025. The prediction is the mean of k heads.

    epochs was chosen by validation within Hospital A (4-class, 2 splits) — 100 epochs 0.766 vs
    200 epochs 0.749. Training longer drops due to overfitting. Same place and same procedure
    used to settle the uncertain settings of the prior models (tune_prior_models).
    """

    def __init__(self, k=32, d_block=256, n_blocks=3, epochs=100, **kw):
        kw.setdefault("epochs", epochs)
        super().__init__(**kw)
        self.k, self.d_block, self.n_blocks = k, d_block, n_blocks

    def _prep(self, X, fit=False):
        X = np.asarray(X, np.float32)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def _net(self, n_feat, n_cls):
        from tabm import TabM
        torch = _torch()
        # TabM.make takes flat kwargs (not a backbone dict).
        core = TabM.make(n_num_features=n_feat, cat_cardinalities=[],
                         d_out=n_cls, k=self.k, n_blocks=self.n_blocks,
                         d_block=self.d_block, dropout=0.1)

        class _Mean(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.core = core

            def forward(s, x):
                return s.core(x).mean(1)       # (B, k, C) -> (B, C)

        return _Mean()


class MambaClf(_TorchClf):
    """[D] Gu & Dao 2024. Takes the preceding L nights within the cycle and classifies the last night.

    win and epochs were chosen by validation within Hospital A (4-class, 2 splits):
      win=5/80 epochs 0.774 · win=10/80 epochs 0.768 · win=5/200 epochs 0.772.
    200 epochs is no better than 80 yet 2.7x slower, so the shorter run wins on both performance and cost.
    A 5-night window also matches the moving windows (1,3,5) chosen by Kilungeja.
    """

    def __init__(self, win=5, d_model=64, n_layers=2, epochs=80, **kw):
        kw.setdefault("epochs", epochs)
        super().__init__(**kw)
        self.win, self.d_model, self.n_layers = win, d_model, n_layers

    def _prep(self, X, fit=False):
        X = np.asarray(X, np.float32)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Z = ((X - self.mu_) / self.sd_).astype(np.float32)
        return causal_windows(Z, self.cyc_, self.day_, self.win)

    def _net(self, n_feat, n_cls):
        from mambapy.mamba import Mamba, MambaConfig
        torch = _torch()

        class _Seq(torch.nn.Module):
            def __init__(s, f, c, d, L):
                super().__init__()
                s.inp = torch.nn.Linear(f, d)
                s.mamba = Mamba(MambaConfig(d_model=d, n_layers=L))
                s.norm = torch.nn.LayerNorm(d)
                s.head = torch.nn.Linear(d, c)

            def forward(s, x):                 # (B, T, F)
                h = s.mamba(s.inp(x))
                return s.head(s.norm(h[:, -1]))   # classify from the last night's representation

        return _Seq(n_feat, n_cls, self.d_model, self.n_layers)


MODELS = {"TabPFNv2": TabPFNv2, "TabICL": TabICL,
          "TabM": TabMClf, "Mamba": MambaClf}


# ───────────────────────── 2024~25 additions (not pretrained, trained on this data) ──────
#
# TabPFN·TabICL are foundation models **pretrained** on synthetic data, which breaks the comparison
# premise of "same feature matrix, only the learning algorithm differs". The four below are all fit
# from scratch on this cohort, so they stand on the same footing as LightGBM·OvCast.

class KANClf(_TorchClf):
    """[E] Kolmogorov-Arnold Networks (Liu et al. 2024).

    Replaces the MLP's fixed activations with learnable splines on the edges. Widely used in
    2024~25 ECG/EEG classification literature. Takes one night as one row.
    """

    def __init__(self, hidden=16, grid=3, k=3, epochs=50, **kw):
        # epochs chosen by Hospital A validation (4-class, 2 splits):
        # 50 epochs 0.818 · 100 epochs 0.802 · 200 epochs 0.779. Training longer drops due to overfitting.
        kw.setdefault("epochs", epochs)
        super().__init__(**kw)
        self.hidden, self.grid, self.k = hidden, grid, k

    def _prep(self, X, fit=False):
        X = np.asarray(X, np.float32)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def _net(self, n_feat, n_cls):
        """Kept in FastKAN (Li 2024) form — a KAN whose edge functions are expanded in RBFs.

        The original implementation (pykan) takes 95 s per fit, making 8,000 bootstrap replicates
        infeasible. The slow part is the grid-update/symbolification machinery, not the expression itself. As Li
        (2024) showed, replacing B-splines with Gaussian RBFs computes the same function family
        with pure matrix multiplications. That is the form used here.
        """
        torch = _torch()
        grid = self.grid

        class _RBFKANLayer(torch.nn.Module):
            def __init__(s, i, o, g):
                super().__init__()
                s.register_buffer("c", torch.linspace(-2, 2, g))
                s.h = 4.0 / (g - 1)
                s.w = torch.nn.Linear(i * g, o)
                s.base = torch.nn.Linear(i, o)

            def forward(s, x):
                phi = torch.exp(-((x[..., None] - s.c) / s.h) ** 2)
                return s.w(phi.flatten(-2)) + s.base(torch.nn.functional.silu(x))

        return torch.nn.Sequential(
            _RBFKANLayer(n_feat, self.hidden, grid),
            _RBFKANLayer(self.hidden, n_cls, grid))


class ConvTranClf(_TorchClf):
    """[F] ConvTran (Foumani et al., Data Min. Knowl. Discov. 2024).

    A transformer dedicated to multivariate time-series **classification**. Its key point is positional
    encoding adapted to time series. Takes the same within-cycle window as Mamba and classifies the last night.
    """

    def __init__(self, win=5, d_model=64, **kw):
        super().__init__(**kw)
        self.win, self.d_model = win, d_model

    def _prep(self, X, fit=False):
        X = np.asarray(X, np.float32)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Z = ((X - self.mu_) / self.sd_).astype(np.float32)
        # tsai uses the (B, C, L) convention. causal_windows returns (B, L, F).
        return np.ascontiguousarray(
            causal_windows(Z, self.cyc_, self.day_, self.win).transpose(0, 2, 1))

    def _net(self, n_feat, n_cls):
        from tsai.models.all import ConvTranPlus
        return ConvTranPlus(n_feat, n_cls, self.win, d_model=self.d_model)


class _PyTabKit(_Struct):
    """[G][H] pytabkit family. Uses the sklearn interface as is."""

    CLS = None

    def __init__(self, random_state=42, n_epochs=128):
        # n_epochs chosen by Hospital A validation: 64 epochs 0.770 · 128 epochs 0.774 · 256 epochs 0.759.
        self.random_state, self.n_epochs = random_state, n_epochs

    def fit(self, X, y):
        from pytabkit import RealMLP_TD_Classifier, RealTabR_D_Classifier  # noqa
        cls = {"RealMLP": RealMLP_TD_Classifier,
               "RealTabR": RealTabR_D_Classifier}[self.CLS]
        self.m_ = cls(random_state=self.random_state, device=_device(),
                      n_epochs=self.n_epochs)
        self.m_.fit(np.asarray(X, np.float32), np.asarray(y))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X, np.float32))

    def predict_proba(self, X):
        return self.m_.predict_proba(np.asarray(X, np.float32))


class RealMLPClf(_PyTabKit):
    """[G] RealMLP (Holzmuller et al., NeurIPS 2024). MLP with tuned defaults."""
    CLS = "RealMLP"


class RealTabRClf(_PyTabKit):
    """[H] RealTabR (Holzmuller et al. 2024; based on TabR, ICLR 2024).

    Tabular model with neighbor retrieval built into the network. Not pretrained — it searches
    neighbors only within this data's training sample, unlike a foundation model.
    """
    CLS = "RealTabR"


MODELS.update({"KAN": KANClf, "ConvTran": ConvTranClf,
               "RealMLP": RealMLPClf, "RealTabR": RealTabRClf})


class ExcelFormerClf(_Struct):
    """[I] ExcelFormer (Chen et al., KDD 2024). Transformer for tabular data.

    Treats columns as tokens and mixes them with semi-permeable attention. Trained from random
    initialization on this cohort only, so it is not a pretrained model — that is where it
    differs from TabPFN·TabICL.

    Standardization statistics (col_stats) are taken **from training data only** and applied to the test set as is.
    Recomputing them on the test set would itself be leakage.
    """

    def __init__(self, d=64, n_layers=3, n_heads=4, epochs=100, lr=1e-3,
                 random_state=42):
        # epochs·lr chosen by Hospital A validation: 100/1e-3 0.786 is best; 300·800 epochs
        # are lower. Before the label-encoding bug was fixed, every setting scored 0.19.
        self.d, self.n_layers, self.n_heads = d, n_layers, n_heads
        self.epochs, self.lr, self.random_state = epochs, lr, random_state

    def _ds(self, X, y, col_stats=None):
        import pandas as pd
        from torch_frame import stype
        from torch_frame.data import Dataset
        X = np.asarray(X, np.float32)
        cols = [f"c{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["y"] = np.asarray(y)
        d = Dataset(df, col_to_stype={**{c: stype.numerical for c in cols},
                                      "y": stype.categorical}, target_col="y")
        d.materialize(col_stats=col_stats)
        return d

    def fit(self, X, y):
        torch = _torch()
        torch.manual_seed(self.random_state)
        from torch_frame.nn.models import ExcelFormer
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        idx = {c: i for i, c in enumerate(self.classes_)}
        self.ds_ = self._ds(X, [idx[v] for v in y])
        dev = _device()
        tf = self.ds_.tensor_frame.to(dev)
        self.net_ = ExcelFormer(in_channels=self.d, out_channels=len(self.classes_),
                                num_cols=np.asarray(X).shape[1],
                                num_layers=self.n_layers, num_heads=self.n_heads,
                                col_stats=self.ds_.col_stats,
                                col_names_dict=tf.col_names_dict).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr)
        # The target uses **our own encoding**. torch_frame encodes categorical targets by
        # frequency (mode = 0), but predictions were mapped back via sorted classes_. So even
        # re-predicting the training data gave an exactly reversed distribution and macro 0.000.
        yt = torch.tensor([idx[v] for v in y], dtype=torch.long, device=dev)
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.net_(tf), yt)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        torch = _torch()
        d = self._ds(X, np.zeros(len(np.asarray(X)), int),
                     col_stats=self.ds_.col_stats)
        tf = d.tensor_frame.to(_device())
        self.net_.eval()
        with torch.no_grad():
            return torch.softmax(self.net_(tf), -1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


MODELS["ExcelFormer"] = ExcelFormerClf
