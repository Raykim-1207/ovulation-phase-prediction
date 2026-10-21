# -*- coding: utf-8 -*-
"""
sota_model_ml.py — faithful implementations of published models (classic ML + LSTM family).

Common I/O (torch):
  x    : (B, T, F)
  mask : (B, T) bool optional (True=valid)
  out  : logits (B, num_classes)

Papers / references
-----------
  LR, SVM, DT, LightGBM : standard sklearn / LightGBM implementations
  LSTM / StackedLSTM    : Hochreiter & Schmidhuber 1997; stacked = Graves et al.
  BiLSTM-Att            : BiLSTM + Bahdanau attention (Bahdanau et al. 2015)
  TPA-LSTM              : Shih et al., MLJ 2019 / arXiv:1809.04206
                          (CNN on hidden-state rows → temporal pattern attention)
  ME-LSTM               : Mixed-effect RNN (fixed LSTM + patient random intercept)
  Temporal Fusion Trans.: Lim et al. 2021 (GRN, VSN, LSTM, interpretable MHA)
                          + next-step forecast aux → classification head
  DeepAR                : Salinas et al. 2020 (AR LSTM → Gaussian NLL aux + clf on h_T)
"""
from __future__ import annotations

from typing import Any, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import ForecastAuxMixin, init_lstm, pool_sequence

# =============================================================================
# Tabular (library-faithful)
# =============================================================================


def build_lr(num_classes: int = 2, C: float = 1.0, max_iter: int = 2000, **kw: Any):
    del num_classes
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=C, max_iter=max_iter, class_weight="balanced", n_jobs=-1, **kw
    )


def build_svm(num_classes: int = 2, C: float = 1.0, kernel: str = "rbf", **kw: Any):
    del num_classes
    from sklearn.svm import SVC

    return SVC(C=C, kernel=kernel, probability=True, class_weight="balanced", **kw)


def build_dt(num_classes: int = 2, max_depth: int = 8, **kw: Any):
    del num_classes
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(
        max_depth=max_depth, class_weight="balanced", random_state=42, **kw
    )


def build_lightgbm(
    num_classes: int = 2,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    **kw: Any,
):
    import lightgbm as lgb

    if num_classes <= 2:
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            objective="binary",
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
            **kw,
        )
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        objective="multiclass",
        num_class=num_classes,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
        **kw,
    )


SKLEARN_BUILDERS = {
    "LR": build_lr,
    "SVM": build_svm,
    "DT": build_dt,
    "LightGBM": build_lightgbm,
}


# =============================================================================
# LSTM / StackedLSTM
# =============================================================================


class LSTMClassifier(nn.Module):
    """Vanilla LSTM classifier (last hidden → Linear)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pool: str = "last",
    ):
        super().__init__()
        self.pool = pool
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden_dim * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, num_classes)
        init_lstm(self.lstm)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        o, _ = self.lstm(x)
        return self.fc(self.drop(pool_sequence(o, mask, self.pool)))


class StackedLSTMClassifier(nn.Module):
    """Stacked LSTM (num_layers>=2), Graves-style deep RNN."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        pool: str = "last",
    ):
        super().__init__()
        self.core = LSTMClassifier(
            input_dim,
            num_classes,
            hidden_dim=hidden_dim,
            num_layers=max(2, num_layers),
            dropout=dropout,
            bidirectional=bidirectional,
            pool=pool,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kw: Any) -> torch.Tensor:
        return self.core(x, mask=mask, **kw)


# =============================================================================
# BiLSTM + Bahdanau Attention
# =============================================================================


class BiLSTMAttClassifier(nn.Module):
    """
    Bidirectional LSTM + Bahdanau attention (additive).
    e_i = v^T tanh(W_h h_i + b); alpha = softmax(e); c = sum alpha_i h_i
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        d = hidden_dim * 2
        self.W_h = nn.Linear(d, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, num_classes)
        init_lstm(self.lstm)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        h, _ = self.lstm(x)  # (B,T,D)
        scores = self.v(torch.tanh(self.W_h(h))).squeeze(-1)  # (B,T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (h * alpha).sum(dim=1)
        return self.fc(self.drop(ctx))


# =============================================================================
# TPA-LSTM (Shih et al. 2019)
# =============================================================================


class TPALSTMClassifier(nn.Module):
    """
    Temporal Pattern Attention LSTM (Shih et al., Machine Learning 2019).

    Given LSTM hidden matrix H ∈ R^{B×T×m}:
      1) Treat each of m hidden dims as a length-T series; apply k Conv1d filters
         of width w → H_C ∈ R^{B×m×k}
      2) Score row i vs current state h_T: f(H_C_i, h_T) = H_C_i^T W_a h_T
      3) α = softmax(f); V = Σ α_i H_C_i
      4) h' = W_h [V; h_T]  → classifier
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_filters: int = 32,
        filter_width: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # paper uses (uni) RNN/LSTM
        )
        # Conv over time for each hidden dimension: in_channels=1, process m channels via group or loop
        # Efficient: treat H as (B*m, 1, T) then conv
        self.cnn = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=filter_width,
            padding=filter_width - 1,  # causal-ish pad left later by cropping
        )
        self.W_a = nn.Parameter(torch.randn(n_filters, hidden_dim) * 0.02)
        self.W_h = nn.Linear(n_filters + hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        init_lstm(self.lstm)
        nn.init.xavier_uniform_(self.cnn.weight)

    def _temporal_pattern_attn(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: (B, T, m) → h_prime (B, m_out) using paper TPA on all steps,
        attending with last hidden state h_T.
        """
        b, t, m = H.shape
        h_t = H[:, -1, :]  # (B, m)

        # For each hidden dim row: conv over time → (B, m, k)
        # H_perm: (B, m, T) → view (B*m, 1, T)
        x = H.transpose(1, 2).contiguous().view(b * m, 1, t)
        hc = self.cnn(x)  # (B*m, k, T')
        # use last valid conv position (paper: filter slides over length w window of history)
        hc = hc[:, :, -1]  # (B*m, k)
        hc = hc.view(b, m, self.n_filters)  # H_C

        # scores: (H_C_i)^T W_a h_t  → (B, m)
        # H_C: (B,m,k), W_a: (k,m), h_t: (B,m)
        # f_i = H_C[:,i,:] @ W_a @ h_t
        wh = torch.matmul(h_t, self.W_a.t())  # (B, k)  = W_a h_t if W_a is (k,m)... 
        # Paper: (H^C_i)^T W_a h_t with W_a ∈ R^{k×m}
        # score_i = sum_k H_C[b,i,k] * (W_a h_t)[b,k]
        scores = (hc * wh.unsqueeze(1)).sum(dim=-1)  # (B, m)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,m,1)
        v = (alpha * hc).sum(dim=1)  # (B, k)
        h_prime = torch.tanh(self.W_h(torch.cat([v, h_t], dim=-1)))
        return h_prime

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        del mask
        H, _ = self.lstm(x)  # (B,T,m)
        h_prime = self._temporal_pattern_attn(H)
        return self.fc(self.drop(h_prime))


# =============================================================================
# ME-LSTM (mixed-effect)
# =============================================================================


class MELSTMClassifier(nn.Module):
    """
    Mixed-effect LSTM:
      fixed effect  = LSTM(x) pooled
      random effect = patient embedding (intercept in logit space)
      logits = W h + U z_patient + b
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_patients: int = 0,
        patient_embedding_dim: int = 16,
        pool: str = "last",
    ):
        super().__init__()
        self.pool = pool
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(d, num_classes)
        self.use_re = num_patients > 0
        if self.use_re:
            self.patient_emb = nn.Embedding(num_patients, patient_embedding_dim)
            self.re_to_logit = nn.Linear(patient_embedding_dim, num_classes, bias=False)
            nn.init.normal_(self.patient_emb.weight, std=0.01)
            nn.init.zeros_(self.re_to_logit.weight)
        self.drop = nn.Dropout(dropout)
        init_lstm(self.lstm)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patient_ids: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        o, _ = self.lstm(x)
        h = pool_sequence(o, mask, self.pool)
        logits = self.fc(self.drop(h))
        if self.use_re and patient_ids is not None:
            logits = logits + self.re_to_logit(self.patient_emb(patient_ids))
        return logits


# =============================================================================
# TFT building blocks (Lim et al. 2021) + classifier head
# =============================================================================


class GRN(nn.Module):
    """Gated Residual Network (TFT paper)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        # GLU gate
        g = torch.sigmoid(self.gate(h))
        y = g * self.fc_out(h)
        return self.norm(self.skip(x) + y)


class VariableSelectionNetwork(nn.Module):
    """Softmax variable weights via GRN (TFT)."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.single = nn.ModuleList([GRN(1, d_model, d_model, dropout) for _ in range(n_vars)])
        self.weight_grn = GRN(n_vars * d_model, d_model, n_vars, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        b, t, f = x.shape
        assert f == self.n_vars
        processed = []
        for i in range(f):
            # (B,T,1) → (B,T,D)
            processed.append(self.single[i](x[:, :, i : i + 1]))
        stacked = torch.stack(processed, dim=-2)  # (B,T,F,D)
        flat = stacked.reshape(b, t, f * self.d_model)
        w = torch.softmax(self.weight_grn(flat), dim=-1).unsqueeze(-1)  # (B,T,F,1)
        return (stacked * w).sum(dim=-2)  # (B,T,D)


class InterpretableMultiHeadAttention(nn.Module):
    """TFT interpretable MHA: shared value, additive attention across heads."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)  # shared value across heads
        self.W_o = nn.Linear(self.d_k, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, t, _ = q.shape
        Q = self.W_q(q).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v)  # (B,T,d_k) shared
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        # average heads then attend to shared V
        attn_mean = attn.mean(dim=1)  # (B,T,T)
        ctx = torch.matmul(attn_mean, V)  # (B,T,d_k)
        return self.W_o(ctx)


class TemporalFusionTransformerClassifier(ForecastAuxMixin, nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., Int. J. Forecasting 2021)
    adapted to sequence classification with next-step forecast auxiliary loss:
      VSN → LSTM encoder → interpretable MHA → GRN → (clf ‖ next-step forecast).
    Static covariates omitted when unavailable; observed inputs only.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        hidden_dim: int = 64,
        n_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        aux_forecast_weight: float = 0.1,
    ):
        super().__init__()
        self.aux_forecast_weight = float(aux_forecast_weight)
        self.vsn = VariableSelectionNetwork(input_dim, d_model, dropout)
        self.lstm = nn.LSTM(
            d_model, hidden_dim, lstm_layers, batch_first=True, bidirectional=False
        )
        self.post_lstm = GRN(hidden_dim, d_model, d_model, dropout)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attn = GRN(d_model, d_model, d_model, dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.forecast_head = nn.Linear(d_model, input_dim)
        self.drop = nn.Dropout(dropout)
        init_lstm(self.lstm)

    def _encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.vsn(x)
        o, _ = self.lstm(z)
        o = self.post_lstm(o)
        o = o + self.attn(o, o, o, mask=mask)
        return self.post_attn(o)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        o = self._encode(x, mask)
        h = pool_sequence(o, mask, "last")
        return self.fc(self.drop(h))

    def _forecast_aux(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.size(1) < 2:
            return x.new_zeros(())
        o = self._encode(x, mask)
        pred = self.forecast_head(o[:, :-1])
        tgt = x[:, 1:]
        err = (pred - tgt).pow(2).mean(dim=-1)
        if mask is not None:
            m = mask[:, 1:].to(err.dtype)
            return (err * m).sum() / m.sum().clamp(min=1.0)
        return err.mean()


# =============================================================================
# DeepAR (Salinas et al.)
# =============================================================================


class DeepARClassifier(ForecastAuxMixin, nn.Module):
    """
    DeepAR (Salinas et al., IJF 2020): autoregressive LSTM emitting Gaussian (μ,σ).
    Task adaptation = Gaussian next-step NLL auxiliary + classification head on h_T.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        aux_forecast_weight: float = 0.1,
    ):
        super().__init__()
        self.aux_forecast_weight = float(aux_forecast_weight)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu = nn.Linear(hidden_dim, input_dim)
        self.sigma = nn.Linear(hidden_dim, input_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        init_lstm(self.lstm)

    def forward_dist(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + 1e-4
        return mu, sigma, h

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        _, _, h = self.forward_dist(x)
        return self.fc(self.drop(pool_sequence(h, mask, "last")))

    def _forecast_aux(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.size(1) < 2:
            return x.new_zeros(())
        mu, sigma, _ = self.forward_dist(x)
        # predict x_{t+1} from state at t
        mu_t, sig_t, tgt = mu[:, :-1], sigma[:, :-1], x[:, 1:]
        nll = 0.5 * (
            torch.log(2 * math.pi * sig_t.pow(2)) + ((tgt - mu_t) / sig_t).pow(2)
        )
        nll = nll.mean(dim=-1)
        if mask is not None:
            m = mask[:, 1:].to(nll.dtype)
            return (nll * m).sum() / m.sum().clamp(min=1.0)
        return nll.mean()


# =============================================================================
# Registry
# =============================================================================

TORCH_ML_MODELS = {
    "LSTM": LSTMClassifier,
    "StackedLSTM": StackedLSTMClassifier,
    "BiLSTM-Att": BiLSTMAttClassifier,
    "TPA-LSTM": TPALSTMClassifier,
    "ME-LSTM": MELSTMClassifier,
    "TemporalFusionTrans": TemporalFusionTransformerClassifier,
    "Temporal Fusion Trans.": TemporalFusionTransformerClassifier,
    "DeepAR": DeepARClassifier,
}


def build_model(name: str, input_dim: int | None = None, num_classes: int = 5, **kwargs: Any):
    key = name.strip()
    if key in SKLEARN_BUILDERS:
        return SKLEARN_BUILDERS[key](num_classes=num_classes, **kwargs)
    if key in TORCH_ML_MODELS:
        if input_dim is None:
            raise ValueError(f"{key} requires input_dim")
        return TORCH_ML_MODELS[key](input_dim=input_dim, num_classes=num_classes, **kwargs)
    raise KeyError(f"Unknown ML model: {name!r}. Choose from {list_models()}")


def list_models() -> list[str]:
    return [
        "LR", "SVM", "DT", "LightGBM",
        "LSTM", "StackedLSTM", "BiLSTM-Att", "TPA-LSTM", "ME-LSTM",
        "Temporal Fusion Trans.", "DeepAR",
    ]


def flatten_seq(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(f"expected (N,T,F) or (N,F), got {x.shape}")
    return x.reshape(x.shape[0], -1)
