# -*- coding: utf-8 -*-
"""Shared utilities — for classifying per-night (24-slot) sequences."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def pool_sequence(
    h: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = "last",
) -> torch.Tensor:
    """
    (B, T, H) → (B, H)

    mode:
      last  — last valid step (if mask is given)
      mean  — mask-aware mean
      max   — max over time
    """
    if mode == "mean":
        if mask is None:
            return h.mean(dim=1)
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    if mode == "max":
        if mask is None:
            return h.max(dim=1).values
        neg = torch.finfo(h.dtype).min
        m = mask.unsqueeze(-1).bool()
        return h.masked_fill(~m, neg).max(dim=1).values
    # last
    if mask is None:
        return h[:, -1, :]
    lengths = mask.long().sum(dim=1).clamp(min=1) - 1  # (B,)
    b = torch.arange(h.size(0), device=h.device)
    return h[b, lengths]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def init_lstm(lstm: nn.LSTM) -> None:
    for name, p in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias" in name:
            nn.init.zeros_(p.data)
            n = p.size(0)
            p.data[n // 4 : n // 2].fill_(1.0)


def to_univariate(x: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """(B, T, F) → (B, T) for univariate foundation models."""
    if x.ndim == 2:
        return x
    if how == "first":
        return x[..., 0]
    return x.mean(dim=-1)


class ForecastAuxMixin:
    """
    Optional next-step forecast auxiliary loss for forecast→classification adaptation.
    Models set `self.aux_forecast_weight` (>0 to enable) and implement `_forecast_aux(x, mask)`.
    """

    aux_forecast_weight: float = 0.1

    def aux_forecast_loss(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.aux_forecast_weight <= 0:
            return x.new_zeros(())
        return self.aux_forecast_weight * self._forecast_aux(x, mask)

    def _forecast_aux(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError
