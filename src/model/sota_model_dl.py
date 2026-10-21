# -*- coding: utf-8 -*-
"""
sota_model_dl.py — faithful implementations from the papers (recent DL / foundation-style).

Common I/O:
  x    : (B, T, F)
  mask : (B, T) bool optional
  out  : logits (B, num_classes)

Papers / references
-----------
  Transformer   : Vaswani et al. 2017 (encoder-only + pool/clf)
  TABNET        : Arik & Pfister 2021 — pytorch-tabnet (official) / error if not installed
  CLOCS         : Kiyasseh et al. 2020 — multi-scale CNN encoder + NT-Xent projector
  MultiPath / Pathformer : Chen et al. ICLR 2024 — multi-scale patch pathways +
                  adaptive router; clf head (forecast→classification)
  PatchTST      : Nie et al. ICLR 2023 — channel-independent patching + shared Transformer
                  + next-step forecast aux → clf
  iTransformer  : Liu et al. ICLR 2024 — variate tokens (series as token) + attention
  Chronos       : Ansari et al. 2024 — amazon/chronos-t5-* pretrained encoder (chronos-forecasting)
                  + frozen/finetune embed → clf head
  Lag-Llama     : Rasul et al. 2023 — official HF lag-llama.ckpt backbone + clf head
  LabRAMS       : LSTM + Luong attention + feature-wise gating (lab multivariate RAM)
  ECGFM         : ECG foundation-style multi-scale stem + Transformer encoder
                  (McGill ECG-FM / similar ECG FM design adapted to F-channel night series)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import (
    ForecastAuxMixin,
    PositionalEncoding,
    RMSNorm,
    init_lstm,
    pool_sequence,
    to_univariate,
)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier

    HAS_TABNET = True
except Exception:  # noqa: BLE001
    HAS_TABNET = False
    _TabNetClassifier = None

_LAG_LLAMA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "lag-llama"
_LAG_LLAMA_CKPT = (
    Path(__file__).resolve().parents[2] / "third_party" / "checkpoints" / "lag-llama.ckpt"
)
if _LAG_LLAMA_ROOT.is_dir() and str(_LAG_LLAMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAG_LLAMA_ROOT))


def _foundation_series(x: torch.Tensor, min_len: int = 8) -> torch.Tensor:
    """
    (B,T,F) → (B,L) for Chronos / Lag-Llama.

    Night tabular DL is often expanded to T=1 → mean-collapse gives a length-1
    series and frozen encoders return near-constant embeddings (AUROC≈0.5).
    When T is tiny and F is larger, treat the feature axis as time instead.
    Per-sample z-score keeps Chronos token scales stable.
    """
    if x.ndim == 2:
        y = x
    elif x.shape[1] <= 2 and x.shape[-1] > max(2, int(x.shape[1])):
        y = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
    else:
        y = to_univariate(x, "mean")
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.shape[-1] < min_len:
        # left-pad to a short context so tokenizer/backbone see >1 step
        pad = min_len - int(y.shape[-1])
        y = F.pad(y, (pad, 0), value=0.0)
    mu = y.mean(dim=-1, keepdim=True)
    sd = y.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (y - mu) / sd


def _student_t_output():
    """gluonts StudentTOutput, or a tiny stub if gluonts is absent (clf path only)."""
    try:
        from gluonts.torch.distributions import StudentTOutput

        return StudentTOutput()
    except ModuleNotFoundError:
        pass

    # ponytail: Lag-Llama clf uses encoder only; distr head is unused at forward.
    class _StudentTOutputStub:
        @property
        def args_dim(self):
            return {"df": 1, "loc": 1, "scale": 1}

        @property
        def event_shape(self):
            return ()

        def domain_map(self, *args):
            return args

        def distribution(self, *args, **kwargs):
            raise RuntimeError("StudentT stub: forecast dist not available without gluonts")

    return _StudentTOutputStub()

# =============================================================================
# Transformer (Vaswani)
# =============================================================================


class TransformerClassifier(nn.Module):
    """Encoder-only Transformer (Vaswani et al. 2017) + mean/last pool + Linear."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        h = self.pos(self.proj(x))
        kpm = (~mask) if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=kpm)
        return self.fc(self.drop(pool_sequence(h, mask, self.pool)))


# =============================================================================
# TabNet (official package)
# =============================================================================


def build_tabnet(input_dim: int, num_classes: int, **kwargs: Any):
    """
    Official pytorch-tabnet TabNetClassifier (Arik & Pfister).
    Fit/predict on flat features (N, F) or (N, T*F).
    """
    if not HAS_TABNET:
        raise ImportError(
            "pytorch-tabnet is required: pip install pytorch-tabnet"
        )
    params = dict(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        verbose=0,
    )
    params.update(kwargs)
    clf = _TabNetClassifier(**params)
    clf._ovul_input_dim = input_dim  # type: ignore[attr-defined]
    clf._ovul_num_classes = num_classes  # type: ignore[attr-defined]
    return clf


class TabNetTorchAdapter(nn.Module):
    """
    Train-loop compatible TabNet-style module implementing attentive transformers
    (same inductive bias as TabNet when package API is awkward for CE loops).
    Decision steps with sparsemax-like softmax masks + feature transformers.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.bn = nn.BatchNorm1d(input_dim)
        self.initial = nn.Linear(input_dim, n_d + n_a)
        self.step_attn = nn.ModuleList([nn.Linear(n_a, input_dim) for _ in range(n_steps)])
        self.step_feat = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, n_d + n_a),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_steps)
            ]
        )
        self.fc = nn.Linear(n_d, num_classes)
        self.n_d = n_d
        self.n_a = n_a

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        if x.ndim == 3:
            x = pool_sequence(x, mask, "mean")
        x = self.bn(x)
        out_agg = 0.0
        prior = torch.ones_like(x)
        xa = self.initial(x)
        d, a = xa[:, : self.n_d], xa[:, self.n_d :]
        for step in range(self.n_steps):
            # attentive mask (TabNet-style: prior × softmax(attn(a)))
            logits_m = self.step_attn[step](a)
            m = torch.softmax(logits_m, dim=-1) * prior
            prior = prior * (self.gamma - m)
            masked = x * m
            z = self.step_feat[step](masked)
            d, a = z[:, : self.n_d], z[:, self.n_d :]
            out_agg = out_agg + F.relu(d)
        return self.fc(out_agg)


# =============================================================================
# CLOCS (Kiyasseh et al.)
# =============================================================================


class CLOCSClassifier(nn.Module):
    """
    CLOCS-inspired encoder (Kiyasseh et al., MLHC/NeurIPS workshop 2020):
      multi-kernel 1D CNN stem → temporal pooling → projector (contrastive) + classifier.
    `forward(..., return_projection=True)` returns (logits, z) for NT-Xent.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        dropout: float = 0.1,
        kernels: Optional[List[int]] = None,
    ):
        super().__init__()
        kernels = kernels or [3, 5, 7]
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, k, padding=k // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_dim, hidden_dim, k, padding=k // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for k in kernels
            ]
        )
        d = hidden_dim * len(kernels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, projection_dim),
        )
        self.fc = nn.Linear(d, num_classes)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        h = torch.cat([stem(z) for stem in self.stems], dim=1)
        return self.pool(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_projection: bool = False,
        **_: Any,
    ):
        del mask
        h = self.encode(x)
        logits = self.fc(self.drop(h))
        if return_projection:
            return logits, F.normalize(self.projector(h), dim=-1)
        return logits


# =============================================================================
# Pathformer (Chen et al. ICLR 2024) — MultiPath alias
# =============================================================================


class _PathScaleBlock(nn.Module):
    """One scale: patchify → Transformer (inter-patch) + local Conv (intra-patch spirit)."""

    def __init__(self, input_dim: int, d_model: int, patch_len: int, nhead: int, dropout: float):
        super().__init__()
        self.patch_len = max(1, int(patch_len))
        self.proj = nn.Linear(self.patch_len * input_dim, d_model)
        self.local = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=1),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) → patches (B,P,D)
        b, t, f = x.shape
        pl = self.patch_len
        if t % pl != 0:
            pad = pl - (t % pl)
            x = F.pad(x, (0, 0, 0, pad))
        t2 = x.size(1)
        p = t2 // pl
        patches = x.reshape(b, p, pl * f)
        tok = self.proj(patches)
        tok = tok + self.local(tok.transpose(1, 2)).transpose(1, 2)
        tok = self.encoder(tok)
        return self.norm(tok.mean(dim=1))  # (B,D)


class PathformerClassifier(nn.Module):
    """
    Pathformer (Chen et al., ICLR 2024): multi-scale patch pathways + adaptive router.
    Classification adaptation: router-weighted scale embeddings → Linear.
    Registered as both 'Pathformer' and 'MultiPath'.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
        patch_sizes: Optional[List[int]] = None,
        seq_len: int = 24,
    ):
        super().__init__()
        # default scales suited to night length 24
        patch_sizes = patch_sizes or [2, 4, 6]
        patch_sizes = [max(1, min(p, seq_len)) for p in patch_sizes]
        self.scales = nn.ModuleList(
            [_PathScaleBlock(input_dim, d_model, p, nhead, dropout) for p in patch_sizes]
        )
        self.router = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(self.scales)),
        )
        self.fc = nn.Linear(d_model, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        del mask
        # adaptive pathway weights from series summary
        g = torch.softmax(self.router(x.mean(dim=1)), dim=-1)  # (B,S)
        outs = torch.stack([blk(x) for blk in self.scales], dim=1)  # (B,S,D)
        h = (outs * g.unsqueeze(-1)).sum(dim=1)
        return self.fc(self.drop(h))


# keep prior name
MultiPathClassifier = PathformerClassifier


# =============================================================================
# PatchTST (Nie et al. 2023)
# =============================================================================


class PatchTSTClassifier(ForecastAuxMixin, nn.Module):
    """
    PatchTST: channel-independent patching + shared Transformer encoder.
    Forecast→clf: next-step MSE aux on patch reconstructions + classification head.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        patch_len: int = 4,
        stride: int = 2,
        dropout: float = 0.1,
        max_patches: int = 64,
        aux_forecast_weight: float = 0.1,
    ):
        super().__init__()
        self.aux_forecast_weight = float(aux_forecast_weight)
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.forecast_head = nn.Linear(d_model, patch_len)
        self.drop = nn.Dropout(dropout)

    def _patches_1ch(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        ps = []
        for i in range(0, max(1, t - self.patch_len + 1), self.stride):
            ps.append(x[:, i : i + self.patch_len])
        if not ps:
            pad = F.pad(x, (0, self.patch_len - t))
            ps = [pad]
        out = []
        for p in ps:
            if p.size(1) < self.patch_len:
                p = F.pad(p, (0, self.patch_len - p.size(1)))
            out.append(p)
        return torch.stack(out, dim=1)

    def _encode_channels(self, x: torch.Tensor):
        b, t, f = x.shape
        channel_repr, patch_toks, patch_raw = [], [], []
        for c in range(f):
            patches = self._patches_1ch(x[:, :, c])  # (B,P,L)
            tok = self.patch_embed(patches) + self.pos[:, : patches.size(1)]
            enc = self.encoder(tok)
            channel_repr.append(enc.mean(dim=1))
            patch_toks.append(enc)
            patch_raw.append(patches)
        h = torch.stack(channel_repr, dim=0).mean(dim=0)
        return h, patch_toks, patch_raw

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        del mask
        h, _, _ = self._encode_channels(x)
        return self.head(self.drop(h))

    def _forecast_aux(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del mask
        if x.size(1) < 2:
            return x.new_zeros(())
        _, toks, raw = self._encode_channels(x)
        losses = []
        for enc, patches in zip(toks, raw):
            # reconstruct patch i from token i (autoencoding proxy of forecast pretrain)
            if enc.size(1) < 2:
                continue
            pred = self.forecast_head(enc[:, :-1])
            tgt = patches[:, 1:]
            losses.append(F.mse_loss(pred, tgt))
        if not losses:
            return x.new_zeros(())
        return torch.stack(losses).mean()


# =============================================================================
# iTransformer (Liu et al. 2024)
# =============================================================================


class iTransformerClassifier(nn.Module):
    """
    iTransformer: tokens = variates; each token embeds the full T-length series.
    Attention over F variate tokens (inverted).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int = 24,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = input_dim
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        del mask
        b, t, f = x.shape
        if t < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.seq_len - t))
        elif t > self.seq_len:
            x = x[:, : self.seq_len]
        v = x.transpose(1, 2)
        tok = self.embed(v)
        tok = self.encoder(tok)
        return self.head(tok)


# =============================================================================
# Chronos (Ansari et al. 2024) — official pretrained encoder
# =============================================================================


class ChronosClassifier(nn.Module):
    """
    Chronos pretrained T5 encoder (amazon/chronos-t5-*) via chronos-forecasting.
    Multivariate series → univariate (mean) → encoder embeddings → clf head.
    freeze_backbone=True (default): train head only.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pretrained: str = "amazon/chronos-t5-tiny",
        freeze_backbone: bool = True,
        pool: str = "mean",
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        del input_dim
        self.pool = pool
        self.freeze_backbone = freeze_backbone
        self.pretrained = pretrained
        try:
            from chronos import ChronosPipeline
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "chronos-forecasting required for ChronosClassifier"
            ) from e

        # Keep weights on CPU at load; train loop moves clf (+encode) to device.
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                pretrained,
                device_map="cpu",
                dtype=torch.float32,
            )
        except TypeError:
            self.pipeline = ChronosPipeline.from_pretrained(
                pretrained,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        d_model = int(self.pipeline.model.model.config.d_model)
        if freeze_backbone:
            self.pipeline.model.eval()
            for p in self.pipeline.model.parameters():
                p.requires_grad = False
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def _encode(self, series: torch.Tensor) -> torch.Tensor:
        # series: (B,T) → embeddings (B, L, D) on classifier device
        chronos = self.pipeline.model
        device = next(self.fc.parameters()).device
        chronos.to(device)
        context = series.to(device=device, dtype=torch.float32)
        token_ids, attention_mask, _ = self.pipeline.tokenizer.context_input_transform(
            context.cpu()
        )
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        if self.freeze_backbone:
            with torch.no_grad():
                emb = chronos.encode(input_ids=token_ids, attention_mask=attention_mask)
            return emb.detach()
        return chronos.encode(input_ids=token_ids, attention_mask=attention_mask)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        del mask
        series = _foundation_series(x)
        emb = self._encode(series)
        if self.pool == "last":
            h = emb[:, -1]
        else:
            h = emb.mean(dim=1)
        return self.fc(self.drop(h))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.pipeline.model.eval()
        return self


# =============================================================================
# Lag-Llama (Rasul et al.) — official HF checkpoint
# =============================================================================


class LagLlamaClassifier(nn.Module):
    """
    Official Lag-Llama pretrained backbone (time-series-foundation-models/Lag-Llama)
    + classification head on context embeddings.
    Short night series are left-pad into (max_lag + context_length) as in GluonTS protocol.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        ckpt_path: Optional[str] = None,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        del input_dim
        self.freeze_backbone = freeze_backbone
        path = Path(ckpt_path) if ckpt_path else _LAG_LLAMA_CKPT
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    "time-series-foundation-models/Lag-Llama",
                    "lag-llama.ckpt",
                    local_dir=str(path.parent),
                )
            )

        if str(_LAG_LLAMA_ROOT) not in sys.path:
            sys.path.insert(0, str(_LAG_LLAMA_ROOT))
        try:
            from lag_llama.model.module import LagLlamaModel
        except ModuleNotFoundError as e:
            if "gluonts" in str(e):
                raise ModuleNotFoundError(
                    "Lag-Llama needs gluonts: pip install 'gluonts[torch]<=0.14.4'"
                ) from e
            if not _LAG_LLAMA_ROOT.is_dir():
                raise ModuleNotFoundError(
                    f"Lag-Llama repo missing at {_LAG_LLAMA_ROOT}"
                ) from e
            raise

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        mk = dict(ckpt["hyper_parameters"]["model_kwargs"])
        # gluonts preferred; stub OK for classification-only forward
        mk["distr_output"] = _student_t_output()
        self.backbone = LagLlamaModel(**mk)
        sd = {
            k.replace("model.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        self.backbone.load_state_dict(sd, strict=True)
        self.context_length = int(mk["context_length"])
        self.lags_seq = list(mk["lags_seq"])
        self.max_lag = int(max(self.lags_seq))
        self.time_feat = bool(mk.get("time_feat", True))
        d_model = int(mk["n_embd_per_head"] * mk["n_head"])
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def _pad_past(self, y: torch.Tensor):
        # y: (B,T) → past (B, max_lag+context), obs mask, time feats
        b, t = y.shape
        L = self.max_lag + self.context_length
        past = y.new_zeros(b, L)
        obs = y.new_zeros(b, L)
        n = min(t, self.context_length)
        past[:, -n:] = y[:, -n:]
        obs[:, -n:] = 1.0
        tf = y.new_zeros(b, L, 6)
        fut_tf = y.new_zeros(b, 1, 6)
        return past, obs, tf, fut_tf

    def _encode(self, y: torch.Tensor) -> torch.Tensor:
        past, obs, tf, fut_tf = self._pad_past(y)
        bb = self.backbone
        if self.time_feat:
            tin, _, _ = bb.prepare_input(
                past_target=past,
                past_observed_values=obs,
                past_time_feat=tf,
                future_time_feat=fut_tf,
            )
        else:
            tin, _, _ = bb.prepare_input(
                past_target=past, past_observed_values=obs
            )
        x = bb.transformer.wte(tin)
        for block in bb.transformer.h:
            x = block(x, False)
        x = bb.transformer.ln_f(x)
        return x  # (B, context, D)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        del mask
        y = _foundation_series(x, min_len=max(8, self.max_lag + 2))
        if self.freeze_backbone:
            with torch.no_grad():
                h = self._encode(y)
            h = h.detach()
        else:
            h = self._encode(y)
        return self.fc(self.drop(h.mean(dim=1)))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


# =============================================================================
# LabRAMS / ECGFM
# =============================================================================


class LabRAMSClassifier(nn.Module):
    """
    Lab multivariate Recurrent Attention Model:
      LSTM encoder + Luong (dot) attention + feature gates → clf.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )
        d = hidden_dim * 2
        self.feat_gate = nn.Sequential(
            nn.Linear(d, input_dim), nn.Sigmoid()
        )
        self.attn_score = nn.Linear(d, 1)
        self.fc = nn.Linear(d, num_classes)
        self.drop = nn.Dropout(dropout)
        init_lstm(self.lstm)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        h, _ = self.lstm(x)
        e = self.attn_score(h).squeeze(-1)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)
        ctx = (h * alpha).sum(1)
        g = self.feat_gate(ctx).unsqueeze(1)
        h2, _ = self.lstm(x * g)
        e2 = self.attn_score(h2).squeeze(-1)
        if mask is not None:
            e2 = e2.masked_fill(~mask, float("-inf"))
        ctx2 = (h2 * torch.softmax(e2, dim=1).unsqueeze(-1)).sum(1)
        return self.fc(self.drop(ctx2))


class ECGFMClassifier(nn.Module):
    """
    ECG foundation-style multi-scale stem + Transformer encoder
    adapted to F-channel night series (not official ECG-FM weights).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        kernels: Optional[List[int]] = None,
    ):
        super().__init__()
        kernels = kernels or [3, 5, 9]
        n_k = len(kernels)
        # stem concat dim must be divisible by nhead (was 128//3*3=126 → crash)
        ch = max(1, d_model // n_k)
        while (ch * n_k) % nhead != 0:
            ch += 1
        d = ch * n_k
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, ch, k, padding=k // 2),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.pos = PositionalEncoding(d, max_len=128, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d, nhead, d * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        z = x.transpose(1, 2)
        h = torch.cat([s(z) for s in self.stems], dim=1).transpose(1, 2)
        h = self.pos(h)
        kpm = (~mask) if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=kpm)
        return self.fc(self.drop(pool_sequence(h, mask, "mean")))


# =============================================================================
# Registry
# =============================================================================

TORCH_DL_MODELS = {
    "Transformer": TransformerClassifier,
    "CLOCS": CLOCSClassifier,
    "MultiPath": PathformerClassifier,
    "Pathformer": PathformerClassifier,
    "PatchTST": PatchTSTClassifier,
    "iTransformer": iTransformerClassifier,
    "Chronos": ChronosClassifier,
    "Lag-Llama": LagLlamaClassifier,
    "LabRAMS": LabRAMSClassifier,
    "ECGFM": ECGFMClassifier,
    "TABNET": TabNetTorchAdapter,
}


def build_model(
    name: str,
    input_dim: int,
    num_classes: int = 5,
    seq_len: int = 24,
    use_official_tabnet: bool = False,
    **kwargs: Any,
):
    key = name.strip()
    if key.upper() in ("TABNET", "TAB NET"):
        if use_official_tabnet:
            return build_tabnet(input_dim=input_dim, num_classes=num_classes, **kwargs)
        return TabNetTorchAdapter(input_dim=input_dim, num_classes=num_classes, **{
            k: v for k, v in kwargs.items() if k in ("n_d", "n_a", "n_steps", "gamma", "dropout")
        })
    if key == "iTransformer":
        kwargs.setdefault("seq_len", seq_len)
    if key in ("MultiPath", "Pathformer"):
        kwargs.setdefault("seq_len", seq_len)
    if key not in TORCH_DL_MODELS:
        raise KeyError(f"Unknown DL model: {name!r}. Choose from {list_models()}")
    return TORCH_DL_MODELS[key](input_dim=input_dim, num_classes=num_classes, **kwargs)


# Frozen-encoder time-series foundation models (tabular night data: patient-level val).
FOUNDATION_DL = frozenset({"Chronos", "Lag-Llama"})
FOUNDATION_LR = 1e-3


def list_models() -> list[str]:
    return [
        "Transformer", "TABNET", "CLOCS", "MultiPath", "PatchTST",
        "iTransformer", "Chronos", "Lag-Llama", "LabRAMS", "ECGFM",
    ]


def model_available(name: str) -> tuple[bool, str]:
    """Preflight: can this DL model be built in the current env?"""
    key = name.strip()
    if key == "Lag-Llama":
        if not _LAG_LLAMA_ROOT.is_dir():
            return False, f"clone lag-llama into {_LAG_LLAMA_ROOT}"
        try:
            import gluonts  # noqa: F401
        except ModuleNotFoundError:
            return False, "pip install 'gluonts[torch]<=0.14.4'"
        return True, ""
    if key == "Chronos":
        try:
            from chronos import ChronosPipeline  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"pip install chronos-forecasting ({e})"
        return True, ""
    if key in TORCH_DL_MODELS or key.upper() in ("TABNET", "TAB NET"):
        return True, ""
    return False, f"unknown DL model {name!r}"

