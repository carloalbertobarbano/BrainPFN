"""
The BrainPFN network: an asymmetric-attention Transformer (TabPFN-style).

Context tokens (x_i, y_i) attend to each other; query tokens (x_j,) attend to the
context only; a scalar head regresses the score. The network standardises its
inputs and target per context, so it predicts on the scale of the context labels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _AsymmetricSelfAttention(nn.Module):
    """Context tokens attend to each other; query tokens attend to context tokens only."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        train_tok: torch.Tensor,   # (B, n_train, d)
        test_tok:  torch.Tensor,   # (B, n_test,  d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n_tr, d = train_tok.shape
        _, n_te, _ = test_tok.shape
        H, Dh = self.n_heads, self.d_head

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # (B, S, d) → (B, H, S, Dh)
            return x.view(x.size(0), x.size(1), H, Dh).transpose(1, 2)

        # --- context self-attention ---
        Qtr = split_heads(self.q(train_tok))
        Ktr = split_heads(self.k(train_tok))
        Vtr = split_heads(self.v(train_tok))
        att_tr = torch.softmax(Qtr @ Ktr.transpose(-1, -2) / self.scale, dim=-1)
        att_tr = self.drop(att_tr)
        out_tr = (att_tr @ Vtr).transpose(1, 2).reshape(B, n_tr, d)
        out_tr = self.out(out_tr)

        # --- query cross-attention (query → context keys/values) ---
        Qte = split_heads(self.q(test_tok))
        att_te = torch.softmax(Qte @ Ktr.transpose(-1, -2) / self.scale, dim=-1)
        att_te = self.drop(att_te)
        out_te = (att_te @ Vtr).transpose(1, 2).reshape(B, n_te, d)
        out_te = self.out(out_te)

        return out_tr, out_te


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = _AsymmetricSelfAttention(d_model, n_heads, dropout)
        self.ff_tr = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ff_te = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln1_tr = nn.LayerNorm(d_model)
        self.ln1_te = nn.LayerNorm(d_model)
        self.ln2_tr = nn.LayerNorm(d_model)
        self.ln2_te = nn.LayerNorm(d_model)

    def forward(
        self,
        train_tok: torch.Tensor,
        test_tok:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # attention + residual
        a_tr, a_te = self.attn(train_tok, test_tok)
        train_tok = self.ln1_tr(train_tok + a_tr)
        test_tok  = self.ln1_te(test_tok  + a_te)
        # feedforward + residual
        train_tok = self.ln2_tr(train_tok + self.ff_tr(train_tok))
        test_tok  = self.ln2_te(test_tok  + self.ff_te(test_tok))
        return train_tok, test_tok


class PFNTransformer(nn.Module):
    """
    Asymmetric Transformer for FC → score regression.

    Input token shapes:
      context : (B, n_context, d_feat + 1)   — x concatenated with the scalar y
      query   : (B, n_query,   d_feat)       — x only

    Output:
      predictions : (B, n_query)             — scalar score per query subject
    """

    def __init__(self, d_feat: int, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_feat = int(d_feat)

        # context token = concat(x_norm, y_norm) → d_model ; query token = x_norm → d_model
        self.train_enc = nn.Linear(d_feat + 1, d_model)
        self.test_enc  = nn.Linear(d_feat,     d_model)

        self.layers = nn.ModuleList([
            _TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x_train: torch.Tensor,   # (B, n_train, d_feat)
        y_train: torch.Tensor,   # (B, n_train)
        x_test:  torch.Tensor,   # (B, n_test,  d_feat)
    ) -> torch.Tensor:           # (B, n_test)

        # normalise x per feature across context subjects for each batch item
        mu  = x_train.mean(dim=1, keepdim=True)   # (B, 1, d)
        std = x_train.std(dim=1, keepdim=True).clamp(min=1e-6)   # (B, 1, d)
        x_tr_n = (x_train - mu) / std
        x_te_n = (x_test  - mu) / std             # same stats from the context

        # normalise y (per batch item)
        y_mu  = y_train.mean(dim=1, keepdim=True)   # (B, 1)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-6)
        y_n   = (y_train - y_mu) / y_std            # (B, n_train)

        # encode
        train_tok = self.train_enc(
            torch.cat([x_tr_n, y_n.unsqueeze(-1)], dim=-1)
        )   # (B, n_train, d_model)
        test_tok  = self.test_enc(x_te_n)            # (B, n_test,  d_model)

        for layer in self.layers:
            train_tok, test_tok = layer(train_tok, test_tok)

        # scalar output + de-normalise
        y_hat_n = self.head(test_tok).squeeze(-1)    # (B, n_test)
        return y_hat_n * y_std + y_mu                # back to the context's scale
