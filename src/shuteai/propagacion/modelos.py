"""Modelos de propagación. Todos producen curvas NO decrecientes: predicen incrementos positivos
(softplus) y los acumulan, igual que el alcance real.

Modo A (solo texto -> curva de 96 pasos):  PropMLP (S1), PropagadorLSTM (S3-S5), PropagadorTransformer (S6)
Modo B (6 h observadas -> 42 h):           ForecasterLSTM (S3/S12), PatchTransformer (S6/S12, PatchTST)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PropMLP(nn.Module):
    """S1 - Embedding de la noticia -> 96 incrementos."""

    def __init__(self, d_txt=768, h=512, pasos=96, dropout=0.2):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(d_txt, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, pasos),
        )

    def forward(self, cls, tokens=None, mask=None, y_real=None, p_real=0.0):
        return torch.cumsum(F.softplus(self.red(cls)), dim=1)


class PropagadorLSTM(nn.Module):
    """S3/S4/S5 - Encoder-decoder: el embedding inicializa el LSTM y el decoder atiende a las palabras.

    Basado en el modelo del plan, con tres mejoras: máscara de padding en la atención, tiempo
    normalizado como entrada extra y muestreo programado (scheduled sampling) en lugar de
    teacher forcing puro, para reducir el sesgo de exposición.
    """

    def __init__(self, d_txt=768, h=256, pasos=96, heads=4, dropout=0.2):
        super().__init__()
        self.pasos = pasos
        self.h0, self.c0 = nn.Linear(d_txt, h), nn.Linear(d_txt, h)
        self.lstm = nn.LSTMCell(2, h)
        self.mem = nn.Linear(d_txt, h)
        self.att = nn.MultiheadAttention(h, num_heads=heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.sal = nn.Linear(2 * h, 1)
        self.ultima_atencion = None

    def forward(self, cls, tokens, mask, y_real=None, p_real=0.0):
        """mask: (B, T) con 1 en tokens reales. y_real y p_real activan el muestreo programado."""
        h, c = torch.tanh(self.h0(cls)), torch.tanh(self.c0(cls))
        mem = self.mem(tokens)
        pad = mask == 0
        acum = torch.zeros(cls.size(0), 1, device=cls.device)
        curva, atenciones = [], []
        for t in range(self.pasos):
            tiempo = torch.full_like(acum, t / self.pasos)
            h, c = self.lstm(torch.cat([acum, tiempo], dim=-1), (h, c))
            ctx, w = self.att(h.unsqueeze(1), mem, mem, key_padding_mask=pad, need_weights=True)
            atenciones.append(w)
            inc = F.softplus(self.sal(self.drop(torch.cat([h, ctx.squeeze(1)], -1))))
            acum = acum + inc
            curva.append(acum)
            if y_real is not None and p_real > 0:
                usar = (torch.rand(acum.size(0), 1, device=acum.device) < p_real).float()
                acum = usar * y_real[:, t : t + 1] + (1 - usar) * acum
        self.ultima_atencion = torch.cat(atenciones, dim=1).detach()  # (B, pasos, T)
        return torch.cat(curva, dim=1)


class PropagadorTransformer(nn.Module):
    """S6 - Decoder Transformer sobre los 96 pasos de tiempo con atención cruzada al texto."""

    def __init__(self, d_txt=768, h=256, pasos=96, heads=4, capas=2, dropout=0.2):
        super().__init__()
        self.pasos = pasos
        self.mem = nn.Linear(d_txt, h)
        self.cls_a_q = nn.Linear(d_txt, h)
        self.pos = nn.Parameter(torch.randn(1, pasos, h) * 0.02)
        capa = nn.TransformerDecoderLayer(
            h, heads, dim_feedforward=4 * h, dropout=dropout, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(capa, capas)
        self.sal = nn.Linear(h, 1)
        self.register_buffer("causal", torch.triu(torch.ones(pasos, pasos, dtype=torch.bool), 1))

    def forward(self, cls, tokens, mask, y_real=None, p_real=0.0):
        q = self.pos + self.cls_a_q(cls).unsqueeze(1)
        x = self.dec(q, self.mem(tokens), tgt_mask=self.causal, memory_key_padding_mask=(mask == 0))
        return torch.cumsum(F.softplus(self.sal(x)).squeeze(-1), dim=1)


class ForecasterLSTM(nn.Module):
    """Modo B (S3/S12) - LSTM sobre las 6 h observadas -> incrementos de las 42 h siguientes."""

    def __init__(self, contexto=12, horizonte=84, h=128, capas=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(2, h, num_layers=capas, batch_first=True, dropout=dropout)
        self.cabeza = nn.Sequential(
            nn.Linear(h, 2 * h), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * h, horizonte)
        )

    def forward(self, ctx):
        delta = torch.diff(ctx, dim=1, prepend=torch.zeros_like(ctx[:, :1]))
        _, (h, _) = self.lstm(torch.stack([ctx, delta], dim=-1))
        return ctx[:, -1:] + torch.cumsum(F.softplus(self.cabeza(h[-1])), dim=1)


class PatchTransformer(nn.Module):
    """Modo B (S6/S12) - Encoder Transformer sobre parches de la serie (PatchTST).

    patch_len=1 y stride=1 equivale a un Transformer "punto a punto"; patch_len=4, stride=2 es PatchTST.
    """

    def __init__(
        self, contexto=12, horizonte=84, patch_len=4, stride=2, h=128, capas=3, heads=4, dropout=0.2
    ):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n = (contexto - patch_len) // stride + 1
        self.proy = nn.Linear(patch_len, h)
        self.pos = nn.Parameter(torch.randn(1, n, h) * 0.02)
        capa = nn.TransformerEncoderLayer(
            h, heads, 4 * h, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(capa, capas)
        self.cabeza = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n * h, horizonte))

    def forward(self, ctx):
        ancla = ctx[:, -1:]
        parches = (ctx - ancla).unfold(1, self.patch_len, self.stride)  # (B, n, patch_len)
        z = self.enc(self.proy(parches) + self.pos)
        return ancla + torch.cumsum(F.softplus(self.cabeza(z)), dim=1)


MODELOS_MODO_A = {"mlp": PropMLP, "lstm": PropagadorLSTM, "transformer": PropagadorTransformer}
