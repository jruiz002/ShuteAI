"""Inferencia del clasificador Transformer: probabilidades calibradas, mapa de atención y embeddings.

Un solo paso hacia adelante entrega las tres cosas, y los embeddings alimentan al modelo de propagación.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from shuteai.clasificador.codificadores import CodificadorTransformer
from shuteai.clasificador.metricas import softmax
from shuteai.clasificador.modelos import ClasificadorTransformer
from shuteai.rutas import MODELOS
from shuteai.utils import dispositivo

DIR_FINAL = MODELOS / "clasificador" / "final"


@dataclass
class ResultadoClasificacion:
    p_verdadera: float
    p_falsa: float
    palabras: list[str]
    pesos_atencion: list[float]  # normalizados a [0, 1], uno por palabra
    cls: torch.Tensor  # (1, d)
    tokens: torch.Tensor  # (1, T, d)
    mascara: torch.Tensor  # (1, T)


def rollout_atencion(atenciones: tuple[torch.Tensor, ...], mascara: torch.Tensor) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema, 2020): propaga la atención capa a capa con conexiones
    residuales. Devuelve la importancia de cada token para [CLS], shape (T,). Batch de 1."""
    n = atenciones[0].size(-1)
    r = torch.eye(n, device=mascara.device)
    for a in atenciones:
        a = a[0].mean(dim=0)  # promedio de cabezas: (T, T)
        a = a + torch.eye(n, device=a.device)  # conexión residual
        a = a / a.sum(dim=-1, keepdim=True)
        r = a @ r
    return r[0]


class ClasificadorFinal:
    """Envuelve el modelo desplegado: models/clasificador/final/{pesos.pt, info.json, temperatura.json}."""

    def __init__(self, directorio: Path = DIR_FINAL, dev: torch.device | None = None):
        self.dev = dev or dispositivo()
        info = json.loads((directorio / "info.json").read_text())
        base = info["base"]
        self.temperatura = json.loads((directorio / "temperatura.json").read_text())["T"]
        self.cod = CodificadorTransformer(base, max_len=256)
        self.modelo = ClasificadorTransformer(base)
        self.modelo.load_state_dict(torch.load(directorio / "pesos.pt", map_location="cpu"))
        self.modelo.to(self.dev).eval()

    @torch.no_grad()
    def clasificar(self, texto: str) -> ResultadoClasificacion:
        enc = self.cod.tok(
            texto,
            truncation=True,
            max_length=self.cod.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids, mask = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
        hf = self.modelo.hf
        out = hf(
            input_ids=ids, attention_mask=mask, output_attentions=True, output_hidden_states=True
        )
        probs = softmax(out.logits.float().cpu().numpy(), self.temperatura)[0]
        oculto = out.hidden_states[-1]
        imp = rollout_atencion(out.attentions, mask).cpu().numpy()
        palabras, pesos = _agrupar_palabras(texto, offsets, imp)
        return ResultadoClasificacion(
            p_verdadera=float(probs[0]),
            p_falsa=float(probs[1]),
            palabras=palabras,
            pesos_atencion=pesos,
            cls=oculto[:, 0].float().cpu(),
            tokens=oculto.float().cpu(),
            mascara=mask.cpu(),
        )

    @torch.no_grad()
    def embeber(self, textos: list[str], max_len: int = 64, batch: int = 16):
        """Embeddings para entrenar el propagador: (cls[N,d], tokens[N,L,d] fp16, mascara[N,L])."""
        cls, tok, msk = [], [], []
        for i in range(0, len(textos), batch):
            enc = self.cod.tok(
                textos[i : i + batch],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            ).to(self.dev)
            h = self.modelo.hf.base_model(**enc).last_hidden_state
            cls.append(h[:, 0].float().cpu())
            tok.append(h.half().cpu())
            msk.append(enc["attention_mask"].cpu())
        return torch.cat(cls), torch.cat(tok), torch.cat(msk)


def _agrupar_palabras(texto: str, offsets: list[tuple[int, int]], imp: np.ndarray):
    """Une subtokens en palabras (separadas por espacios) sumando su importancia; normaliza a [0, 1]."""
    palabras, pesos, fin_prev = [], [], -1
    for (ini, fin), w in zip(offsets, imp, strict=True):
        if ini == fin:  # tokens especiales <s>, </s>
            continue
        if palabras and ini == fin_prev:  # continúa la palabra anterior (sin espacio)
            palabras[-1] += texto[ini:fin]
            pesos[-1] += float(w)
        else:
            palabras.append(texto[ini:fin])
            pesos.append(float(w))
        fin_prev = fin
    if pesos:
        p = np.array(pesos)
        # Se recorta en el percentil 90: el rollout concentra masa en 1-2 tokens (cercanos a </s>)
        # y aplastaría al resto al normalizar.
        p = np.clip(p, 0, np.percentile(p, 90))
        p = (p - p.min()) / (p.max() - p.min() + 1e-9)
        pesos = p.tolist()
    return palabras, pesos
