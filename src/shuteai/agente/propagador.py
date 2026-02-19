"""Envoltorio de inferencia del modelo de propagación (modo A) con banda de incertidumbre empírica (bootstrap de residuos)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shuteai.propagacion.entrenar import DIR_PROP, construir_modo_a
from shuteai.propagacion.incertidumbre import ARCHIVO, curvas_desde_residuos
from shuteai.utils import cargar_config, dispositivo

MAX_TOKENS = 64  # igual que en el entrenamiento (embeddings precalculados con 64 tokens)


class PropagadorFinal:
    def __init__(self, directorio: Path | None = None, dev: torch.device | None = None):
        self.dev = dev or dispositivo()
        cfg = cargar_config().propagacion
        self.n_muestras = cfg.muestras_incertidumbre
        if directorio is None:
            tipo = json.loads((DIR_PROP / "final.json").read_text())["tipo"]
            directorio = DIR_PROP / f"a_{tipo}"
        else:
            tipo = json.loads((directorio / "info.json").read_text())["tipo"]
        self.tipo = tipo
        self.modelo = construir_modo_a(tipo, cfg)
        self.modelo.load_state_dict(torch.load(directorio / "pesos.pt", map_location="cpu"))
        self.modelo.to(self.dev).eval()
        ruta = DIR_PROP / ARCHIVO
        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta {ruta}: ejecuta `python -m shuteai.propagacion.incertidumbre`."
            )
        self.residuos = np.load(ruta)

    @torch.no_grad()
    def muestras(
        self, cls: torch.Tensor, tokens: torch.Tensor, mascara: torch.Tensor
    ) -> np.ndarray:
        """Devuelve (n, 96) curvas log1p posibles: predicción + residuos reales de validación."""
        cls = cls.to(self.dev)
        tokens, mascara = tokens[:, :MAX_TOKENS].to(self.dev), mascara[:, :MAX_TOKENS].to(self.dev)
        self.modelo.eval()
        pred = self.modelo(cls, tokens, mascara).cpu().numpy()[0]
        return curvas_desde_residuos(pred, self.residuos, self.n_muestras)
