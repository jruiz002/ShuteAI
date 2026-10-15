"""Utilidades transversales: configuración, semillas y dispositivo."""

from __future__ import annotations

import random
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from shuteai.rutas import CONFIG


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


@lru_cache(maxsize=1)
def cargar_config() -> SimpleNamespace:
    """Carga configs/default.yaml como objeto con acceso por atributo (cfg.clasificador.lr_base)."""
    with open(CONFIG, encoding="utf-8") as f:
        return _ns(yaml.safe_load(f))


def fijar_semilla(semilla: int | None = None) -> int:
    semilla = cargar_config().semilla if semilla is None else semilla
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    return semilla


def dispositivo() -> torch.device:
    """CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
