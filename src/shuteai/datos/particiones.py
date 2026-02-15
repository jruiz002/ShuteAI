"""Particiones deterministas: el mismo ID cae siempre en la misma partición (sin fugas de datos)."""

from __future__ import annotations

import hashlib


def particion_por_id(identificador: str | int, val: float = 0.15, test: float = 0.15) -> str:
    """Asigna 'train' / 'val' / 'test' según un hash estable del ID (independiente de la semilla)."""
    h = int(hashlib.md5(str(identificador).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"
