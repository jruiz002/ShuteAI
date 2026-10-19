"""Incertidumbre del propagador por remuestreo de residuos de validación (bootstrap de residuos).

El MC-Dropout subestima mucho el error real (el dropout apenas mueve la salida), así que la banda se
calibra empíricamente: curva_k = curva_predicha + residuo_k, con residuo_k = real - predicho en un
hilo de validación. Se conserva la correlación entre pasos y se mide la cobertura en prueba.

    python -m shuteai.propagacion.incertidumbre
"""

from __future__ import annotations

import json

import numpy as np
import torch

from shuteai.propagacion.entrenar import (
    DIR_PROP,
    cargar_embeddings,
    construir_modo_a,
    particiones,
    predecir_modo_a,
)
from shuteai.utils import cargar_config, dispositivo

ARCHIVO = "residuos_val.npy"


def curvas_desde_residuos(pred: np.ndarray, residuos: np.ndarray, n: int, rng=None) -> np.ndarray:
    """(n, 96) curvas posibles: predicción + residuos muestreados, no negativas y no decrecientes."""
    rng = rng or np.random.default_rng(0)
    r = residuos[rng.integers(0, len(residuos), n)]
    return np.maximum.accumulate(np.clip(pred[None, :] + r, 0.0, None), axis=1)


def main() -> None:
    tipo = json.loads((DIR_PROP / "final.json").read_text())["tipo"]
    cfg, dev = cargar_config().propagacion, dispositivo()
    modelo = construir_modo_a(tipo, cfg)
    modelo.load_state_dict(torch.load(DIR_PROP / f"a_{tipo}" / "pesos.pt", map_location="cpu"))
    modelo.to(dev).eval()
    datos = cargar_embeddings("twitter")
    idx, y = particiones(datos), datos["curvas"].numpy()
    res_val = y[idx["val"]] - predecir_modo_a(modelo, datos, idx["val"], dev)
    np.save(DIR_PROP / ARCHIVO, res_val.astype(np.float32))

    pred_test = predecir_modo_a(modelo, datos, idx["test"], dev)
    rng = np.random.default_rng(0)
    dentro, alcance_dentro = [], []
    for p, real in zip(pred_test, y[idx["test"]], strict=True):
        c = curvas_desde_residuos(p, res_val, 200, rng)
        lo, hi = np.percentile(c, [10, 90], axis=0)
        dentro.append(np.mean((real >= lo) & (real <= hi)))
        alcance_dentro.append(lo[-1] <= real[-1] <= hi[-1])
    out = {
        "cobertura_nominal": 0.8,
        "cobertura_pasos_test": float(np.mean(dentro)),
        "cobertura_alcance_final_test": float(np.mean(alcance_dentro)),
        "n_val": len(res_val),
    }
    (DIR_PROP / "incertidumbre.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
