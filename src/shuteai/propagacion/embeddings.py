"""Precalcula los embeddings del clasificador final para los textos de propagación (se hace una vez).

Salida: data/processed/emb_{twitter,pheme}.pt  con cls (N,d), tokens (N,L,d) fp16 y mascara (N,L).
En PHEME solo se usan hilos de rumor (los non-rumor no son noticias dudosas y harían el archivo enorme).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from shuteai.clasificador.inferencia import ClasificadorFinal
from shuteai.rutas import PROCESADOS


def main() -> None:
    clf = ClasificadorFinal()
    for nombre in ("twitter", "pheme"):
        meta = pd.read_csv(PROCESADOS / f"propagacion_{nombre}.csv", dtype={"id": str})
        curvas = np.load(PROCESADOS / f"propagacion_{nombre}.npy")
        if nombre == "pheme":
            m = (meta["etiqueta_original"] != "non-rumor").values
            meta, curvas = meta[m].reset_index(drop=True), curvas[m]
        cls, tokens, mascara = clf.embeber(meta["texto"].tolist())
        torch.save(
            {
                "cls": cls,
                "tokens": tokens,
                "mascara": mascara,
                "curvas": torch.tensor(curvas),
                "meta": meta,
            },
            PROCESADOS / f"emb_{nombre}.pt",
        )
        print(f"[{nombre}] {tuple(tokens.shape)} guardado")


if __name__ == "__main__":
    main()
