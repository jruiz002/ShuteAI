"""Evaluación del agente completo en las 50 noticias propias: aciertos y tiempo de respuesta."""

from __future__ import annotations

import json

import numpy as np

from shuteai.agente.agente import Agente
from shuteai.clasificador.metricas import metricas_clasificacion
from shuteai.datos.eval_propia import cargar
from shuteai.rutas import REPORTES


def main() -> None:
    df = cargar()
    ag = Agente()
    ag.analizar(df["texto"].iloc[0])  # calentamiento: no cuenta en los tiempos
    informes = [ag.analizar(t) for t in df["texto"]]
    probs = np.array([[i.p_verdadera, i.p_falsa] for i in informes])
    m = metricas_clasificacion(df["y"].values, probs)
    seg = np.array([i.segundos for i in informes])
    m |= {
        "tiempo_medio_s": float(seg.mean()),
        "tiempo_p95_s": float(np.percentile(seg, 95)),
        "propagacion_ejecutada": int(sum(i.es_falsa for i in informes)),
    }
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / "agente_resultados.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
