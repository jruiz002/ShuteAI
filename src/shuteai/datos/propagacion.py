"""Construcción de las series de propagación (96 pasos de 30 min, alcance acumulado en log1p).

Fuentes:
  * Twitter15/16: árboles con el retraso (min) de cada retuit  -> entrenamiento/evaluación.
  * PHEME: hilos de respuestas con marcas de tiempo             -> validación externa.
Salida: data/processed/propagacion_{twitter,pheme}.npz  +  .csv con metadatos.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shuteai.datos.particiones import particion_por_id
from shuteai.datos.textos import cargar_twitter_fuentes
from shuteai.preprocesamiento.limpieza import limpiar_texto
from shuteai.rutas import CRUDOS, PROCESADOS
from shuteai.utils import cargar_config


def curva_desde_tiempos(tiempos, horas: int = 48, paso_min: int = 30) -> np.ndarray:
    """Alcance acumulado (log1p) en `horas*60/paso_min` pasos a partir de retrasos en minutos."""
    bordes = np.arange(0, horas * 60 + paso_min, paso_min)
    conteo, _ = np.histogram(np.asarray(tiempos, dtype=float), bins=bordes)
    return np.log1p(np.cumsum(conteo)).astype(np.float32)


def tiempos_arbol(ruta_arbol: Path) -> list[float]:
    """Retraso en minutos de cada nodo (incluida la fuente en t=0) de un árbol de Twitter15/16."""
    vistos: dict[str, float] = {}
    for linea in ruta_arbol.read_text(encoding="utf-8").splitlines():
        if "->" not in linea:
            continue
        _, hijo = linea.strip().split("->")
        _, tweet_id, t = ast.literal_eval(hijo)
        vistos.setdefault(str(tweet_id), float(t))
    return list(vistos.values())


def construir_twitter() -> tuple[np.ndarray, pd.DataFrame]:
    cfg = cargar_config().propagacion_datos
    fuentes = cargar_twitter_fuentes()
    base = CRUDOS / "twitter1516" / "rumor_detection_acl2017"
    curvas, filas = [], []
    for fila in fuentes.itertuples():
        ruta = base / fila.corpus / "tree" / f"{fila.id}.txt"
        if not ruta.exists():
            continue
        tiempos = tiempos_arbol(ruta)
        curvas.append(curva_desde_tiempos(tiempos, cfg.horas, cfg.paso_min))
        filas.append({**fila._asdict(), "n_nodos": len(tiempos)})
    meta = pd.DataFrame(filas).drop(columns="Index")
    return np.stack(curvas), meta


_FORMATO_FECHA = "%a %b %d %H:%M:%S %z %Y"


def _fecha(tuit: dict) -> datetime:
    return datetime.strptime(tuit["created_at"], _FORMATO_FECHA)


def _etiqueta_pheme(anotacion: dict) -> str:
    if anotacion.get("is_rumour") != "rumour":
        return "non-rumor"
    if str(anotacion.get("misinformation")) == "1":
        return "false"
    if str(anotacion.get("true")) == "1":
        return "true"
    return "unverified"


def construir_pheme() -> tuple[np.ndarray, pd.DataFrame]:
    cfg = cargar_config().propagacion_datos
    raiz = CRUDOS / "pheme" / "all-rnr-annotated-threads"
    curvas, filas = [], []
    for evento in sorted(p for p in raiz.iterdir() if p.is_dir()):
        for hilo in sorted(evento.glob("*/*")):
            fuente_json = next((hilo / "source-tweets").glob("*.json"), None)
            if fuente_json is None or not (hilo / "annotation.json").exists():
                continue
            fuente = json.loads(fuente_json.read_text(encoding="utf-8"))
            t0 = _fecha(fuente)
            tiempos = [0.0]
            for r in (hilo / "reactions").glob("*.json"):
                dt = (_fecha(json.loads(r.read_text(encoding="utf-8"))) - t0).total_seconds() / 60
                if dt >= 0:
                    tiempos.append(dt)
            anot = json.loads((hilo / "annotation.json").read_text(encoding="utf-8"))
            curvas.append(curva_desde_tiempos(tiempos, cfg.horas, cfg.paso_min))
            filas.append(
                {
                    "id": hilo.name,
                    "texto": limpiar_texto(fuente["text"]),
                    "etiqueta_original": _etiqueta_pheme(anot),
                    "evento": evento.name.replace("-all-rnr-threads", ""),
                    "n_nodos": len(tiempos),
                    "particion": particion_por_id(hilo.name),
                }
            )
    return np.stack(curvas), pd.DataFrame(filas)


def guardar(nombre: str, curvas: np.ndarray, meta: pd.DataFrame) -> None:
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    np.save(PROCESADOS / f"propagacion_{nombre}.npy", curvas)
    meta.to_csv(PROCESADOS / f"propagacion_{nombre}.csv", index=False)
    print(f"[{nombre}] curvas {curvas.shape}")
    print(meta.groupby("etiqueta_original")["n_nodos"].agg(["count", "median", "max"]))


def main() -> None:
    guardar("twitter", *construir_twitter())
    guardar("pheme", *construir_pheme())


if __name__ == "__main__":
    main()
