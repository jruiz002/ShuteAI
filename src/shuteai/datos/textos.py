"""Unificación de los conjuntos de texto del clasificador (verdadera=0 / falsa=1).

Fuentes: Spanish Fake News Corpus + FakeDeS (es), LIAR (en) y Twitter15/16 (en).
Salida: data/processed/clasificador.csv con columnas
    id, texto, etiqueta, idioma, origen, particion
"""

from __future__ import annotations

import pandas as pd

from shuteai.datos.particiones import particion_por_id
from shuteai.preprocesamiento.limpieza import limpiar_texto, quitar_firmas
from shuteai.rutas import CRUDOS, PROCESADOS
from shuteai.utils import cargar_config

COLUMNAS = ["id", "texto", "etiqueta", "idioma", "origen", "particion"]
SALIDA = PROCESADOS / "clasificador.csv"


def _recortar(texto: str, max_palabras: int) -> str:
    return " ".join(texto.split()[:max_palabras])


def _es_falsa(valor) -> bool:
    """Train/dev usan 'Fake'/'True'; el test de FakeDeS trae booleanos (False = noticia falsa)."""
    if isinstance(valor, bool):
        return not valor
    return str(valor).strip().lower() in {"fake", "false"}


def cargar_espanol() -> pd.DataFrame:
    """Particiones oficiales: train, development (=val) y test (FakeDeS, IberLEF 2021)."""
    cfg = cargar_config().datos
    partes = []
    for archivo, particion in (("train", "train"), ("development", "val"), ("test", "test")):
        df = pd.read_excel(CRUDOS / "spanish_fakenews" / f"{archivo}.xlsx")
        df.columns = [c.strip().capitalize() for c in df.columns]  # el test viene en MAYÚSCULAS
        titular = df["Headline"].fillna("").astype(str)
        cuerpo = df["Text"].fillna("").astype(str)
        # Si el cuerpo ya empieza con el titular no se repite.
        ya_incluido = [c.startswith(t[:40]) for t, c in zip(titular, cuerpo, strict=True)]
        crudo = (titular + ". " + cuerpo).where(
            (titular.str.len() > 0) & ~pd.Series(ya_incluido), cuerpo
        )
        texto = [
            _recortar(quitar_firmas(limpiar_texto(t), s), cfg.max_palabras_texto)
            for t, s in zip(crudo, df["Source"], strict=True)
        ]
        partes.append(
            pd.DataFrame(
                {
                    "id": [f"es-{archivo}-{i}" for i in df["Id"]],
                    "texto": texto,
                    "etiqueta": df["Category"].map(_es_falsa).astype(int),
                    "idioma": "es",
                    "origen": "spanish_fakenews",
                    "particion": particion,
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def cargar_liar() -> pd.DataFrame:
    cfg = cargar_config().datos
    mapa = {**{k: 1 for k in cfg.liar_falsas}, **{k: 0 for k in cfg.liar_verdaderas}}
    partes = []
    for archivo, particion in (("train", "train"), ("valid", "val"), ("test", "test")):
        df = pd.read_csv(
            CRUDOS / "liar" / f"{archivo}.tsv",
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["id", "label", "statement"],
            quoting=3,
        )
        partes.append(
            pd.DataFrame(
                {
                    "id": "liar-" + df["id"].astype(str),
                    "texto": df["statement"].astype(str).map(limpiar_texto),
                    "etiqueta": df["label"].map(mapa),
                    "idioma": "en",
                    "origen": "liar",
                    "particion": particion,
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def cargar_twitter_fuentes() -> pd.DataFrame:
    """Tuits fuente de Twitter15/16 con sus 4 etiquetas originales (se usan también en propagación)."""
    cfg = cargar_config().datos
    filas = []
    base = CRUDOS / "twitter1516" / "rumor_detection_acl2017"
    for corpus in ("twitter15", "twitter16"):
        etiquetas = {}
        for linea in (base / corpus / "label.txt").read_text(encoding="utf-8").splitlines():
            etiqueta, tid = linea.strip().split(":")
            etiquetas[tid] = etiqueta
        for linea in (base / corpus / "source_tweets.txt").read_text(encoding="utf-8").splitlines():
            tid, texto = linea.split("\t", 1)
            texto = limpiar_texto(texto.replace("URL", " "))
            filas.append(
                {
                    "id": tid,
                    "texto": texto,
                    "etiqueta_original": etiquetas[tid],
                    "corpus": corpus,
                    "particion": particion_por_id(tid, cfg.prop_val_frac, cfg.prop_test_frac),
                }
            )
    df = pd.DataFrame(filas).drop_duplicates("id")
    return df[df["texto"].str.split().str.len() > 0].reset_index(drop=True)


def cargar_twitter_clasificador() -> pd.DataFrame:
    """Solo rumores confirmados verdaderos / falsos (se descartan 'unverified' y 'non-rumor')."""
    df = cargar_twitter_fuentes()
    df = df[df["etiqueta_original"].isin(["true", "false"])]
    return pd.DataFrame(
        {
            "id": "tw-" + df["id"],
            "texto": df["texto"],
            "etiqueta": (df["etiqueta_original"] == "false").astype(int),
            "idioma": "en",
            "origen": "twitter1516",
            "particion": df["particion"],
        }
    )


def construir() -> pd.DataFrame:
    df = pd.concat(
        [cargar_espanol(), cargar_liar(), cargar_twitter_clasificador()], ignore_index=True
    )
    df = df.dropna(subset=["etiqueta"]).astype({"etiqueta": int})
    df = df[df["texto"].str.split().str.len() >= 3]
    # Duplicados exactos de texto: se conserva el primero (evita fuga entre particiones).
    df = df.drop_duplicates("texto", keep="first")
    return df[COLUMNAS].reset_index(drop=True)


def main() -> None:
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    df = construir()
    df.to_csv(SALIDA, index=False)
    print(f"Guardado {SALIDA} ({len(df)} filas)")
    print(df.groupby(["origen", "particion"])["etiqueta"].agg(["count", "mean"]).round(3))


if __name__ == "__main__":
    main()
