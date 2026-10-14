"""Descarga reproducible de los conjuntos públicos (idempotente: no repite lo ya descargado).

Uso:  python -m shuteai.datos.descarga
"""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from pathlib import Path

from shuteai.rutas import CRUDOS

_GH = "https://raw.githubusercontent.com/jpposadas/FakeNewsCorpusSpanish/master"

# nombre -> (carpeta, [(url, archivo)], descompresión)
FUENTES = {
    "spanish_fakenews": (
        [(f"{_GH}/{f}", f) for f in ("train.xlsx", "development.xlsx", "test.xlsx", "LICENSE.txt")],
        None,
    ),
    "liar": (
        [("https://www.cs.ucsb.edu/~william/data/liar_dataset.zip", "liar_dataset.zip")],
        "zip",
    ),
    "twitter1516": (
        [("https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1", "rumdetect2017.zip")],
        "zip",
    ),
    "pheme": (
        [("https://ndownloader.figshare.com/files/11767817", "PHEME_veracity.tar.bz2")],
        "tar",
    ),
}

# Archivo que, si existe, indica que la fuente ya está completa.
MARCADORES = {
    "spanish_fakenews": "test.xlsx",
    "liar": "train.tsv",
    "twitter1516": "rumor_detection_acl2017/twitter15/label.txt",
    "pheme": "all-rnr-annotated-threads",
}


def _bajar(url: str, destino: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ShuteAI/0.1 (proyecto academico)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(destino, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def descargar(nombre: str) -> Path:
    archivos, modo = FUENTES[nombre]
    carpeta = CRUDOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / MARCADORES[nombre]).exists():
        print(f"[ok] {nombre}: ya descargado")
        return carpeta
    for url, archivo in archivos:
        print(f"[descargando] {nombre}/{archivo}")
        _bajar(url, carpeta / archivo)
        if modo == "zip":
            with zipfile.ZipFile(carpeta / archivo) as z:
                z.extractall(carpeta)
        elif modo == "tar":
            with tarfile.open(carpeta / archivo) as t:
                t.extractall(carpeta, filter="data")
    return carpeta


def main() -> None:
    for nombre in FUENTES:
        descargar(nombre)
    print(f"Datos listos en {CRUDOS}")


if __name__ == "__main__":
    main()
