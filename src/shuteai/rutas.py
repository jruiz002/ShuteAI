"""Rutas centrales del proyecto (una sola fuente de verdad)."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
CRUDOS = DATOS / "raw"
PROCESADOS = DATOS / "processed"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
CONFIG = RAIZ / "configs" / "default.yaml"
EVAL_PROPIA = DATOS / "eval_propia" / "noticias_prueba.csv"
