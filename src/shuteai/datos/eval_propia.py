"""Carga y validación del conjunto de prueba propio (50 noticias, 25 falsas + 25 verdaderas)."""

from __future__ import annotations

import pandas as pd

from shuteai.rutas import EVAL_PROPIA

COLUMNAS = ["texto", "etiqueta", "medio_o_verificador", "url_verificacion", "fecha"]


def cargar(estricto: bool = True) -> pd.DataFrame:
    df = pd.read_csv(EVAL_PROPIA).dropna(subset=["texto"])
    errores = []
    if list(df.columns) != COLUMNAS:
        errores.append(f"Columnas esperadas: {COLUMNAS}")
    else:
        if not df["etiqueta"].isin(["falsa", "verdadera"]).all():
            errores.append("`etiqueta` debe ser 'falsa' o 'verdadera'.")
        if df["url_verificacion"].isna().any():
            errores.append("Toda noticia necesita `url_verificacion` (evidencia de la etiqueta).")
        if df["texto"].duplicated().any():
            errores.append("Hay textos duplicados.")
        if estricto:
            n = df["etiqueta"].value_counts()
            if n.get("falsa", 0) != 25 or n.get("verdadera", 0) != 25:
                errores.append(f"Se requieren 25 falsas y 25 verdaderas (hay {dict(n)}).")
    if errores:
        raise ValueError("Conjunto de prueba propio inválido:\n- " + "\n- ".join(errores))
    df["y"] = (df["etiqueta"] == "falsa").astype(int)
    return df


if __name__ == "__main__":
    d = cargar()
    print(f"OK: {len(d)} noticias ({int(d.y.sum())} falsas, {int((1 - d.y).sum())} verdaderas)")
