"""Evaluación y calibración de los clasificadores entrenados.

    python -m shuteai.clasificador.evaluar                 # tabla comparativa
    python -m shuteai.clasificador.evaluar --final xlmr_en_es   # elige el modelo desplegado
Genera reports/clasificador_resultados.{json,md} y diagramas de confiabilidad en reports/figuras/.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shuteai.clasificador.calibracion import ajustar_temperatura
from shuteai.clasificador.metricas import (
    brier,
    curva_confiabilidad,
    ece,
    metricas_clasificacion,
    softmax,
)
from shuteai.rutas import MODELOS, REPORTES

DIR = MODELOS / "clasificador"
FIGURAS = REPORTES / "figuras"
GRUPOS = {
    "español": lambda d: d.idioma == "es",
    "inglés": lambda d: d.idioma == "en",
    "todo": lambda d: d.idioma.isin(["es", "en"]),
}


def _logits(df: pd.DataFrame) -> np.ndarray:
    return df[["logit_verdadera", "logit_falsa"]].values


def evaluar_modelo(nombre: str) -> dict:
    val = pd.read_csv(DIR / nombre / "logits_val.csv")
    test = pd.read_csv(DIR / nombre / "logits_test.csv")
    T = ajustar_temperatura(_logits(val), val["etiqueta"].values)
    out = {"temperatura": T, "grupos": {}}
    for g, f in GRUPOS.items():
        t = test[f(test)]
        if t.empty:  # p. ej. modelos solo-español no se evalúan en inglés
            continue
        y, lg = t["etiqueta"].values, _logits(t)
        antes, despues = softmax(lg), softmax(lg, T)
        out["grupos"][g] = {
            **metricas_clasificacion(y, despues),
            "ece_antes": ece(y, antes),
            "ece_despues": ece(y, despues),
            "brier_antes": brier(y, antes),
            "brier_despues": brier(y, despues),
        }
    return out


def diagrama_confiabilidad(nombre: str, T: float) -> None:
    test = pd.read_csv(DIR / nombre / "logits_test.csv")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="calibración perfecta")
    y, lg = test["etiqueta"].values, _logits(test)
    for etiqueta, probs in (("antes", softmax(lg)), (f"después (T={T:.2f})", softmax(lg, T))):
        x, a, _ = curva_confiabilidad(y, probs)
        ax.plot(x, a, "o-", label=etiqueta)
    ax.set(
        xlabel="Confianza predicha", ylabel="Exactitud observada", title=f"Confiabilidad – {nombre}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / f"confiabilidad_{nombre}.png", dpi=150)
    plt.close(fig)


def _tabla_ablaciones(res: dict) -> list[str]:
    """Ablaciones de S8: cada variante frente a su referencia (misma semilla y datos)."""
    pares = [("bilstm", "abl_bilstm_", "todo"), ("xlmr_solo_es", "abl_xlmr_", "español")]
    filas = [
        "",
        "#### Ablaciones (S8)",
        "",
        "| Referencia | Variante | Grupo | F1 macro | ECE (calibrado) |",
        "|---|---|---|---|---|",
    ]
    for ref, prefijo, grupo in pares:
        for n in [ref] + sorted(k for k in res if k.startswith(prefijo)):
            if n in res and grupo in res[n]["grupos"]:
                m = res[n]["grupos"][grupo]
                filas.append(
                    f"| {ref} | {'(referencia)' if n == ref else n.removeprefix(prefijo)} | {grupo} | "
                    f"{m['f1_macro']:.3f} | {m['ece_despues']:.3f} |"
                )
    return filas if len(filas) > 5 else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--final", help="Nombre del modelo a desplegar (crea models/clasificador/final)"
    )
    args = ap.parse_args()
    nombres = sorted(
        p.name for p in DIR.iterdir() if (p / "logits_test.csv").exists() and p.name != "final"
    )
    res = {n: evaluar_modelo(n) for n in nombres}
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / "clasificador_resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False)
    )

    filas = [
        "| Modelo | Grupo | Acc | F1 macro | AUC | ECE antes→después | Brier antes→después |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, r in res.items():
        if n.startswith("abl_"):
            continue
        for g, m in r["grupos"].items():
            filas.append(
                f"| {n} | {g} | {m['accuracy']:.3f} | {m['f1_macro']:.3f} | "
                f"{(m['auc_roc'] or 0):.3f} | {m['ece_antes']:.3f}→{m['ece_despues']:.3f} | "
                f"{m['brier_antes']:.3f}→{m['brier_despues']:.3f} |"
            )
    filas += _tabla_ablaciones(res)
    (REPORTES / "clasificador_resultados.md").write_text("\n".join(filas) + "\n")
    print("\n".join(filas))

    for n in nombres:
        diagrama_confiabilidad(n, res[n]["temperatura"])
    if args.final:
        destino = DIR / args.final
        (destino / "temperatura.json").write_text(json.dumps({"T": res[args.final]["temperatura"]}))
        enlace = DIR / "final"
        if enlace.is_symlink() or enlace.exists():
            enlace.unlink()
        os.symlink(destino.name, enlace)
        print(f"Modelo final = {args.final} (T={res[args.final]['temperatura']:.3f})")


if __name__ == "__main__":
    main()
