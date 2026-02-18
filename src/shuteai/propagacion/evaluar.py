"""Evaluación de propagación: modo A (texto -> 48 h), modo B (6 h -> 42 h) y validación en PHEME.

    python -m shuteai.propagacion.evaluar
Genera reports/propagacion_resultados.{json,md} y reports/figuras/propagacion_*.png.
"""

from __future__ import annotations

import json
import json as _json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shuteai.propagacion.baselines import CurvaPromedio, ModeloSIR
from shuteai.propagacion.entrenar import (
    DIR_PROP,
    cargar_embeddings,
    construir_modo_a,
    construir_modo_b,
    particiones,
    predecir_modo_a,
)
from shuteai.propagacion.metricas import errores_por_rango, resumen_modo_a, resumen_modo_b
from shuteai.rutas import REPORTES
from shuteai.utils import cargar_config, dispositivo

FIGURAS = REPORTES / "figuras"
HORAS = (np.arange(96) + 1) * 0.5


def _cargar(nombre: str, constructor, *args):
    d = DIR_PROP / nombre
    if not (d / "pesos.pt").exists():
        return None
    info = _json.loads((d / "info.json").read_text())
    m = constructor(info["tipo"], *args)
    m.load_state_dict(torch.load(d / "pesos.pt", map_location="cpu"))
    return m.to(dispositivo()).eval()


def evaluar_modo_a(cfg) -> dict:
    dev = dispositivo()
    tw, ph = cargar_embeddings("twitter"), cargar_embeddings("pheme")
    idx = particiones(tw)
    etiq = tw["meta"]["etiqueta_original"].values
    y = tw["curvas"].numpy()
    falsas_train = idx["train"][etiq[idx["train"]] == "false"]
    medio = CurvaPromedio().ajustar(y[falsas_train])
    todo_ph = np.arange(len(ph["meta"]))
    ph_y = ph["curvas"].numpy()
    ph_falsas = todo_ph[ph["meta"]["etiqueta_original"].values == "false"]

    modelos = {"curva promedio": None}
    for t in ("mlp", "lstm", "transformer"):
        m = _cargar(f"a_{t}", construir_modo_a, cfg)
        if m is not None:
            modelos[t] = m

    res = {}
    for nombre, m in modelos.items():

        def pred(datos, ix, m=m):
            if m is None:
                return medio.predecir(len(ix))
            return predecir_modo_a(m, datos, ix, dev)

        test_f = idx["test"][etiq[idx["test"]] == "false"]
        res[nombre] = {
            "twitter_test_todas": resumen_modo_a(y[idx["test"]], pred(tw, idx["test"])),
            "twitter_test_falsas": resumen_modo_a(y[test_f], pred(tw, test_f)),
            "pheme_falsas_zero_shot": resumen_modo_a(ph_y[ph_falsas], pred(ph, ph_falsas)),
            "por_rango_alcance_test": errores_por_rango(y[idx["test"]], pred(tw, idx["test"])),
        }
    return res


def evaluar_modo_b(cfg) -> dict:
    dev = dispositivo()
    cp = cargar_config().propagacion_datos
    ctx = int(cp.horas_contexto_modo_b * 60 / cp.paso_min)
    tw = cargar_embeddings("twitter")
    idx = particiones(tw)
    y = tw["curvas"].numpy()
    hor = y.shape[1] - ctx
    etiq = tw["meta"]["etiqueta_original"].values
    test = idx["test"]
    test_f = test[etiq[test] == "false"]
    medio = CurvaPromedio().ajustar(y[idx["train"][etiq[idx["train"]] == "false"]])
    res = {}
    for grupo, ix in (("todas", test), ("falsas", test_f)):
        c, f = y[ix, :ctx], y[ix, ctx:]
        res[f"curva promedio anclada|{grupo}"] = resumen_modo_b(f, medio.predecir_futuro(c, hor))
        res[f"SIR|{grupo}"] = resumen_modo_b(f, ModeloSIR().predecir_futuro(c, hor))
        for t in ("lstm", "transformer", "patchtst"):
            m = _cargar(f"b_{t}", construir_modo_b, cfg, ctx, hor)
            if m is None:
                continue
            with torch.no_grad():
                p = m(torch.tensor(c).to(dev)).cpu().numpy()
            res[f"{t}|{grupo}"] = resumen_modo_b(f, p)
    return res


def analisis_falsas_vs_verdaderas() -> dict:
    """Análisis del plan: ¿las falsas se difunden más lejos/rápido que las verdaderas (Vosoughi 2018)?"""
    import pandas as pd

    from shuteai.propagacion.metricas import alcance_final, hora_mitad_alcance
    from shuteai.rutas import PROCESADOS

    out = {}
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, nombre in zip(axs, ("twitter", "pheme"), strict=True):
        curvas = np.load(PROCESADOS / f"propagacion_{nombre}.npy")
        meta = pd.read_csv(PROCESADOS / f"propagacion_{nombre}.csv")
        for etq, color in (("false", "#dc3545"), ("true", "#198754")):
            c = curvas[(meta["etiqueta_original"] == etq).values]
            ax.plot(HORAS, np.expm1(c).mean(0), color=color, label=f"{etq} (n={len(c)})")
            q = np.percentile(np.expm1(c), [25, 75], axis=0)
            ax.fill_between(HORAS, q[0], q[1], color=color, alpha=0.12)
            out[f"{nombre}_{etq}"] = {
                "n": len(c),
                "alcance_medio": float(alcance_final(c).mean()),
                "alcance_mediano": float(np.median(alcance_final(c))),
                "t50_mediano_h": float(np.median(hora_mitad_alcance(c))),
            }
        ax.set(title=nombre.capitalize(), xlabel="Horas", ylabel="Alcance acumulado medio")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "propagacion_falsas_vs_verdaderas.png", dpi=150)
    plt.close(fig)
    return out


def _tabla(titulo: str, filas: dict, cols: list[str]) -> list[str]:
    out = [
        f"#### {titulo}",
        "",
        "| Modelo | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for n, m in filas.items():
        out.append(f"| {n} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return out + [""]


def main() -> None:
    cfg = cargar_config().propagacion
    a, b, c = evaluar_modo_a(cfg), evaluar_modo_b(cfg), analisis_falsas_vs_verdaderas()
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / "propagacion_resultados.json").write_text(
        json.dumps(
            {"modo_a": a, "modo_b": b, "falsas_vs_verdaderas": c}, indent=2, ensure_ascii=False
        )
    )
    cols_a = ["rmse", "mae", "error_pico_h", "error_alcance_log"]
    md = ["### Modo A – solo texto → curva de 48 h (escala log)", ""]
    for k, t in (
        ("twitter_test_falsas", "Twitter15/16 – prueba, noticias falsas"),
        ("twitter_test_todas", "Twitter15/16 – prueba, todas"),
        ("pheme_falsas_zero_shot", "PHEME – validación externa (falsas, sin reentrenar)"),
    ):
        md += _tabla(t, {n: r[k] for n, r in a.items()}, cols_a)
    md += ["### Modo B – 6 h observadas → 42 h pronosticadas", ""]
    for g in ("falsas", "todas"):
        md += _tabla(
            f"Prueba, {g}",
            {k.split("|")[0]: v for k, v in b.items() if k.endswith(g)},
            ["rmse", "mae", "mape"],
        )
    md += [
        "### Falsas vs. verdaderas (alcance a 48 h)",
        "",
        "| Corpus | n | alcance medio | mediano | t50 mediano (h) |",
        "|---|---|---|---|---|",
    ]
    for k, v in c.items():
        md.append(
            f"| {k} | {v['n']} | {v['alcance_medio']:.1f} | {v['alcance_mediano']:.1f} | {v['t50_mediano_h']:.1f} |"
        )
    (REPORTES / "propagacion_resultados.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
