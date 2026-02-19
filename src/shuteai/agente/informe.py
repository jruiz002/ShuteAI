"""Redacción del informe final (plantilla de texto) y gráfica de la curva de propagación."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shuteai.agente.agente import Informe
from shuteai.agente.informe_tipos import LIMITACION

HORAS = (np.arange(96) + 1) * 0.5


def _rango(t: tuple[float, float, float], unidad: str = "h") -> str:
    _, lo, hi = t
    return f"{lo:g}–{hi:g} {unidad}" if lo != hi else f"{lo:g} {unidad}"


def redactar(i: Informe) -> str:
    """Informe en Markdown. Siempre muestra incertidumbre y la limitación de no verificar hechos."""
    lineas = [
        f"### La noticia es **{i.veredicto}** ({i.confianza:.0%})",
        f"P(verdadera) = {i.p_verdadera:.1%} · P(falsa) = {i.p_falsa:.1%} · umbral = {i.umbral:.0%}",
    ]
    if i.es_falsa and i.curva is not None:
        pico, alc, t50 = i.pico_h, i.alcance, i.t50_h
        lineas += [
            "",
            f"Se estima que alcanzaría su **pico de difusión entre las horas {_rango(pico, '')}**"
            f" (mediana: hora {pico[0]:g}) y que la mitad de su alcance total ocurriría en las primeras "
            f"**{t50[0]:g} horas**.",
            f"Alcance estimado a 48 h: **~{alc[0]:,.0f} compartidos** "
            f"(rango plausible {alc[1]:,.0f}–{alc[2]:,.0f}).",
            "",
            "_La curva es una estimación del patrón de propagación aprendido de Twitter (2015–2016), "
            "no una predicción exacta._",
        ]
    else:
        lineas += [
            "",
            "No se estima propagación porque la noticia no supera el umbral de falsedad.",
        ]
    lineas += ["", f"> {LIMITACION}"]
    for a in i.advertencias:
        lineas.append(f"> ⚠️ {a}")
    return "\n".join(lineas)


def figura_propagacion(i: Informe) -> go.Figure:
    fig = go.Figure()
    if i.curva is None:
        return fig
    reach = np.expm1
    fig.add_trace(
        go.Scatter(
            x=HORAS, y=reach(i.banda_alta), line=dict(width=0), hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=HORAS,
            y=reach(i.banda_baja),
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(220,53,69,0.18)",
            name="Incertidumbre (10–90 %)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=HORAS,
            y=reach(i.curva),
            name="Alcance acumulado estimado",
            line=dict(color="#dc3545", width=3),
            hovertemplate="Hora %{x:.1f}<br>~%{y:,.0f} compartidos<extra></extra>",
        )
    )
    fig.add_vline(
        x=i.pico_h[0],
        line_dash="dot",
        line_color="#6c757d",
        annotation_text=f"Pico ≈ h {i.pico_h[0]:g}",
        annotation_position="top",
    )
    fig.add_vline(
        x=i.t50_h[0],
        line_dash="dash",
        line_color="#0d6efd",
        annotation_text=f"50 % alcance ≈ h {i.t50_h[0]:g}",
        annotation_position="bottom",
    )
    fig.update_layout(
        xaxis_title="Horas desde la publicación",
        yaxis_title="Veces compartida (acumulado)",
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    return fig
