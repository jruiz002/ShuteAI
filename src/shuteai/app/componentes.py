"""Piezas visuales (HTML) de la interfaz. Funciones puras: fáciles de probar y de reutilizar."""

from __future__ import annotations

from html import escape

from shuteai.agente.agente import Informe

VACIO = (
    '<div class="sh-vacio">Pega una noticia a la izquierda y pulsa <b>Analizar</b>. '
    "Verás la probabilidad de que sea falsa y, si lo es, cómo podría propagarse en 48 horas.</div>"
)


def tarjeta_veredicto(i: Informe) -> str:
    clase = "falsa" if i.es_falsa else "verdadera"
    icono = "⚠️" if i.es_falsa else "✅"
    return f"""
<div class="sh-veredicto sh-{clase}">
  <div class="sh-titulo">{icono} Esta noticia es <b>{escape(i.veredicto)}</b></div>
  <div class="sh-confianza">{i.confianza:.0%} de probabilidad</div>
  {_barra("Verdadera", i.p_verdadera, "ok")}
  {_barra("Falsa", i.p_falsa, "mal")}
  <div class="sh-nota">Umbral de decisión: P(falsa) ≥ {i.umbral:.0%} · idioma detectado: {escape(i.idioma)}
  · {i.segundos:.1f} s</div>
</div>"""


def _barra(nombre: str, p: float, tono: str) -> str:
    return (
        f'<div class="sh-fila"><span class="sh-etq">{nombre}</span>'
        f'<div class="sh-pista"><div class="sh-relleno sh-{tono}" style="width:{p * 100:.1f}%"></div></div>'
        f'<span class="sh-val">{p:.1%}</span></div>'
    )


def tarjetas_metricas(i: Informe) -> str:
    if not i.es_falsa or i.pico_h is None:
        return ""
    pico, alc, t50 = i.pico_h, i.alcance, i.t50_h

    def rango(t, unidad=""):
        return f"{t[1]:g}–{t[2]:g}{unidad}" if t[1] != t[2] else f"{t[0]:g}{unidad}"

    cajas = [
        ("Pico de difusión", f"hora {pico[0]:g}", f"rango probable: {rango(pico, ' h')}"),
        ("50 % del alcance", f"{t50[0]:g} h", f"rango probable: {rango(t50, ' h')}"),
        ("Alcance a 48 h", f"~{alc[0]:,.0f}", f"rango plausible: {alc[1]:,.0f}–{alc[2]:,.0f}"),
    ]
    html = "".join(
        f'<div class="sh-caja"><div class="sh-caja-t">{t}</div><div class="sh-caja-v">{v}</div>'
        f'<div class="sh-caja-s">{s}</div></div>'
        for t, v, s in cajas
    )
    return f'<div class="sh-cajas">{html}</div>'


CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
.sh-hero { padding: 8px 4px 2px; }
.sh-hero h1 { margin: 0; font-size: 2rem; letter-spacing: -0.5px; }
.sh-hero p { margin: 4px 0 0; opacity: .75; }
.sh-vacio { padding: 36px 20px; text-align: center; border: 2px dashed var(--border-color-primary);
  border-radius: 14px; opacity: .8; }
.sh-veredicto { border-radius: 14px; padding: 18px 20px; border: 1px solid var(--border-color-primary); }
.sh-falsa { background: rgba(220,53,69,.10); border-color: rgba(220,53,69,.5); }
.sh-verdadera { background: rgba(25,135,84,.10); border-color: rgba(25,135,84,.5); }
.sh-titulo { font-size: 1.25rem; }
.sh-confianza { font-size: 2.2rem; font-weight: 700; margin: 2px 0 10px; }
.sh-falsa .sh-confianza { color: #dc3545; } .sh-verdadera .sh-confianza { color: #198754; }
.sh-fila { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.sh-etq { width: 78px; font-size: .9rem; } .sh-val { width: 56px; text-align: right; font-variant-numeric: tabular-nums; }
.sh-pista { flex: 1; height: 12px; border-radius: 8px; background: rgba(128,128,128,.22); overflow: hidden; }
.sh-relleno { height: 100%; border-radius: 8px; transition: width .6s ease; }
.sh-ok { background: #198754; } .sh-mal { background: #dc3545; }
.sh-nota { margin-top: 10px; font-size: .8rem; opacity: .65; }
.sh-cajas { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0; }
.sh-caja { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px; }
.sh-caja-t { font-size: .8rem; opacity: .7; } .sh-caja-v { font-size: 1.5rem; font-weight: 700; }
.sh-caja-s { font-size: .75rem; opacity: .6; }
@media (max-width: 700px) { .sh-cajas { grid-template-columns: 1fr; } }
"""
