"""Interfaz web de ShuteAI (Gradio).  Uso:  python -m shuteai.app.interfaz  ->  http://127.0.0.1:7860"""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go

from shuteai.agente.agente import Agente
from shuteai.agente.informe import figura_propagacion, redactar
from shuteai.agente.informe_tipos import LIMITACION
from shuteai.app.componentes import CSS, VACIO, tarjeta_veredicto, tarjetas_metricas
from shuteai.rutas import REPORTES

# Ejemplos ilustrativos escritos para la demo (no son noticias reales verificadas).
EJEMPLOS = [
    [
        "URGENTE: el gobierno anunció que mañana cortarán el agua en toda la ciudad durante 15 días, "
        "compartan antes de que lo borren"
    ],
    [
        "El Banco Central mantuvo su tasa de interés en 5,25 % tras la reunión de política monetaria "
        "del jueves, según el comunicado oficial publicado en su sitio web."
    ],
    [
        "BREAKING: Scientists confirm that drinking hot lemon water cures cancer in 48 hours and "
        "doctors are hiding it. Share before it gets deleted!"
    ],
]

_agente: Agente | None = None


def _obtener_agente() -> Agente:
    global _agente
    if _agente is None:
        _agente = Agente()
    return _agente


def analizar(texto: str, umbral: float, agente: Agente | None = None):
    """Devuelve (veredicto_html, metricas_html, figura, palabras, informe_md)."""
    if not texto or not texto.strip():
        raise gr.Error("Escribe o pega una noticia para analizar.")
    try:
        informe = (agente or _obtener_agente()).analizar(texto, umbral=umbral)
    except FileNotFoundError as e:
        raise gr.Error(
            "Faltan los modelos entrenados. Ejecuta `make entrenar` (ver README) antes de usar la app."
        ) from e
    except ValueError as e:
        raise gr.Error(str(e)) from e
    palabras = list(zip(informe.palabras, informe.pesos_atencion, strict=True))
    return (
        tarjeta_veredicto(informe),
        tarjetas_metricas(informe),
        figura_propagacion(informe) if informe.es_falsa else go.Figure(),
        palabras,
        redactar(informe),
    )


def _tabla_resultados() -> str:
    ruta = REPORTES / "clasificador_resultados.md"
    prop = REPORTES / "propagacion_resultados.md"
    partes = []
    for titulo, p in (("Clasificador (conjunto de prueba)", ruta), ("Propagación", prop)):
        partes.append(
            f"### {titulo}\n"
            + (p.read_text() if p.exists() else "_Aún no hay resultados. Ejecuta `make evaluar`._")
        )
    return "\n\n".join(partes)


COMO_FUNCIONA = """
### Cómo funciona ShuteAI
1. **Preparación del texto** – se limpian URLs, emojis y espacios, y se extrae la noticia si escribiste
   algo como «¿es falsa esta noticia?: …».
2. **Clasificador Transformer** (XLM-RoBERTa con fine-tuning) – estima **P(verdadera)** y **P(falsa)**.
   Las probabilidades están *calibradas* (temperature scaling) para que un «80 %» signifique ≈80 %.
3. **Modelo de propagación** – solo si **P(falsa) ≥ umbral**: un encoder-decoder LSTM con atención
   pronostica cuántas veces se compartiría en 48 h. La banda sombreada es la incertidumbre
   (bootstrap de residuos reales de validación; cobertura ≈75 % en prueba para una banda nominal del 80 %).

**Palabras que más pesaron**: mapa de atención del clasificador (*attention rollout*). Es una pista
de qué texto influyó, no una explicación causal.
"""

ETICA = f"""
### Límites y uso responsable
{LIMITACION}

- Los datos de propagación provienen de **Twitter 2015–2016**; la curva es una estimación del patrón, no
  una predicción exacta, y las noticias virales son raras y muy variables.
- Con pocos datos en español, el desempeño es **mejor en inglés**; consulta la pestaña de resultados.
- El modelo puede aprender el **estilo del medio** y no la veracidad. Se quitaron nombres de medios y firmas
  al entrenar, pero el sesgo no desaparece.
- Úsalo como apoyo para **priorizar qué verificar**, nunca como juez final ni para censurar contenido.
"""


def construir(agente: Agente | None = None) -> gr.Blocks:
    with gr.Blocks(title="ShuteAI · Detector de noticias falsas") as demo:
        gr.HTML(
            '<div class="sh-hero"><h1>🛡️ ShuteAI</h1>'
            "<p>Detecta noticias probablemente falsas y estima cómo se propagarían en redes durante 48 horas.</p></div>"
        )
        with gr.Tabs():
            with gr.Tab("Analizar"):
                with gr.Row():
                    with gr.Column(scale=5):
                        texto = gr.Textbox(
                            label="Noticia",
                            lines=9,
                            max_lines=20,
                            placeholder="Pega aquí el texto de la noticia (español o inglés)…",
                        )
                        with gr.Accordion("Opciones avanzadas", open=False):
                            umbral = gr.Slider(
                                0.05,
                                0.95,
                                value=0.5,
                                step=0.05,
                                label="Umbral de falsedad",
                                info="Si P(falsa) lo supera, se estima la propagación.",
                            )
                        with gr.Row():
                            limpiar = gr.Button("Limpiar")
                            enviar = gr.Button("Analizar", variant="primary")
                        gr.Examples(EJEMPLOS, inputs=texto, label="Ejemplos (ilustrativos)")
                    with gr.Column(scale=7):
                        veredicto = gr.HTML(VACIO)
                        metricas = gr.HTML()
                        curva = gr.Plot(label="Curva de propagación estimada (48 h)")
                        palabras = gr.HighlightedText(
                            label="Palabras que más pesaron en la decisión", show_legend=False
                        )
                        informe = gr.Markdown()
                salidas = [veredicto, metricas, curva, palabras, informe]
                fn = lambda t, u: analizar(t, u, agente)  # noqa: E731
                enviar.click(fn, [texto, umbral], salidas, api_name="analizar")
                texto.submit(fn, [texto, umbral], salidas)
                limpiar.click(
                    lambda: ("", VACIO, "", None, [], ""),
                    None,
                    [texto, veredicto, metricas, curva, palabras, informe],
                )
            with gr.Tab("Cómo funciona"):
                gr.Markdown(COMO_FUNCIONA)
            with gr.Tab("Resultados del modelo"):
                gr.Markdown(_tabla_resultados())
            with gr.Tab("Límites y ética"):
                gr.Markdown(ETICA)
    return demo


def main() -> None:
    construir().launch(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS)


if __name__ == "__main__":
    main()
