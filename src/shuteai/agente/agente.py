"""El agente de ShuteAI: orquesta los 3 pasos del plan (limpieza -> clasificador -> propagación).

La decisión de veracidad es SIEMPRE del clasificador entrenado (sin LLM, sin RAG, sin internet).
La propagación solo se ejecuta si P(falsa) >= umbral.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from shuteai.preprocesamiento.limpieza import preparar_prompt
from shuteai.propagacion.metricas import hora_mitad_alcance, hora_pico
from shuteai.utils import cargar_config

PASO_H = 0.5


class Clasificador(Protocol):
    def clasificar(self, texto: str): ...


class Propagador(Protocol):
    def muestras(self, cls, tokens, mascara) -> np.ndarray: ...


@dataclass
class Informe:
    texto: str
    idioma: str
    advertencias: list[str]
    p_verdadera: float
    p_falsa: float
    es_falsa: bool
    umbral: float
    palabras: list[str] = field(default_factory=list)
    pesos_atencion: list[float] = field(default_factory=list)
    # Solo si es probablemente falsa (curvas en escala log1p, 96 pasos de 30 min)
    curva: np.ndarray | None = None
    banda_baja: np.ndarray | None = None
    banda_alta: np.ndarray | None = None
    pico_h: tuple[float, float, float] | None = None  # (mediana, p25, p75)
    alcance: tuple[float, float, float] | None = None  # (mediana, p10, p90) compartidos
    t50_h: tuple[float, float, float] | None = None
    segundos: float = 0.0

    @property
    def veredicto(self) -> str:
        return "probablemente falsa" if self.es_falsa else "probablemente verdadera"

    @property
    def confianza(self) -> float:
        return self.p_falsa if self.es_falsa else self.p_verdadera


def _q(x: np.ndarray, qs) -> tuple[float, float, float]:
    a, b, c = np.percentile(x, qs)
    return float(a), float(b), float(c)


class Agente:
    def __init__(
        self,
        clasificador: Clasificador | None = None,
        propagador: Propagador | None = None,
        umbral: float | None = None,
    ):
        self._clasificador, self._propagador = clasificador, propagador
        self.umbral = cargar_config().clasificador.umbral_falsa if umbral is None else umbral

    # Carga perezosa: la interfaz abre rápido y los modelos se cargan en la primera consulta.
    @property
    def clasificador(self) -> Clasificador:
        if self._clasificador is None:
            from shuteai.clasificador.inferencia import ClasificadorFinal

            self._clasificador = ClasificadorFinal()
        return self._clasificador

    @property
    def propagador(self) -> Propagador:
        if self._propagador is None:
            from shuteai.agente.propagador import PropagadorFinal

            self._propagador = PropagadorFinal()
        return self._propagador

    def analizar(self, prompt: str, umbral: float | None = None) -> Informe:
        t0 = time.perf_counter()
        umbral = self.umbral if umbral is None else umbral
        noticia = preparar_prompt(prompt)  # Paso 1
        if not noticia.texto:
            raise ValueError("El texto está vacío. Escribe o pega una noticia.")
        r = self.clasificador.clasificar(noticia.texto)  # Paso 2
        informe = Informe(
            texto=noticia.texto,
            idioma=noticia.idioma,
            advertencias=list(noticia.advertencias),
            p_verdadera=r.p_verdadera,
            p_falsa=r.p_falsa,
            es_falsa=r.p_falsa >= umbral,
            umbral=umbral,
            palabras=r.palabras,
            pesos_atencion=r.pesos_atencion,
        )
        if informe.es_falsa:  # Paso 3
            m = self.propagador.muestras(r.cls, r.tokens, r.mascara)  # (mc, 96)
            m = np.maximum.accumulate(m, axis=1)
            informe.curva = np.median(m, axis=0)
            informe.banda_baja, informe.banda_alta = np.percentile(m, [10, 90], axis=0)
            informe.pico_h = _q(hora_pico(m), [50, 25, 75])
            informe.alcance = _q(np.expm1(m[:, -1]), [50, 10, 90])
            informe.t50_h = _q(hora_mitad_alcance(m), [50, 25, 75])
        informe.segundos = time.perf_counter() - t0
        return informe
