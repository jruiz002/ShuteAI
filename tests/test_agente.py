from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shuteai.agente.agente import Agente
from shuteai.agente.informe import figura_propagacion, redactar


class ClasificadorFalso:
    def __init__(self, p_falsa):
        self.p = p_falsa

    def clasificar(self, texto):
        return SimpleNamespace(
            p_verdadera=1 - self.p,
            p_falsa=self.p,
            palabras=texto.split(),
            pesos_atencion=[0.5] * len(texto.split()),
            cls=torch.zeros(1, 4),
            tokens=torch.zeros(1, 3, 4),
            mascara=torch.ones(1, 3),
        )


class PropagadorFalso:
    llamadas = 0

    def muestras(self, cls, tokens, mascara):
        PropagadorFalso.llamadas += 1
        base = np.log1p(np.cumsum(np.r_[np.zeros(4), np.full(10, 20.0), np.full(82, 1.0)]))
        return np.stack([base * f for f in np.linspace(0.8, 1.2, 30)])


TEXTO = "URGENTE el gobierno anunció que mañana cortarán el agua durante 15 días compartan"


def test_noticia_falsa_ejecuta_propagacion():
    PropagadorFalso.llamadas = 0
    inf = Agente(ClasificadorFalso(0.88), PropagadorFalso()).analizar(
        "¿Es falsa esta noticia?: " + TEXTO
    )
    assert inf.es_falsa and inf.curva is not None and PropagadorFalso.llamadas == 1
    assert inf.texto == TEXTO
    assert inf.pico_h[1] <= inf.pico_h[0] <= inf.pico_h[2]
    assert "probablemente falsa" in redactar(inf) and "no verifica hechos" in redactar(inf)
    assert len(figura_propagacion(inf).data) == 3


def test_noticia_verdadera_no_ejecuta_propagacion():
    PropagadorFalso.llamadas = 0
    inf = Agente(ClasificadorFalso(0.10), PropagadorFalso()).analizar(TEXTO)
    assert not inf.es_falsa and inf.curva is None and PropagadorFalso.llamadas == 0
    assert "probablemente verdadera" in redactar(inf)


def test_umbral_configurable():
    ag = Agente(ClasificadorFalso(0.6), PropagadorFalso())
    assert ag.analizar(TEXTO, umbral=0.5).es_falsa
    assert not ag.analizar(TEXTO, umbral=0.7).es_falsa


def test_texto_vacio_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="vacío"):
        Agente(ClasificadorFalso(0.5), PropagadorFalso()).analizar("   ")
