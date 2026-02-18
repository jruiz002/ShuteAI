import numpy as np
import pytest

from shuteai.clasificador.inferencia import _agrupar_palabras
from shuteai.propagacion.incertidumbre import curvas_desde_residuos


def test_curvas_con_residuos_son_validas():
    rng = np.random.default_rng(1)
    pred = np.log1p(np.cumsum(np.full(96, 2.0)))
    residuos = rng.normal(0, 1.0, (50, 96)).astype(np.float32)
    c = curvas_desde_residuos(pred, residuos, 100, rng)
    assert c.shape == (100, 96)
    assert np.all(c >= 0) and np.all(np.diff(c, axis=1) >= 0)
    assert c[:, -1].std() > 0.1  # la banda no es degenerada


def test_agrupa_subtokens_en_palabras():
    texto = "Hola mundo"
    # offsets: <s>, "Ho", "la", "mundo", </s>
    offsets = [(0, 0), (0, 2), (2, 4), (5, 10), (0, 0)]
    palabras, pesos = _agrupar_palabras(texto, offsets, np.array([0.1, 0.2, 0.3, 0.9, 0.1]))
    assert palabras == ["Hola", "mundo"]
    assert len(pesos) == 2 and max(pesos) == pytest.approx(1.0) and min(pesos) == pytest.approx(0.0)
