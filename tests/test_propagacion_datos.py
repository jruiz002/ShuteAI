import numpy as np
import pytest

from shuteai.datos.particiones import particion_por_id
from shuteai.datos.propagacion import curva_desde_tiempos, tiempos_arbol


def test_curva_tiene_96_pasos_no_decreciente():
    c = curva_desde_tiempos([0, 10, 29, 31, 600, 2000])
    assert c.shape == (96,)
    assert np.all(np.diff(c) >= 0)
    assert c[0] == pytest.approx(np.log1p(3))  # 0, 10 y 29 min caen en el primer intervalo
    assert c[-1] == pytest.approx(np.log1p(6))


def test_tiempos_fuera_de_48h_no_cuentan():
    assert curva_desde_tiempos([0, 5000])[-1] == pytest.approx(np.log1p(1))


def test_lee_arbol(tmp_path):
    f = tmp_path / "1.txt"
    f.write_text(
        "['ROOT', 'ROOT', '0.0']->['10', '1', '0.0']\n"
        "['10', '1', '0.0']->['20', '2', '45.5']\n"
        "['10', '1', '0.0']->['20', '2', '45.5']\n"  # duplicado: cuenta una vez
    )
    assert sorted(tiempos_arbol(f)) == [0.0, 45.5]


def test_particion_es_estable_y_cubre_todo():
    assert particion_por_id("123") == particion_por_id("123")
    cuenta = {p: 0 for p in ("train", "val", "test")}
    for i in range(5000):
        cuenta[particion_por_id(i)] += 1
    assert 0.65 < cuenta["train"] / 5000 < 0.75
