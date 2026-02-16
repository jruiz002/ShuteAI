import numpy as np

from shuteai.clasificador.calibracion import ajustar_temperatura
from shuteai.clasificador.metricas import brier, ece, softmax
from shuteai.propagacion.metricas import hora_mitad_alcance, hora_pico, rmse


def test_temperature_scaling_reduce_ece_en_modelo_sobreconfiado():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    # Logits muy grandes pero que aciertan solo ~75 % de las veces: sobreconfianza.
    signo = np.where(rng.random(4000) < 0.75, 1, -1) * np.where(y == 1, 1, -1)
    logits = np.stack([-signo * 6.0, signo * 6.0], axis=1)
    T = ajustar_temperatura(logits, y)
    assert T > 1.5
    assert ece(y, softmax(logits, T)) < ece(y, softmax(logits)) / 2


def test_brier_perfecto_es_cero():
    y = np.array([0, 1, 1])
    assert brier(y, np.array([[1, 0], [0, 1], [0, 1]], dtype=float)) == 0


def test_pico_y_t50():
    # 10 compartidos en el paso 4 (hora 2.0) y nada más.
    c = np.log1p(np.concatenate([np.zeros(3), np.full(93, 10.0)]))
    assert hora_pico(c)[0] == 2.0
    assert hora_mitad_alcance(c)[0] == 2.0


def test_rmse():
    assert rmse(np.zeros(4), np.full(4, 2.0)) == 2.0
