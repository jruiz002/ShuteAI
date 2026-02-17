"""Baselines de propagación (S12): curva promedio (modo A) y modelo SIR (modo B)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

PASO_H = 0.5


class CurvaPromedio:
    """Baseline 1: la curva promedio de las noticias falsas del entrenamiento (ignora el texto)."""

    def ajustar(self, curvas_train: np.ndarray) -> CurvaPromedio:
        self.curva = curvas_train.mean(axis=0)
        return self

    def predecir(self, n: int) -> np.ndarray:
        return np.tile(self.curva, (n, 1))

    def predecir_futuro(self, contexto: np.ndarray, horizonte: int) -> np.ndarray:
        """Modo B: la forma promedio de crecimiento, anclada al último valor observado."""
        t0 = contexto.shape[1]
        crecimiento = self.curva[t0 : t0 + horizonte] - self.curva[t0 - 1]
        return contexto[:, -1:] + np.maximum(crecimiento, 0)[None, :]


def _sir_alcance(params: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Alcance acumulado = N - S(t) para un SIR con I(0)=1. params = (beta, gamma, log10 N)."""
    beta, gamma, log_n = params
    n = 10.0**log_n

    def f(_, y):
        s, i = y
        nuevos = beta * s * i / n
        return [-nuevos, nuevos - gamma * i]

    sol = solve_ivp(f, (0, t_eval[-1]), [n - 1.0, 1.0], t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if sol.y.shape[1] != len(t_eval):
        return np.full(len(t_eval), n)
    return n - sol.y[0] + 1.0  # +1: la fuente original cuenta como alcance


class ModeloSIR:
    """Baseline 2 (modo B): SIR ajustado por mínimos cuadrados a las primeras horas de cada curva."""

    def __init__(self, beta_max: float = 20.0, gamma_max: float = 5.0, log_n_max: float = 3.5):
        # log10(N) <= 3.5 (~3,000 compartidos): el máximo observado en Twitter15/16 es ~380.
        self.beta_max, self.gamma_max, self.log_n_max = beta_max, gamma_max, log_n_max

    def _ajustar_uno(self, obs_log: np.ndarray) -> np.ndarray:
        t_obs = (np.arange(len(obs_log)) + 1) * PASO_H
        piso = min(np.log10(max(np.expm1(obs_log[-1]), 2.0)), self.log_n_max - 0.01)

        def residuo(p):
            return np.log1p(_sir_alcance(p, t_obs)) - obs_log

        mejor, mejor_c = None, np.inf
        for x0 in ([1.0, 0.3, piso + 0.5], [3.0, 0.8, piso + 1.5], [0.5, 0.1, piso + 1.0]):
            try:
                r = least_squares(
                    residuo,
                    x0,
                    bounds=([1e-3, 1e-3, piso], [self.beta_max, self.gamma_max, self.log_n_max]),
                    max_nfev=60,
                )
            except Exception:
                continue
            if r.cost < mejor_c:
                mejor, mejor_c = r.x, r.cost
        return mejor if mejor is not None else np.array([1.0, 0.5, piso + 1.0])

    def predecir_futuro(self, contexto: np.ndarray, horizonte: int) -> np.ndarray:
        t0 = contexto.shape[1]
        t_all = (np.arange(t0 + horizonte) + 1) * PASO_H
        salida = np.zeros((len(contexto), horizonte), dtype=np.float32)
        for k, obs in enumerate(contexto):
            p = self._ajustar_uno(obs)
            fut = np.log1p(_sir_alcance(p, t_all))[t0:]
            salida[k] = np.maximum.accumulate(np.maximum(fut, obs[-1]))
        return salida
