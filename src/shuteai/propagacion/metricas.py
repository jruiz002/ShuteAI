"""Métricas de las curvas de propagación. Las curvas viven en escala log1p del alcance acumulado."""

from __future__ import annotations

import numpy as np

PASO_H = 0.5  # cada paso = 30 min


def rmse(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def mae(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(real - pred)))


def mape(real_log: np.ndarray, pred_log: np.ndarray) -> float:
    """MAPE sobre el alcance en escala lineal (expm1); se ignoran puntos con alcance real 0."""
    r, p = np.expm1(real_log), np.expm1(pred_log)
    m = r > 0
    return float(np.mean(np.abs(r[m] - p[m]) / r[m]) * 100)


def hora_pico(curva_log: np.ndarray) -> np.ndarray:
    """Hora (fin del intervalo) con más compartidos nuevos. Acepta (T,) o (N, T)."""
    c = np.atleast_2d(np.expm1(curva_log))
    incrementos = np.diff(c, axis=1, prepend=0.0)
    return (incrementos.argmax(axis=1) + 1) * PASO_H


def alcance_final(curva_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.atleast_2d(curva_log)[:, -1])


def hora_mitad_alcance(curva_log: np.ndarray) -> np.ndarray:
    """t50: primera hora en que se alcanza el 50 % del alcance total de las 48 h."""
    c = np.atleast_2d(np.expm1(curva_log))
    mitad = c[:, -1:] * 0.5
    return ((c >= mitad).argmax(axis=1) + 1) * PASO_H


def resumen_modo_a(real: np.ndarray, pred: np.ndarray) -> dict:
    """RMSE/MAE de la curva (log) + error en hora del pico y en alcance final (en log y absoluto)."""
    return {
        "rmse": rmse(real, pred),
        "mae": mae(real, pred),
        "error_pico_h": float(np.mean(np.abs(hora_pico(real) - hora_pico(pred)))),
        "error_alcance_log": float(np.mean(np.abs(real[:, -1] - pred[:, -1]))),
        "error_alcance_abs": float(np.mean(np.abs(alcance_final(real) - alcance_final(pred)))),
    }


def resumen_modo_b(real_futuro: np.ndarray, pred_futuro: np.ndarray) -> dict:
    return {
        "rmse": rmse(real_futuro, pred_futuro),
        "mae": mae(real_futuro, pred_futuro),
        "mape": mape(real_futuro, pred_futuro),
    }


def errores_por_rango(real: np.ndarray, pred: np.ndarray) -> dict:
    """Mitigación de curvas muy variables: error reportado por rango de alcance real final."""
    alcance = alcance_final(real)
    rangos = {"<10": (0, 10), "10-50": (10, 50), "50-200": (50, 200), ">=200": (200, np.inf)}
    out = {}
    for nombre, (lo, hi) in rangos.items():
        m = (alcance >= lo) & (alcance < hi)
        if m.sum() > 0:
            out[nombre] = {
                "n": int(m.sum()),
                "rmse": rmse(real[m], pred[m]),
                "mae": mae(real[m], pred[m]),
            }
    return out
