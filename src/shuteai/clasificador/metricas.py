"""Métricas del clasificador y de calibración (S8)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metricas_clasificacion(y: np.ndarray, probs: np.ndarray) -> dict:
    """probs: (N, 2) = [P(verdadera), P(falsa)]."""
    pred = probs.argmax(axis=1)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    out["auc_roc"] = float(roc_auc_score(y, probs[:, 1])) if len(np.unique(y)) == 2 else None
    return out


def ece(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error sobre la confianza de la clase predicha."""
    conf = probs.max(axis=1)
    acierto = (probs.argmax(axis=1) == y).astype(float)
    bordes = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(bordes[:-1], bordes[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs[:, 1] - y) ** 2))


def curva_confiabilidad(y: np.ndarray, probs: np.ndarray, bins: int = 10):
    """Devuelve (confianza_media, exactitud, conteo) por intervalo para el diagrama de confiabilidad."""
    conf = probs.max(axis=1)
    acierto = (probs.argmax(axis=1) == y).astype(float)
    bordes = np.linspace(0.5, 1.0, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bordes[:-1], bordes[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(acierto[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
