"""Temperature scaling (S8): un único escalar T aprendido en validación, probs = softmax(logits / T)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def ajustar_temperatura(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Minimiza la entropía cruzada en validación respecto a T (L-BFGS sobre log T para que T > 0)."""
    logits = torch.tensor(logits_val, dtype=torch.float32)
    y = torch.tensor(y_val, dtype=torch.long)
    log_t = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def paso():
        opt.zero_grad()
        perdida = F.cross_entropy(logits / log_t.exp(), y)
        perdida.backward()
        return perdida

    opt.step(paso)
    return float(log_t.exp().item())
