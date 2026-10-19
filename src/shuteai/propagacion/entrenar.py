"""Entrenamiento de los modelos de propagación (modo A: texto -> 96 pasos; modo B: 6 h -> 42 h).

    python -m shuteai.propagacion.entrenar --modo a --modelo lstm
    python -m shuteai.propagacion.entrenar --modo b --modelo patchtst
Artefactos en models/propagacion/<modo>_<modelo>/.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from shuteai.propagacion.modelos import (
    MODELOS_MODO_A,
    ForecasterLSTM,
    PatchTransformer,
)
from shuteai.rutas import MODELOS, PROCESADOS
from shuteai.utils import cargar_config, dispositivo, fijar_semilla

DIR_PROP = MODELOS / "propagacion"


def cargar_embeddings(nombre: str = "twitter") -> dict:
    return torch.load(PROCESADOS / f"emb_{nombre}.pt", weights_only=False)


def particiones(datos: dict) -> dict[str, np.ndarray]:
    p = datos["meta"]["particion"].values
    return {k: np.where(p == k)[0] for k in ("train", "val", "test")}


def construir_modo_a(tipo: str, cfg, d_txt: int = 768):
    kw = {"d_txt": d_txt, "dropout": cfg.dropout}
    if tipo != "mlp":
        kw |= {"h": cfg.hidden, "heads": cfg.heads}
    return MODELOS_MODO_A[tipo](**kw)


def construir_modo_b(tipo: str, cfg, contexto: int, horizonte: int):
    if tipo == "lstm":
        return ForecasterLSTM(contexto, horizonte, dropout=cfg.dropout)
    if tipo == "transformer":  # Transformer "punto a punto"
        return PatchTransformer(contexto, horizonte, patch_len=1, stride=1, dropout=cfg.dropout)
    if tipo == "patchtst":
        return PatchTransformer(contexto, horizonte, patch_len=4, stride=2, dropout=cfg.dropout)
    raise ValueError(tipo)


@torch.no_grad()
def predecir_modo_a(modelo, datos, idx, dev, batch=64, mc: int = 0):
    """mc=0: predicción determinista. mc>0: (N, mc, 96) muestras con MC-Dropout."""
    salida = []
    for i in range(0, len(idx), batch):
        j = idx[i : i + batch]
        args = (
            datos["cls"][j].to(dev),
            datos["tokens"][j].float().to(dev),
            datos["mascara"][j].to(dev),
        )
        if mc == 0:
            modelo.eval()
            salida.append(modelo(*args).cpu())
        else:
            modelo.train()  # dropout activo = incertidumbre epistémica aproximada
            salida.append(torch.stack([modelo(*args).cpu() for _ in range(mc)], dim=1))
    modelo.eval()
    return torch.cat(salida).numpy()


def entrenar_modo_a(tipo, cfg, dev, nombre):
    fijar_semilla()
    datos = cargar_embeddings("twitter")
    idx = particiones(datos)
    modelo = construir_modo_a(tipo, cfg).to(dev)
    y = datos["curvas"]
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=0.01)
    pasos = cfg.epocas * int(np.ceil(len(idx["train"]) / cfg.batch))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * pasos), pasos)
    mejor, mejor_estado, sin_mejora, hist = np.inf, None, 0, []
    for ep in range(1, cfg.epocas + 1):
        modelo.train()
        perm = np.random.permutation(idx["train"])
        p_real = max(0.0, 0.7 * (1 - ep / (0.6 * cfg.epocas))) if tipo == "lstm" else 0.0
        tot = 0.0
        for i in tqdm(range(0, len(perm), cfg.batch), disable=not sys.stderr.isatty()):
            j = perm[i : i + cfg.batch]
            yb = y[j].to(dev)
            pred = modelo(
                datos["cls"][j].to(dev),
                datos["tokens"][j].float().to(dev),
                datos["mascara"][j].to(dev),
                y_real=yb,
                p_real=p_real,
            )
            perdida = F.huber_loss(pred, yb, delta=cfg.huber_delta)
            opt.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += perdida.item() * len(j)
        pv = predecir_modo_a(modelo, datos, idx["val"], dev)
        rmse_val = float(np.sqrt(np.mean((pv - y[idx["val"]].numpy()) ** 2)))
        hist.append({"epoca": ep, "loss_train": tot / len(perm), "rmse_val": rmse_val})
        print(hist[-1], flush=True)
        if rmse_val < mejor:
            mejor, sin_mejora = rmse_val, 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= cfg.paciencia:
                print("early stopping")
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, {"historial": hist, "mejor_rmse_val": mejor, "tipo": tipo, "modo": "a"}


def entrenar_modo_b(tipo, cfg, dev, nombre):
    fijar_semilla()
    cp = cargar_config().propagacion_datos
    contexto = int(cp.horas_contexto_modo_b * 60 / cp.paso_min)
    datos = cargar_embeddings("twitter")
    idx = particiones(datos)
    y = datos["curvas"]
    horizonte = y.shape[1] - contexto
    modelo = construir_modo_b(tipo, cfg, contexto, horizonte).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr * 0.5, weight_decay=0.01)
    pasos = cfg.epocas * int(np.ceil(len(idx["train"]) / cfg.batch))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * pasos), pasos)
    mejor, mejor_estado, sin_mejora, hist = np.inf, None, 0, []
    for ep in range(1, cfg.epocas + 1):
        modelo.train()
        perm = np.random.permutation(idx["train"])
        tot = 0.0
        for i in range(0, len(perm), cfg.batch):
            j = perm[i : i + cfg.batch]
            ctx, fut = y[j, :contexto].to(dev), y[j, contexto:].to(dev)
            perdida = F.huber_loss(modelo(ctx), fut, delta=cfg.huber_delta)
            opt.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += perdida.item() * len(j)
        modelo.eval()
        with torch.no_grad():
            pv = modelo(y[idx["val"], :contexto].to(dev)).cpu()
        rmse_val = float(torch.sqrt(F.mse_loss(pv, y[idx["val"], contexto:])))
        hist.append({"epoca": ep, "loss_train": tot / len(perm), "rmse_val": rmse_val})
        if ep % 10 == 0 or ep == 1:
            print(hist[-1], flush=True)
        if rmse_val < mejor:
            mejor, sin_mejora = rmse_val, 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= cfg.paciencia:
                print("early stopping")
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, {
        "historial": hist,
        "mejor_rmse_val": mejor,
        "tipo": tipo,
        "modo": "b",
        "contexto": contexto,
        "horizonte": horizonte,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", required=True, choices=["a", "b"])
    ap.add_argument(
        "--modelo",
        required=True,
        help="modo a: mlp|lstm|transformer ; modo b: lstm|transformer|patchtst",
    )
    args = ap.parse_args()
    cfg, dev = cargar_config().propagacion, dispositivo()
    nombre = f"{args.modo}_{args.modelo}"
    print(f"Entrenando {nombre} en {dev}")
    f = entrenar_modo_a if args.modo == "a" else entrenar_modo_b
    modelo, info = f(args.modelo, cfg, dev, nombre)
    salida = DIR_PROP / nombre
    salida.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), salida / "pesos.pt")
    (salida / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
