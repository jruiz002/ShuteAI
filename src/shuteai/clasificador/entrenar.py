"""Entrenamiento de los 4 clasificadores (S1, S2, S3, S6) con las prácticas de S8.

Prácticas: entropía cruzada ponderada, dropout, AdamW, warmup + cosine, recorte de gradiente,
early stopping sobre F1 macro de validación. Artefactos en models/clasificador/<nombre>/.

Ejemplos:
    python -m shuteai.clasificador.entrenar --modelo mlp
    python -m shuteai.clasificador.entrenar --modelo transformer --receta en_es --nombre xlmr_en_es
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from shuteai.clasificador.codificadores import (
    CodificadorTfidf,
    CodificadorTransformer,
    CodificadorVocab,
)
from shuteai.clasificador.modelos import construir_modelo
from shuteai.datos.textos import SALIDA
from shuteai.rutas import MODELOS
from shuteai.utils import cargar_config, dispositivo, fijar_semilla

DIR_CLASIFICADOR = MODELOS / "clasificador"


def _a_dispositivo(batch: dict, dev) -> dict:
    return {k: v.to(dev) for k, v in batch.items()}


@torch.no_grad()
def predecir_logits(modelo, codificador, textos: list[str], dev, batch: int = 64) -> np.ndarray:
    modelo.eval()
    # Orden por longitud: menos relleno y más velocidad; se restaura al final.
    orden = np.argsort([len(t) for t in textos])
    salida = np.zeros((len(textos), 2), dtype=np.float32)
    for i in range(0, len(textos), batch):
        idx = orden[i : i + batch]
        b = _a_dispositivo(codificador.colate([textos[j] for j in idx]), dev)
        salida[idx] = modelo(**b).float().cpu().numpy()
    return salida


def entrenar(
    modelo,
    codificador,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    epocas: int,
    lr: float,
    batch: int,
    dev,
    pesos_clase: bool = True,
    scheduler: str = "cosine",
    warmup: float = 0.1,
    weight_decay: float = 0.01,
    paciencia: int = 3,
    verbose: bool = True,
    acum: int = 1,
) -> dict:
    """Bucle de entrenamiento genérico. Deja en `modelo` los mejores pesos de validación."""
    modelo.to(dev)
    textos = train["texto"].tolist()
    y = torch.tensor(train["etiqueta"].values, dtype=torch.long)
    if pesos_clase:
        frec = torch.bincount(y, minlength=2).float()
        w = (frec.sum() / (2 * frec)).to(dev)
    else:
        w = None

    entrenables = {n for n, p in modelo.named_parameters() if p.requires_grad}
    sin_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm")
    grupos = [
        {
            "params": [
                p
                for n, p in modelo.named_parameters()
                if p.requires_grad and not any(s in n for s in sin_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in modelo.named_parameters()
                if p.requires_grad and any(s in n for s in sin_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(grupos, lr=lr)
    pasos = epocas * int(np.ceil(len(textos) / (batch * acum)))
    n_warm = int(warmup * pasos)
    sched = (
        get_cosine_schedule_with_warmup(opt, n_warm, pasos)
        if scheduler == "cosine"
        else get_constant_schedule_with_warmup(opt, 0)
    )

    mejor, mejor_estado, sin_mejora, hist = -1.0, None, 0, []
    for ep in range(1, epocas + 1):
        modelo.train()
        perm = np.random.permutation(len(textos))
        total, t0 = 0.0, time.time()
        barra = tqdm(
            range(0, len(perm), batch),
            desc=f"época {ep}/{epocas}",
            disable=not (verbose and sys.stderr.isatty()),
        )
        for k, i in enumerate(barra):
            idx = perm[i : i + batch]
            b = _a_dispositivo(codificador.colate([textos[j] for j in idx]), dev)
            logits = modelo(**b)
            perdida = F.cross_entropy(logits, y[idx].to(dev), weight=w)
            (perdida / acum).backward()
            if (k + 1) % acum == 0 or i + batch >= len(perm):
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            total += perdida.item() * len(idx)
            barra.set_postfix(loss=f"{perdida.item():.3f}")
            if verbose and (k + 1) % 200 == 0:
                print(f"    paso {k + 1}: loss media {total / (i + len(idx)):.4f}", flush=True)
        lv = predecir_logits(modelo, codificador, val["texto"].tolist(), dev)
        f1 = f1_score(val["etiqueta"], lv.argmax(1), average="macro")
        val_loss = F.cross_entropy(torch.tensor(lv), torch.tensor(val["etiqueta"].values)).item()
        hist.append(
            {
                "epoca": ep,
                "loss_train": total / len(textos),
                "loss_val": val_loss,
                "f1_val": float(f1),
                "seg": round(time.time() - t0, 1),
            }
        )
        if verbose:
            print(f"  época {ep}: {hist[-1]}", flush=True)
        if f1 > mejor:
            mejor, sin_mejora = f1, 0
            # Solo se copian los parámetros entrenables (ahorra RAM con embeddings congelados).
            mejor_estado = {
                k: v.detach().cpu().clone()
                for k, v in modelo.state_dict().items()
                if k in entrenables
            }
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                if verbose:
                    print("  early stopping")
                break
    modelo.load_state_dict(mejor_estado, strict=False)
    return {"historial": hist, "mejor_f1_val": float(mejor)}


def _dividir(df: pd.DataFrame, receta: str):
    train, val = df[df.particion == "train"], df[df.particion == "val"]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo", required=True, choices=["mlp", "cnn", "bilstm", "transformer"])
    ap.add_argument("--nombre", help="Carpeta de salida (por defecto = --modelo)")
    ap.add_argument(
        "--receta",
        default="mezcla",
        choices=["mezcla", "en_es", "solo_es"],
        help="mezcla: todo junto | en_es: preentrena en inglés y ajusta en español (S9) "
        "| solo_es: únicamente español",
    )
    ap.add_argument("--base", help="Checkpoint HF (por defecto el de configs/default.yaml)")
    ap.add_argument("--sin-pesos", action="store_true", help="Ablación: sin pesos de clase")
    ap.add_argument("--dropout", type=float, help="Ablación: sobrescribe el dropout")
    ap.add_argument(
        "--lr-constante", action="store_true", help="Ablación: lr fijo, sin warmup+cosine"
    )
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--lr", type=float, help="Sobrescribe el learning rate")
    ap.add_argument("--muestra", type=int, help="Usa solo N filas de train (prueba rápida)")
    args = ap.parse_args()

    cfg = cargar_config().clasificador
    if args.dropout is not None:
        cfg.dropout = args.dropout
    fijar_semilla()
    dev = dispositivo()
    nombre = args.nombre or args.modelo
    salida = DIR_CLASIFICADOR / nombre
    salida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SALIDA)
    es_tr = args.modelo == "transformer"
    base = args.base or cfg.modelo_transformer
    print(
        f"Dispositivo: {dev} | modelo: {args.modelo} ({base if es_tr else 'desde cero'}) | "
        f"receta: {args.receta}"
    )

    if args.receta == "solo_es":
        df = df[df.idioma == "es"]
    train_all, val_all = df[df.particion == "train"], df[df.particion == "val"]
    if args.muestra:
        train_all = train_all.sample(min(args.muestra, len(train_all)), random_state=0)

    if es_tr:
        cod = CodificadorTransformer(base, cfg.max_len_largo)
    elif args.modelo == "mlp":
        cod = CodificadorTfidf(cfg.tfidf_terminos).ajustar(train_all["texto"].tolist())
    else:
        cod = CodificadorVocab(max_len=cfg.max_len_largo).ajustar(train_all["texto"].tolist())
    modelo = construir_modelo(args.modelo, codificador=cod, cfg=cfg, nombre_hf=base)

    kw = dict(
        batch=cfg.batch if es_tr else 64,
        dev=dev,
        pesos_clase=not args.sin_pesos,
        scheduler="constante" if args.lr_constante else "cosine",
        warmup=cfg.warmup_frac,
        weight_decay=cfg.weight_decay,
        paciencia=cfg.paciencia,
        acum=cfg.acumulacion if es_tr else 1,
    )
    lr = args.lr or (cfg.lr_transformer if es_tr else cfg.lr_base)
    epocas = args.epocas or (cfg.epocas_transformer if es_tr else cfg.epocas_base)

    info = {
        "modelo": args.modelo,
        "base": base if es_tr else None,
        "receta": args.receta,
        "args": vars(args),
        "etapas": [],
    }
    if args.receta == "en_es":
        en_tr, en_val = train_all[train_all.idioma == "en"], val_all[val_all.idioma == "en"]
        print(f"== Etapa 1: inglés ({len(en_tr)} ejemplos)")
        info["etapas"].append(
            {"etapa": "en", **entrenar(modelo, cod, en_tr, en_val, epocas=epocas, lr=lr, **kw)}
        )
        es_tr_df, es_val = train_all[train_all.idioma == "es"], val_all[val_all.idioma == "es"]
        print(f"== Etapa 2: español ({len(es_tr_df)} ejemplos), lr reducido")
        info["etapas"].append(
            {
                "etapa": "es",
                **entrenar(modelo, cod, es_tr_df, es_val, epocas=epocas, lr=lr / 2, **kw),
            }
        )
    else:
        info["etapas"].append(
            {
                "etapa": "unica",
                **entrenar(modelo, cod, train_all, val_all, epocas=epocas, lr=lr, **kw),
            }
        )

    # Logits de validación/prueba para calibración y evaluación (se evalúan TODOS los idiomas).
    for part in ("val", "test"):
        sub = df[df.particion == part].copy()
        lg = predecir_logits(modelo, cod, sub["texto"].tolist(), dev)
        sub["logit_verdadera"], sub["logit_falsa"] = lg[:, 0], lg[:, 1]
        sub.drop(columns="texto").to_csv(salida / f"logits_{part}.csv", index=False)

    torch.save({k: v for k, v in modelo.state_dict().items()}, salida / "pesos.pt")
    if not es_tr:
        with open(salida / "codificador.pkl", "wb") as f:
            pickle.dump(cod, f)
    (salida / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
