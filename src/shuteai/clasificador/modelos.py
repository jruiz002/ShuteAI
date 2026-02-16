"""Los 4 clasificadores del plan. Todos devuelven logits [verdadera, falsa] (S1, S2, S3, S6)."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """S1 - Línea base: TF-IDF -> 2 capas densas con dropout."""

    def __init__(self, dim_entrada: int, ocultas=(512, 128), dropout: float = 0.3):
        super().__init__()
        a, b = ocultas
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, a),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(b, 2),
        )

    def forward(self, x):
        return self.red(x)


class CNN1D(nn.Module):
    """S2 - Embeddings -> filtros de 3, 4 y 5 palabras -> max-pooling -> capa densa."""

    def __init__(self, tam_vocab: int, dim_emb=128, filtros=100, ventanas=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(tam_vocab, dim_emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(dim_emb, filtros, k) for k in ventanas)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filtros * len(ventanas), 2)

    def forward(self, ids):
        e = self.emb(ids).transpose(1, 2)  # (B, E, T)
        p = [torch.relu(c(e)).amax(dim=2) for c in self.convs]  # max-pooling temporal
        return self.fc(self.drop(torch.cat(p, dim=1)))


class BiLSTM(nn.Module):
    """S3 - Embeddings -> BiLSTM de 2 capas -> (max + media) -> capa densa."""

    def __init__(self, tam_vocab: int, dim_emb=128, oculta=128, capas=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(tam_vocab, dim_emb, padding_idx=0)
        self.lstm = nn.LSTM(
            dim_emb,
            oculta,
            num_layers=capas,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if capas > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * oculta, 2)

    def forward(self, ids):
        mascara = (ids != 0).unsqueeze(-1)
        h, _ = self.lstm(self.drop(self.emb(ids)))
        maximo = h.masked_fill(~mascara, -1e9).amax(dim=1)
        media = (h * mascara).sum(1) / mascara.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([maximo, media], dim=1)))


class ClasificadorTransformer(nn.Module):
    """S5/S6/S9 - XLM-RoBERTa o BETO con fine-tuning y cabeza de clasificación."""

    def __init__(self, nombre: str, dropout: float = 0.1, congelar_embeddings: bool = True):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.nombre = nombre
        self.hf = AutoModelForSequenceClassification.from_pretrained(
            nombre, num_labels=2, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout
        )
        if congelar_embeddings:
            # La matriz de palabras (250k x 768) es ~70 % de XLM-R: congelarla ahorra ~2.5 GB de
            # gradientes/estado de AdamW y apenas afecta la calidad del fine-tuning.
            self.hf.base_model.embeddings.word_embeddings.weight.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        return self.hf(input_ids=input_ids, attention_mask=attention_mask).logits


def construir_modelo(tipo: str, *, codificador=None, cfg=None, nombre_hf: str | None = None):
    d = cfg.dropout if cfg is not None else 0.3
    if tipo == "mlp":
        return MLP(codificador.dim, dropout=d)
    if tipo == "cnn":
        return CNN1D(codificador.tam_vocab, dropout=d)
    if tipo == "bilstm":
        return BiLSTM(codificador.tam_vocab, dropout=d)
    if tipo == "transformer":
        return ClasificadorTransformer(nombre_hf)
    raise ValueError(f"Modelo desconocido: {tipo}")
