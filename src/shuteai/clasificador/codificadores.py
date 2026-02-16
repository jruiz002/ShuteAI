"""Codificadores de texto -> tensores, uno por familia de modelo. Todos exponen `colate(textos)`."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN = re.compile(r"[a-záéíóúñü0-9#@']+")
PAD, UNK = 0, 1


def tokenizar_palabras(texto: str) -> list[str]:
    return _TOKEN.findall(texto.lower())


class CodificadorTfidf:
    """TF-IDF de 5,000 términos (uni+bigramas) para el MLP de la línea base (S1)."""

    def __init__(self, terminos: int):
        self.vec = TfidfVectorizer(
            max_features=terminos,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            tokenizer=tokenizar_palabras,
            token_pattern=None,
            lowercase=False,
        )

    @property
    def dim(self) -> int:
        return len(self.vec.vocabulary_)

    def ajustar(self, textos: list[str]) -> CodificadorTfidf:
        self.vec.fit(textos)
        return self

    def colate(self, textos: list[str]) -> dict[str, torch.Tensor]:
        x = self.vec.transform(textos).toarray().astype(np.float32)
        return {"x": torch.from_numpy(x)}


class CodificadorVocab:
    """Vocabulario de palabras -> ids, para CNN 1D y BiLSTM entrenados desde cero."""

    def __init__(self, max_vocab: int = 30_000, min_freq: int = 2, max_len: int = 256):
        self.max_vocab, self.min_freq, self.max_len = max_vocab, min_freq, max_len
        self.vocab: dict[str, int] = {}

    @property
    def tam_vocab(self) -> int:
        return len(self.vocab) + 2

    def ajustar(self, textos: list[str]) -> CodificadorVocab:
        cuenta = Counter(p for t in textos for p in tokenizar_palabras(t))
        comunes = [p for p, c in cuenta.most_common(self.max_vocab) if c >= self.min_freq]
        self.vocab = {p: i + 2 for i, p in enumerate(comunes)}
        return self

    def colate(self, textos: list[str]) -> dict[str, torch.Tensor]:
        ids = [
            [self.vocab.get(p, UNK) for p in tokenizar_palabras(t)[: self.max_len]] or [UNK]
            for t in textos
        ]
        largo = max(len(s) for s in ids)
        # El ancho mínimo (5) evita que los filtros de 5 palabras de la CNN fallen con textos cortos.
        largo = max(largo, 5)
        x = torch.full((len(ids), largo), PAD, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
        return {"ids": x}


class CodificadorTransformer:
    """Tokenizador de HuggingFace con padding dinámico."""

    def __init__(self, nombre_modelo: str, max_len: int = 256):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(nombre_modelo)
        self.max_len = max_len

    def colate(self, textos: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tok(
            textos, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
