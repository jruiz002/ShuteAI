"""Paso 1 del agente: preparación del prompt (reglas y expresiones regulares, sin modelo)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff"
    "\U0000fe00-\U0000fe0f\U0000200d\U00002b00-\U00002bff]+"
)
_ESPACIOS = re.compile(r"\s+")
# Firmas típicas de agencias/medios: "(Reuters) -", "MADRID, 3 may (EFE).-", "Por Juan Pérez"
_FIRMA_AGENCIA = re.compile(
    r"^\s*[A-ZÁÉÍÓÚÑ ,]{3,40}[,\-–—]\s*\d{1,2}\s+\w+\s*\(\w+\)\s*[.\-–—:]*\s*"
)
_PAREN_AGENCIA = re.compile(r"\((?:Reuters|AP|AFP|EFE|Europa Press|UPI|Notimex)\)\s*[-–—:]?", re.I)
_POR_AUTOR = re.compile(
    r"^\s*(?:Por|By)\s+[A-ZÁÉÍÓÚÑ][\w.\-]+(?:\s+[A-ZÁÉÍÓÚÑ][\w.\-]+){0,3}\s*[\n\-–—:]", re.M
)

# "¿Es falsa esta noticia?: ...", "Verifica esto: ...", "Is this true? ..."
_PREGUNTA = re.compile(
    r"^\s*[¿¡]?\s*(?:"
    r"(?:es|será|sera)\s+(?:falsa|verdadera|cierta|real|fake)\s+(?:esta|la)\s+(?:noticia|nota|información)"
    r"|(?:esto|esta\s+noticia)\s+(?:es|será)\s+(?:falsa|verdadera|cierta|real)"
    r"|(?:verifica|verifícame|revisa|analiza|chequea)\s+(?:esta\s+noticia|la\s+noticia|esto|esta)"
    r"|is\s+this\s+(?:news\s+)?(?:true|real|fake|false)"
    r")\s*[?!]*\s*[:\-–—]?\s*",
    re.IGNORECASE,
)
_COMILLAS = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'"})

MIN_PALABRAS = 5


@dataclass(frozen=True)
class NoticiaPreparada:
    texto: str
    idioma: str  # "es", "en" u "otro"/"desconocido"
    advertencias: tuple[str, ...] = ()


def limpiar_texto(texto: str) -> str:
    """Quita URLs, emojis y espacios sobrantes. Idéntico en entrenamiento e inferencia."""
    texto = unicodedata.normalize("NFC", texto).translate(_COMILLAS)
    texto = _URL.sub(" ", texto)
    texto = _EMOJI.sub(" ", texto)
    return _ESPACIOS.sub(" ", texto).strip()


def quitar_firmas(texto: str, fuente: str | None = None) -> str:
    """Mitigación del riesgo "el modelo aprende el medio": elimina agencia, autor y nombre del medio."""
    texto = _PAREN_AGENCIA.sub(" ", texto)
    texto = _FIRMA_AGENCIA.sub("", texto)
    texto = _POR_AUTOR.sub("", texto)
    if fuente and isinstance(fuente, str) and len(fuente.strip()) >= 3:
        texto = re.sub(re.escape(fuente.strip()), " ", texto, flags=re.IGNORECASE)
    return _ESPACIOS.sub(" ", texto).strip()


def extraer_noticia(prompt: str) -> str:
    """Si el usuario escribió '¿es falsa esta noticia?: ...' devuelve solo la noticia."""
    prompt = prompt.strip()
    sin_pregunta = _PREGUNTA.sub("", prompt, count=1)
    candidato = sin_pregunta if sin_pregunta.strip() else prompt
    if len(candidato) >= 2 and candidato[0] == candidato[-1] == '"':
        candidato = candidato[1:-1]
    return limpiar_texto(candidato)


def detectar_idioma(texto: str) -> str:
    """Detección ligera de idioma. Usa langdetect si está disponible; si no, heurística de stopwords."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        codigo = detect(texto)
        return codigo if codigo in {"es", "en"} else "otro"
    except Exception:
        palabras = set(re.findall(r"[a-záéíóúñü]+", texto.lower()))
        es = len(
            palabras & {"el", "la", "de", "que", "en", "los", "se", "del", "las", "por", "con"}
        )
        en = len(palabras & {"the", "of", "and", "to", "in", "is", "that", "for", "with", "on"})
        if es == en == 0:
            return "desconocido"
        return "es" if es >= en else "en"


def preparar_prompt(prompt: str) -> NoticiaPreparada:
    """Paso 1 completo: extracción + limpieza + idioma + advertencias de calidad de entrada."""
    texto = extraer_noticia(prompt)
    advertencias: list[str] = []
    if len(texto.split()) < MIN_PALABRAS:
        advertencias.append(
            "El texto es muy corto: las probabilidades serán poco fiables. Pega la noticia completa."
        )
    idioma = detectar_idioma(texto) if texto else "desconocido"
    if idioma not in {"es", "en"}:
        advertencias.append(
            "No se detectó español ni inglés; el modelo fue entrenado con ambos idiomas."
        )
    return NoticiaPreparada(texto=texto, idioma=idioma, advertencias=tuple(advertencias))
