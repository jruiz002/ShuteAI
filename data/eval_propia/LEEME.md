# Conjunto de prueba propio (50 noticias) — NUNCA se usa para entrenar

Completa `noticias_prueba.csv` con **25 noticias verificadas como falsas** (por un medio de verificación:
Chequeado, Maldita, AFP Factual, Ecuador Chequea, etc.) y **25 verdaderas** (de medios formales), recientes y en español.

| columna | contenido |
|---|---|
| `texto` | La noticia redactada por el equipo (2–6 oraciones, sin nombre del medio ni URL) |
| `etiqueta` | `falsa` o `verdadera` |
| `medio_o_verificador` | Quién la verificó / publicó |
| `url_verificacion` | Enlace a la verificación (obligatorio: es la evidencia de la etiqueta) |
| `fecha` | AAAA-MM-DD |

Valida el archivo con `python -m shuteai.datos.eval_propia`.
