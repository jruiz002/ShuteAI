# 🛡️ ShuteAI

**Agente que detecta noticias probablemente falsas y pronostica cómo se propagarían en redes sociales durante 48 horas.**
Proyecto integrador de Deep Learning (UVG, 2026). Antes llamado *VeritasAI*; el plan original está en `VeritasIA_Plan_de_Ejecucion.pdf`.

> **Qué hace, en una frase:** pegas una noticia → un Transformer estima `P(verdadera)` y `P(falsa)` calibradas →
> si `P(falsa)` supera un umbral, un modelo encoder-decoder LSTM con atención pronostica la curva de
> compartidos a 48 h (pico, tiempo al 50 % del alcance, alcance total, con banda de incertidumbre).
> **Sin RAG, sin internet, sin imágenes y sin LLM:** todo el conocimiento está en los pesos entrenados.

---

## Índice
1. [Resumen de resultados (leer primero)](#1-resumen-de-resultados-leer-primero)
2. [Inicio rápido: usar la aplicación](#2-inicio-rápido-usar-la-aplicación)
3. [Instalación](#3-instalación)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Los datos](#5-los-datos)
6. [Cómo funciona cada parte](#6-cómo-funciona-cada-parte)
7. [Reproducir todo el pipeline](#7-reproducir-todo-el-pipeline)
8. [Resultados detallados](#8-resultados-detallados)
9. [Decisiones de ingeniería y desviaciones del plan](#9-decisiones-de-ingeniería-y-desviaciones-del-plan)
10. [Limitaciones y ética](#10-limitaciones-y-ética)
11. [Pendientes del equipo](#11-pendientes-del-equipo)
12. [Calidad de código y pruebas](#12-calidad-de-código-y-pruebas)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Mapa temas del curso → código](#14-mapa-temas-del-curso--código)
15. [Referencias y licencias de datos](#15-referencias-y-licencias-de-datos)

---

## 1. Resumen de resultados (leer primero)

Todos los números salen de `reports/` y se regeneran con `make evaluar`. **Son resultados reales, incluidos los poco favorables.**

| Componente | Resultado | Lectura honesta |
|---|---|---|
| Clasificador (Transformer **XLM-R, inglés→español**, desplegado) | F1 macro **0.645** (es) / **0.626** (en) en prueba | Detectar desinformación solo con texto es difícil (LIAR binario ronda 0.62–0.70). XLM-R **no supera claramente** al MLP TF-IDF (0.623 / 0.641). |
| Mejor modelo en español | XLM-R **solo español**: F1 **0.686** | El preentrenamiento en inglés (LIAR) **no ayudó** a noticias en español; ver [§9](#9-decisiones-de-ingeniería-y-desviaciones-del-plan). |
| Calibración (temperature scaling, T≈2.0) | ECE total **0.131 → 0.023** | Funciona muy bien: un «70 %» realmente significa ≈70 %. |
| Propagación modo A (solo texto → 48 h) | RMSE log **1.151** (LSTM) vs **1.188** (curva promedio) en noticias falsas | El texto aporta **poca** información sobre cuánto se viralizará algo: la mejora sobre la curva promedio es pequeña. |
| Propagación modo B (6 h → 42 h) | PatchTST MAE **0.149** vs SIR **0.294**; curva promedio anclada 0.173 | Los modelos neuronales ganan a SIR con claridad, y a la curva promedio por poco. |
| Banda de incertidumbre | Cobertura **75 %** (nominal 80 %) en prueba | Calibrada empíricamente con residuos de validación. |
| ¿Se reproduce «falsas se difunden más» (Vosoughi 2018)? | Twitter15/16: sí, levemente (alcance mediano 19 vs 15). PHEME: **no** (6 vs 12). | Ver [§8.3](#83-falsas-vs-verdaderas). Los dos corpus miden cosas distintas. |

**El agente completo en 50 noticias propias aún no está medido**: el conjunto lo debe armar y verificar el equipo ([§11](#11-pendientes-del-equipo)).

---

## 2. Inicio rápido: usar la aplicación

```bash
make app            # o: .venv/bin/python -m shuteai.app.interfaz
```
Abre **http://127.0.0.1:7860**. La primera consulta tarda ~7 s (carga de modelos); las siguientes, 0.2–1.5 s.

**Cómo usar la interfaz**

1. Pestaña **Analizar**. Pega una noticia (español o inglés) en el cuadro. Puedes escribir «¿Es falsa esta noticia?: …» y el sistema extrae solo la noticia.
2. Pulsa **Analizar** (o Enter). También hay **ejemplos** pre-cargados (son ilustrativos, no noticias reales verificadas).
3. Lee el resultado, de arriba abajo:
   - **Tarjeta de veredicto:** «probablemente falsa/verdadera» con su probabilidad y las dos barras `P(verdadera)` / `P(falsa)`.
   - **Tarjetas de propagación** (solo si es probablemente falsa): *pico de difusión*, *hora en que se alcanza el 50 %* y *alcance a 48 h*, cada una con su rango probable.
   - **Gráfica:** curva de compartidos acumulados con banda de incertidumbre (10–90 %) y marcas del pico y del 50 %.
   - **Palabras que más pesaron:** mapa de atención del clasificador (más oscuro = más influyente).
   - **Informe en texto** con la limitación de uso.
4. **Opciones avanzadas → Umbral de falsedad** (0.05–0.95, por defecto 0.5): si lo subes, solo se estima propagación para noticias más claramente falsas.
5. Otras pestañas: **Cómo funciona**, **Resultados del modelo** (tablas de `reports/`) y **Límites y ética**.

**Cómo interpretar** (importante): el sistema **no verifica hechos**. Detecta patrones de lenguaje típicos de la desinformación (urgencia, alarmismo, llamados a compartir). Usa la salida para **priorizar qué verificar**, no como veredicto.

**Usarlo desde código**

```python
from shuteai.agente.agente import Agente
informe = Agente().analizar("¿Es falsa esta noticia?: URGENTE: cortarán el agua 15 días, compartan")
informe.p_falsa, informe.es_falsa, informe.pico_h, informe.alcance   # (mediana, p25/p10, p75/p90)
```

---

## 3. Instalación

Requisitos: Python 3.10+ (probado con 3.13), macOS/Linux, ~6 GB libres (modelos + datos + entorno). Se entrena en CUDA, Apple Silicon (MPS) o CPU (más lento); se elige solo.

```bash
make instalar       # crea .venv e instala requirements-dev.txt y el paquete
make datos          # descarga y prepara los 4 conjuntos públicos (~630 MB)
```
Si ya tienes los modelos entrenados en `models/`, basta `make app`.

---

## 4. Estructura del repositorio

```
ShuteAI/
├── README.md                    ← este archivo
├── Makefile                     ← atajos: instalar, datos, entrenar-*, evaluar, app, test, lint
├── configs/default.yaml         ← TODOS los hiperparámetros (nada "mágico" en el código)
├── pyproject.toml, requirements*.txt
├── data/
│   ├── raw/                     ← descargas originales (no se versionan)
│   ├── processed/               ← CSV/NPY/PT generados (no se versionan)
│   └── eval_propia/             ← las 50 noticias de prueba propias (plantilla + LEEME)
├── models/                      ← pesos entrenados (no se versionan)
│   ├── clasificador/{mlp,cnn,bilstm,xlmr_en_es,…,final→xlmr_en_es}
│   └── propagacion/{a_*,b_*,final.json,residuos_val.npy,incertidumbre.json}
├── reports/                     ← tablas (.md/.json), figuras y logs de entrenamiento
├── scripts/                     ← colas de entrenamiento (experimentos y ablaciones)
├── src/shuteai/
│   ├── datos/                   ← descarga, unificación de texto, series de propagación, set propio
│   ├── preprocesamiento/        ← Paso 1: limpieza del prompt
│   ├── clasificador/            ← Paso 2: 4 modelos, entrenamiento, calibración, evaluación, inferencia
│   ├── propagacion/             ← Paso 3: baselines, 5 modelos, entrenamiento, incertidumbre, evaluación
│   ├── agente/                  ← orquestación de los 3 pasos, informe, evaluación del agente
│   └── app/                     ← interfaz Gradio
└── tests/                       ← 28 pruebas automáticas
```
Los nombres `datos/ clasificador/ propagacion/ agente/ app/` siguen el entregable del plan.

---

## 5. Los datos

Todo es **público** y se descarga con `python -m shuteai.datos.descarga` (idempotente). Etiqueta unificada: **0 = verdadera, 1 = falsa**.

| Conjunto | Qué es | Idioma | Uso en ShuteAI | Tras procesar |
|---|---|---|---|---|
| **Spanish Fake News Corpus** (Posadas-Durán et al., 2019) + test de **FakeDeS** (IberLEF 2021) | Noticias completas, titular + cuerpo, `True`/`Fake` | es | Clasificador (español). Particiones **oficiales** train/dev/test | 676 / 294 / 572 |
| **LIAR** (Wang, 2017) | 12.8 k afirmaciones cortas de PolitiFact en 6 niveles | en | Clasificador (preentrenamiento). `pants-fire, false, barely-true → falsa`; `half-true, mostly-true, true → verdadera` | 10 246 / 1 279 / 1 277 |
| **Twitter15 y Twitter16** (Ma et al., 2017) | Tuit fuente + etiqueta (`true/false/unverified/non-rumor`) + árbol de retuits con retraso en minutos | en | **Propagación** (todas las etiquetas) y clasificador (solo `true`/`false`) | 2 139 curvas; 738 / 148 / 169 tuits para clasificar |
| **PHEME** (Kochkina et al., 2018) | Hilos de 9 eventos con respuestas y marcas de tiempo | en | **Validación externa** de propagación (solo hilos de rumor, 2 402) | — |
| **Conjunto propio** | 50 noticias (25 falsas + 25 verdaderas) | es | Evaluación final del agente, **nunca para entrenar** | **pendiente** |

**Particiones sin fugas.** Twitter/PHEME se dividen con un *hash del ID* (70/15/15), no con una semilla: el mismo tuit cae siempre en la misma partición para clasificación y propagación. Los textos duplicados exactos se eliminan antes de dividir.

**Series de propagación.** Para cada árbol: histograma de retrasos en 96 intervalos de 30 min → acumulado → `log1p`. La fuente cuenta en t=0. Retuits después de 48 h no cuentan. Implementado en `datos/propagacion.py` (equivale al código del plan, más robusto a duplicados).

**Mitigación de «el modelo aprende el medio, no la veracidad».** `preprocesamiento/limpieza.py` elimina URLs, emojis, firmas de agencia (`(Reuters) -`), autores («Por …») y el **nombre del medio** del campo `Source`. Aun así, el corpus español tiene sesgos de estilo (p. ej., titulares en MAYÚSCULAS en las falsas) y el test de FakeDeS trae **medios y temas distintos** (COVID): mide generalización real, por eso su F1 es menor que el de validación.

---

## 6. Cómo funciona cada parte

### 6.1 Paso 1 — Preparación del prompt (`preprocesamiento/limpieza.py`)
Reglas y expresiones regulares (sin modelo): extrae la noticia de frases como «¿Es falsa esta noticia?: …» o «Is this true? …», quita comillas envolventes, URLs, emojis y espacios, y detecta idioma (`langdetect`). Devuelve **advertencias** (texto muy corto, idioma no soportado). El **mismo** `limpiar_texto` se usa al entrenar y al inferir.

### 6.2 Paso 2 — Clasificador verdadera/falsa (`clasificador/`)
Se comparan **cuatro** modelos con los mismos datos (S1, S2, S3, S6):

| Modelo | Arquitectura | Archivo |
|---|---|---|
| MLP | TF-IDF (5 000 términos, uni+bigramas) → 512 → 128 → 2, dropout | `modelos.py::MLP` |
| CNN 1D | Embeddings → filtros de 3, 4 y 5 palabras → max-pooling → densa | `CNN1D` |
| BiLSTM | Embeddings → BiLSTM de 2 capas → (max ‖ media) → densa | `BiLSTM` |
| **Transformer (final)** | **XLM-RoBERTa base** con fine-tuning (o BETO, `--base`) | `ClasificadorTransformer` |

**Entrenamiento** (`entrenar.py`, prácticas de S8): entropía cruzada **ponderada por clase**, dropout, **AdamW** con *weight decay* (sin decay en bias/normalización), **warmup + coseno**, recorte de gradiente, **early stopping** sobre F1 macro de validación. Para XLM-R se **congela la matriz de embeddings de palabras** (≈70 % de sus parámetros; ahorra ~2.5 GB de RAM) y se usa batch 8 con **acumulación ×4** (batch efectivo 32) y lr 1e-5.

**Recetas** (`--receta`): `mezcla` (todo junto), `en_es` (preentrena en inglés y ajusta en español con lr/2 — **transfer learning, S9**) y `solo_es`.

**Calibración** (`calibracion.py`): *temperature scaling*. Se aprende un único escalar `T` minimizando la entropía cruzada en validación y se calcula `softmax(logits / T)`. Se mide con **ECE**, **Brier** y diagramas de confiabilidad (`reports/figuras/confiabilidad_*.png`). Para el modelo desplegado `T≈2.0`: el modelo crudo era sobreconfiado.

**Mapa de atención** (`inferencia.py`): *attention rollout* (Abnar & Zuidema, 2020) agregando subtokens a palabras. Se recorta en el percentil 90 antes de normalizar porque el rollout concentra masa en 1–2 tokens. **Es una pista de qué texto influyó, no una explicación causal.**

### 6.3 Paso 3 — Modelo de propagación (`propagacion/`)
Entrada: embeddings del clasificador (`[CLS]` y tokens de 768 dim, precalculados en `data/processed/emb_*.pt`). Salida: **96 valores** (48 h × 30 min) de alcance acumulado en `log1p`. Todos los modelos predicen **incrementos positivos (softplus) y los acumulan**, así la curva nunca decrece.

**Modo A — solo texto (el que usa el agente):**

| Modelo | Detalle |
|---|---|
| Baseline 1 | Curva promedio de las noticias falsas de entrenamiento |
| MLP (S1) | Embedding → 96 incrementos |
| **Encoder-decoder LSTM con atención (S3–S5, desplegado)** | El embedding inicializa `h0, c0`; en cada paso el decoder atiende a las palabras de la noticia (`PropagadorLSTM`) |
| Transformer (S6) | Decoder Transformer sobre los 96 pasos con atención cruzada al texto |

**Modo B — pronóstico (S12):** se ven las primeras **6 h** (12 pasos) y se pronostican las **42 h** siguientes (84 pasos). Modelos: baseline curva promedio anclada, **SIR** (ajustado por mínimos cuadrados a cada serie, `baselines.py`), LSTM, Transformer punto a punto y **PatchTST** (parches de 4, paso 2). Solo usan la serie.

**Pérdida:** Huber (δ=0.5) sobre la escala log, para tolerar curvas muy variables (pocas noticias se vuelven virales).

**Incertidumbre** (`incertidumbre.py`): se tomó el **bootstrap de residuos de validación**: `curva_k = predicción + residuo_k`, donde `residuo_k = real − predicho` en un hilo de validación. Se descartó MC-Dropout porque subestimaba muchísimo el error (alcance «15, rango 12–17» cuando el error típico del modelo es de un factor ≈3 en el alcance). Cobertura medida en prueba: **75 %** para banda nominal 80 %.

### 6.4 El agente (`agente/agente.py`)
```
prompt ─▶ [1] limpieza ─▶ [2] clasificador calibrado ─▶ P(falsa) ≥ umbral? ─ no ─▶ informe «probablemente verdadera»
                                                              │ sí
                                                              ▼
                                           [3] propagación (96 pasos + banda) ─▶ pico, t50, alcance ─▶ informe
```
- La decisión de veracidad es **siempre** del clasificador entrenado (nunca un LLM).
- Carga **perezosa** de modelos; acepta clasificador y propagador **inyectados** (así se prueba con modelos simulados, `tests/test_agente.py`).
- `redactar()` (`agente/informe.py`) genera el informe con una plantilla en español que **siempre** incluye la incertidumbre y la limitación de no verificar hechos.

---

## 7. Reproducir todo el pipeline

```bash
make datos                  # ~2 min   descarga + preparación
make entrenar-base          # ~5 min   MLP, CNN, BiLSTM
make entrenar-transformer   # ~35 min  XLM-R inglés→español + calibración + fija el modelo final
make embeddings             # ~20 s    embeddings para propagación
make entrenar-prop          # ~35 min  6 modelos de propagación + incertidumbre
make evaluar                # ~3 min   reports/*.md, *.json, figuras
make app                    # interfaz
```
Tiempos medidos en un MacBook con **Apple M2 Pro, 16 GB** (MPS). Experimentos adicionales: `scripts/cola_entrenamiento.sh` (XLM-R solo-es, BETO, mezcla) y `scripts/cola_ablaciones.sh`.

Entrenar una variante a mano:
```bash
python -m shuteai.clasificador.entrenar --modelo transformer --receta solo_es --epocas 10 --nombre xlmr_solo_es
python -m shuteai.clasificador.entrenar --modelo transformer --receta solo_es --base dccuchile/bert-base-spanish-wwm-cased --nombre beto_solo_es
python -m shuteai.clasificador.evaluar --final xlmr_solo_es     # cambia el modelo desplegado
```
> **Si cambias el clasificador final**, ejecuta además `make embeddings entrenar-prop`: el propagador se entrenó con los embeddings del modelo anterior y hay que regenerarlos.

---

## 8. Resultados detallados

### 8.1 Clasificador (conjunto de prueba; probabilidades calibradas con T de validación)
`ECE` y `Brier` se muestran **antes → después** de temperature scaling. «español» = test de FakeDeS; «inglés» = LIAR + Twitter test.

| Modelo | Grupo | Acc | F1 macro | AUC | ECE antes→después | Brier antes→después |
|---|---|---|---|---|---|---|
| beto_solo_es | español | 0.668 | 0.664 | 0.751 | 0.214→0.130 | 0.259→0.223 |
| beto_solo_es | todo | 0.668 | 0.664 | 0.751 | 0.214→0.130 | 0.259→0.223 |
| bilstm | español | 0.605 | 0.602 | 0.660 | 0.237→0.119 | 0.294→0.248 |
| bilstm | inglés | 0.632 | 0.608 | 0.683 | 0.097→0.037 | 0.231→0.222 |
| bilstm | todo | 0.624 | 0.615 | 0.670 | 0.133→0.042 | 0.249→0.230 |
| cnn | español | 0.577 | 0.562 | 0.642 | 0.116→0.053 | 0.250→0.238 |
| cnn | inglés | 0.628 | 0.611 | 0.668 | 0.116→0.031 | 0.241→0.226 |
| cnn | todo | 0.613 | 0.610 | 0.660 | 0.112→0.032 | 0.244→0.230 |
| mlp | español | 0.636 | 0.623 | 0.711 | 0.256→0.140 | 0.287→0.240 |
| mlp | inglés | 0.641 | 0.641 | 0.691 | 0.086→0.041 | 0.232→0.224 |
| mlp | todo | 0.640 | 0.637 | 0.677 | 0.130→0.046 | 0.248→0.228 |
| xlmr_en_es | español | 0.645 | 0.645 | 0.699 | 0.087→0.026 | 0.228→0.220 |
| xlmr_en_es | inglés | 0.626 | 0.626 | 0.669 | 0.153→0.038 | 0.256→0.232 |
| xlmr_en_es | todo | 0.631 | 0.631 | 0.674 | 0.131→0.023 | 0.248→0.229 |
| xlmr_mezcla | español | 0.638 | 0.630 | 0.714 | 0.055→0.058 | 0.223→0.222 |
| xlmr_mezcla | inglés | 0.627 | 0.619 | 0.669 | 0.046→0.024 | 0.228→0.225 |
| xlmr_mezcla | todo | 0.630 | 0.629 | 0.680 | 0.042→0.023 | 0.226→0.224 |
| xlmr_solo_es | español | 0.687 | 0.686 | 0.746 | 0.167→0.098 | 0.233→0.210 |
| xlmr_solo_es | todo | 0.687 | 0.686 | 0.746 | 0.167→0.098 | 0.233→0.210 |

**Lecturas.** (1) Temperature scaling reduce el ECE en **todos** los modelos. (2) En inglés, los cuatro modelos están en una banda estrecha (F1 0.61–0.64): LIAR es ruidoso y apenas hay señal textual. (3) En español, los modelos entrenados **solo con español** son los mejores; el transfer inglés→español no ayudó. (4) Hay brecha validación→prueba en español (0.74→0.65 para `xlmr_en_es`) por el cambio de medios y temas.

### 8.2 Propagación

#### Modo A – solo texto → curva de 48 h (escala log)

##### Twitter15/16 – prueba, noticias falsas

| Modelo | rmse | mae | error_pico_h | error_alcance_log |
|---|---|---|---|---|
| curva promedio | 1.188 | 0.977 | 0.747 | 0.987 |
| mlp | 1.160 | 0.949 | 0.988 | 0.945 |
| lstm | 1.151 | 0.940 | 0.747 | 0.949 |
| transformer | 1.210 | 0.986 | 4.982 | 0.962 |

##### Twitter15/16 – prueba, todas

| Modelo | rmse | mae | error_pico_h | error_alcance_log |
|---|---|---|---|---|
| curva promedio | 1.087 | 0.875 | 0.331 | 0.885 |
| mlp | 1.059 | 0.849 | 0.589 | 0.849 |
| lstm | 1.025 | 0.825 | 0.331 | 0.838 |
| transformer | 1.155 | 0.928 | 4.919 | 0.881 |

##### PHEME – validación externa (falsas, sin reentrenar)

| Modelo | rmse | mae | error_pico_h | error_alcance_log |
|---|---|---|---|---|
| curva promedio | 1.304 | 1.054 | 0.357 | 1.093 |
| mlp | 1.334 | 1.077 | 0.632 | 1.089 |
| lstm | 1.565 | 1.290 | 0.357 | 1.366 |
| transformer | 1.289 | 1.054 | 5.001 | 1.058 |

#### Modo B – 6 h observadas → 42 h pronosticadas

##### Prueba, falsas

| Modelo | rmse | mae | mape |
|---|---|---|---|
| curva promedio anclada | 0.280 | 0.173 | 17.969 |
| SIR | 0.763 | 0.294 | 360.637 |
| lstm | 0.290 | 0.163 | 15.834 |
| transformer | 0.267 | 0.156 | 15.674 |
| patchtst | 0.280 | 0.149 | 13.748 |

##### Prueba, todas

| Modelo | rmse | mae | mape |
|---|---|---|---|
| curva promedio anclada | 0.213 | 0.141 | 15.408 |
| SIR | 0.553 | 0.191 | 179.458 |
| lstm | 0.213 | 0.124 | 12.655 |
| transformer | 0.203 | 0.125 | 13.376 |
| patchtst | 0.206 | 0.109 | 10.769 |

#### Falsas vs. verdaderas (alcance a 48 h)

| Corpus | n | alcance medio | mediano | t50 mediano (h) |
|---|---|---|---|---|
| twitter_false | 575 | 31.6 | 19.0 | 1.0 |
| twitter_true | 579 | 27.4 | 15.0 | 0.5 |
| pheme_false | 638 | 10.2 | 6.0 | 0.5 |
| pheme_true | 1067 | 14.3 | 12.0 | 0.5 |

**Lecturas.** (1) **Modo A:** la ganancia del LSTM sobre la curva promedio es pequeña (RMSE 1.15 vs 1.19): saber de qué trata la noticia dice poco de cuán viral será. El Transformer **colapsa** a una curva casi constante (error de pico ≈ 5 h) y no se usa. (2) **PHEME no valida el modo A:** el LSTM empeora (RMSE 1.57 vs 1.30). PHEME cuenta **respuestas**, no retuits, y los hilos son más pequeños, así que es otro fenómeno; se reporta como límite de generalización, no como éxito. (3) **Modo B:** PatchTST obtiene el menor MAE y MAPE; el Transformer, el menor RMSE; la curva promedio anclada es un baseline fuerte. **SIR** queda claramente peor; una explicación plausible (no verificada) es que 6 h con pocos compartidos no identifican bien sus tres parámetros. Los MAPE son altos en general porque el denominador (alcance real) es pequeño.

### 8.3 Falsas vs. verdaderas
Figura: `reports/figuras/propagacion_falsas_vs_verdaderas.png`. En **Twitter15/16** las falsas tienen algo más de alcance (mediana 19 vs 15; media 31.6 vs 27.4) y tardan algo más en llegar al 50 % (t50 mediano 1.0 h vs 0.5 h), coherente en parte con Vosoughi et al. (2018); en **PHEME** ocurre lo contrario (6 vs 12). No son comparables directamente (retuits vs respuestas, eventos distintos) y no se ha hecho prueba de significancia.

### 8.4 Ablaciones (S8)
Cada variante cambia **una** cosa respecto a su referencia (misma semilla y datos). La referencia de la BiLSTM usa mezcla de idiomas; la de XLM-R, solo español.

| Referencia | Variante | Grupo | F1 macro | ECE (calibrado) |
|---|---|---|---|---|
| bilstm | (referencia) | todo | 0.615 | 0.042 |
| bilstm | lr_constante | todo | 0.611 | 0.051 |
| bilstm | sin_dropout | todo | 0.614 | 0.025 |
| bilstm | sin_pesos | todo | 0.611 | 0.032 |
| xlmr_solo_es | (referencia) | español | 0.686 | 0.098 |
| xlmr_solo_es | lr_constante | español | 0.701 | 0.101 |
| xlmr_solo_es | sin_pesos | español | 0.687 | 0.100 |

**Resultado:** ninguna ablación mueve el F1 más de 0.015 (la mayor, `lr_constante` en XLM-R, *mejora* 0.686→0.701), así que con esta evidencia **no se puede afirmar** que pesos de clase, dropout o warmup+coseno aporten aquí; solo `sin_dropout` cambia algo visible en calibración (ECE 0.042→0.025).

> Notas: en XLM-R solo-español las clases están balanceadas (50/50), por lo que quitar los pesos de clase **no puede** diferir mucho; el efecto es visible en la BiLSTM (datos 44 % falsas). La ablación de dropout se hizo solo en la BiLSTM; XLM-R usa dropout 0.1 fijo. Con una sola semilla las diferencias pequeñas (≲0.02 de F1) **no son concluyentes**.

---

## 9. Decisiones de ingeniería y desviaciones del plan

| Tema | Decisión | Por qué |
|---|---|---|
| XLM-R final | `xlmr_en_es` aunque `xlmr_solo_es` gana en español (0.686 vs 0.645) | Es **bilingüe** (el plan admite ambos idiomas), el mejor calibrado (ECE 0.026) y sus embeddings ya alimentan al propagador. Si tu prioridad es español, cambia con `--final xlmr_solo_es` y regenera embeddings ([§7](#7-reproducir-todo-el-pipeline)). |
| Embeddings congelados en XLM-R | Matriz de palabras congelada | 16 GB sin swap: el entrenamiento completo fue terminado por el sistema (código 137) |
| Hiperparámetros XLM-R | lr 1e-5, batch efectivo 32 | Con 2e-5 y batch 8 **colapsó** a azar (pérdida 0.693 en todas las épocas) |
| Propagador LSTM | + máscara de *padding* en la atención, tiempo normalizado como entrada, *scheduled sampling* | El código del plan usaba teacher forcing puro (sesgo de exposición) y no enmascaraba el relleno |
| Entrenamiento modo A | Con las **4** etiquetas de Twitter15/16 (2 139 curvas); se evalúa en las «falsas» | Más datos para aprender texto→curva; el plan fija la curva promedio solo con falsas, así se hace en el baseline |
| Incertidumbre | Bootstrap de residuos, no MC-Dropout | Cobertura empírica medida (75 %) |
| Modo B | Solo la serie, sin texto | Es la evaluación de forecasting (S12) |
| Cota de SIR | `log10 N ≤ 3.5` | Con cota 10⁶ extrapolaba hasta 10⁶ compartidos; el máximo observado es ~380 |
| Autoencoders (S4) | No se usan | Lo dice el plan |
| **GAN (S7)** | **No se usa** | No aporta al flujo (clasificar + pronosticar). Generar noticias sintéticas no mejora ni la calibración ni el pronóstico. |
| **VAE condicional (S14, opcional)** | **No implementado** | Opcional; la banda de incertidumbre se cubre con residuos |
| Idioma de propagación | Mismo encoder multilingüe para es/en | El propagador se entrenó con tuits en inglés; en español el comportamiento **no está validado** |

---

## 10. Limitaciones y ética

- **No comprueba hechos.** Sin RAG ni internet, detecta patrones de lenguaje y estilo. Entrega **probabilidades**, no un veredicto definitivo, y **no debe usarse para censurar contenido**.
- **Rendimiento modesto.** F1 ≈ 0.63–0.69: se equivoca con frecuencia. Una noticia verdadera con tono alarmista puede salir «falsa» y una falsa bien redactada puede pasar.
- **Sesgo de fuente y de estilo.** Puede aprender a reconocer el estilo de ciertos medios o temas más que la veracidad (aunque se quitaron nombres y firmas).
- **Datos de propagación antiguos** (Twitter 2015–2016) y solo de retuits: la curva es una estimación del patrón, no una predicción exacta. La mayoría del alcance llega en los primeros 30 min, por eso el pico suele ser la hora 0.5.
- **Español sin validar en propagación.** El modelo de propagación solo vio tuits en inglés.
- La **atención no es una explicación causal.**
- Úsalo para **priorizar qué verificar**, nunca como juez final.

---

## 11. Pendientes del equipo

1. **Confirmar con el catedrático** (primer día, según el plan): (a) que se acepta un Transformer preentrenado con fine-tuning como clasificador final; (b) que la propagación se modele con datos públicos de Twitter.
2. **Armar las 50 noticias propias** en `data/eval_propia/noticias_prueba.csv` (25 falsas verificadas por medios de verificación + 25 verdaderas de medios formales; **con enlace de verificación obligatorio**). Valida con `python -m shuteai.datos.eval_propia` y mide con `python -m shuteai.agente.evaluar`. *No se inventaron noticias «verificadas»: la etiqueta necesita evidencia real.*
3. **Informe técnico** (tipo artículo), **presentación** con demo en vivo y **video de respaldo**.
4. *(Opcional)* VAE condicional de escenarios (S14); comparar más semillas para volver concluyentes las ablaciones.
5. *(Opcional)* Si el set propio muestra que el español rinde peor de lo esperado, cambiar a `xlmr_solo_es` o a BETO.

---

## 12. Calidad de código y pruebas

```bash
make test     # 28 pruebas (pytest)
make lint     # ruff check + format
```
- Pruebas: limpieza del prompt, curvas y particiones, métricas y temperature scaling, **monotonía y formas** de todos los modelos de propagación, enmascarado del relleno, agente con clasificador/propagador simulados, cobertura de curvas con residuos y agrupación de subtokens.
- Configuración centralizada (`configs/default.yaml`), semillas fijas, particiones deterministas por hash, rutas centralizadas (`rutas.py`), carga perezosa de modelos, inyección de dependencias en el agente.
- Los datos y pesos no se versionan; el pipeline es reproducible con `make`.

---

## 13. Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `Faltan los modelos entrenados` en la app | Falta `models/clasificador/final` o `models/propagacion/`. Ejecuta `make entrenar-transformer entrenar-prop` |
| El equipo se traba o el proceso muere con código 137 | Falta de RAM (sin swap). Cierra apps pesadas, baja `clasificador.batch` en `configs/default.yaml` (y sube `acumulacion` para conservar el batch efectivo) |
| El Transformer no aprende (pérdida ≈ 0.693) | lr demasiado alto para XLM-R: usa 1e-5 y acumulación de gradiente |
| `models/` ocupa ~3.6 GB | Cada XLM-R pesa ~1.1 GB. Borra los experimentos que no uses (`xlmr_mezcla`, `abl_*`…); `final` apunta a `xlmr_en_es` |
| Un `true`/`false` del YAML se vuelve booleano | Entrecomilla los textos en `default.yaml` (ya pasó con las etiquetas de LIAR) |
| Descarga falla | Reintenta `python -m shuteai.datos.descarga`; las URLs están en `datos/descarga.py` |

---

## 14. Mapa temas del curso → código

| Sesión | Tema | Dónde |
|---|---|---|
| S1 | MLP / líneas base | `clasificador/modelos.py::MLP`, `propagacion/modelos.py::PropMLP` |
| S2 | CNN 1D para texto | `clasificador/modelos.py::CNN1D` |
| S3 | RNN, LSTM, GRU | `BiLSTM`; `PropagadorLSTM`; `ForecasterLSTM` |
| S4 | Encoder-decoder | `PropagadorLSTM` (el encoder lee la noticia, el decoder genera la curva) |
| S5 | Atención | Atención del decoder sobre las palabras; *attention rollout* del clasificador |
| S6 | Transformers | `ClasificadorTransformer`; `PropagadorTransformer`; `PatchTransformer` |
| S7 | GAN | **No se usa** (justificación en [§9](#9-decisiones-de-ingeniería-y-desviaciones-del-plan)) |
| S8 | Pérdidas, regularización, optimización | Entropía cruzada ponderada, Huber, dropout, AdamW, warmup+coseno, early stopping, temperature scaling, ablaciones |
| S9 | Transfer learning / fine-tuning | Receta `en_es`; BETO vs XLM-R |
| S12 | Series de tiempo | Modo B, SIR, curva promedio, PatchTST |
| S13 | NLP | Tokenización, embeddings, limpieza del prompt, informe |
| S14 | IA generativa | Opcional, no implementado |

---

## 15. Referencias y licencias de datos

- Vosoughi, Roy, Aral (2018). *The spread of true and false news online.* Science.
- Posadas-Durán et al. (2019). *Detection of fake news in a new corpus for the Spanish language.* (CC BY 4.0) · FakeDeS, IberLEF 2021.
- Wang (2017). *“Liar, Liar Pants on Fire”.* ACL.
- Ma, Gao, Wong (2017). *Detect rumors in microblog posts using propagation structure via kernel learning.* ACL. (Twitter15/16)
- Kochkina, Liakata, Zubiaga (2018). *PHEME dataset for rumour detection and veracity classification.* (CC BY 4.0)
- Conneau et al. (2020), *XLM-R*; Cañete et al. (2020), *BETO*; Nie et al. (2023), *PatchTST*; Guo et al. (2017), *On calibration of modern neural networks*; Abnar & Zuidema (2020), *Quantifying attention flow*.
- Revisa la licencia/términos de cada conjunto antes de redistribuirlo; este repositorio **no versiona** los datos.
