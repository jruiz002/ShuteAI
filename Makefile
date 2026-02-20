PY := .venv/bin/python
.PHONY: instalar datos entrenar-base entrenar-transformer embeddings entrenar-prop evaluar app test lint todo

instalar:            ## Crea el entorno e instala dependencias
	python3.13 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e .

datos:               ## Descarga y prepara todos los conjuntos
	$(PY) -m shuteai.datos.descarga
	$(PY) -m shuteai.datos.textos
	$(PY) -m shuteai.datos.propagacion

entrenar-base:       ## MLP, CNN 1D y BiLSTM (minutos)
	for m in mlp cnn bilstm; do $(PY) -m shuteai.clasificador.entrenar --modelo $$m; done

entrenar-transformer: ## XLM-R: preentrena en inglés y ajusta en español (~1 h en Apple Silicon)
	$(PY) -m shuteai.clasificador.entrenar --modelo transformer --receta en_es --nombre xlmr_en_es
	$(PY) -m shuteai.clasificador.evaluar --final xlmr_en_es

embeddings:          ## Embeddings del clasificador para el modelo de propagación
	$(PY) -m shuteai.propagacion.embeddings

entrenar-prop:       ## Modelos de propagación (modo A y B)
	for m in mlp lstm transformer; do $(PY) -m shuteai.propagacion.entrenar --modo a --modelo $$m; done
	for m in lstm transformer patchtst; do $(PY) -m shuteai.propagacion.entrenar --modo b --modelo $$m; done
	echo '{"tipo": "lstm"}' > models/propagacion/final.json
	$(PY) -m shuteai.propagacion.incertidumbre

evaluar:             ## Tablas y figuras en reports/
	$(PY) -m shuteai.clasificador.evaluar
	$(PY) -m shuteai.propagacion.evaluar

app:                 ## Interfaz web en http://127.0.0.1:7860
	$(PY) -m shuteai.app.interfaz

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

todo: datos entrenar-base entrenar-transformer embeddings entrenar-prop evaluar
