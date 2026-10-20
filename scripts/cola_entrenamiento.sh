#!/usr/bin/env bash
# Cola secuencial (no satura la RAM de 16 GB). Logs en reports/logs/.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=reports/logs
for m in mlp lstm transformer; do $PY -m shuteai.propagacion.entrenar --modo a --modelo $m > $L/prop_a_$m.log 2>&1; done
for m in lstm transformer patchtst; do $PY -m shuteai.propagacion.entrenar --modo b --modelo $m > $L/prop_b_$m.log 2>&1; done
echo '{"tipo": "lstm"}' > models/propagacion/final.json
$PY -m shuteai.clasificador.entrenar --modelo transformer --receta solo_es --nombre xlmr_solo_es --epocas 10 > $L/clf_xlmr_solo_es.log 2>&1
$PY -m shuteai.clasificador.entrenar --modelo transformer --receta solo_es --base dccuchile/bert-base-spanish-wwm-cased --nombre beto_solo_es --epocas 10 > $L/clf_beto_solo_es.log 2>&1
$PY -m shuteai.clasificador.entrenar --modelo transformer --receta mezcla --nombre xlmr_mezcla > $L/clf_xlmr_mezcla.log 2>&1
echo COLA_TERMINADA > $L/cola.done
