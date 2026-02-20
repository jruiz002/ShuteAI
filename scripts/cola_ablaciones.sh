#!/usr/bin/env bash
# Ablaciones de S8 sobre BiLSTM (mezcla) y XLM-R solo español. Referencias: bilstm y xlmr_solo_es.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs
E="-m shuteai.clasificador.entrenar"
$PY $E --modelo bilstm --sin-pesos --nombre abl_bilstm_sin_pesos > $L/abl_bilstm_sin_pesos.log 2>&1
$PY $E --modelo bilstm --dropout 0.0 --nombre abl_bilstm_sin_dropout > $L/abl_bilstm_sin_dropout.log 2>&1
$PY $E --modelo bilstm --lr-constante --nombre abl_bilstm_lr_constante > $L/abl_bilstm_lr_constante.log 2>&1
T="--modelo transformer --receta solo_es --epocas 10"
$PY $E $T --sin-pesos --nombre abl_xlmr_sin_pesos > $L/abl_xlmr_sin_pesos.log 2>&1
$PY $E $T --lr-constante --nombre abl_xlmr_lr_constante > $L/abl_xlmr_lr_constante.log 2>&1
echo ok > $L/ablaciones.done
