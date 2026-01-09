#!/bin/bash
# FIX CRITICO PER MAC OS (Apple Silicon / Intel)
# Esportiamo queste variabili PRIMA di avviare Python
export KMP_DUPLICATE_LIB_OK=True
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== AVVIO TRAINING F1 PREDICTION ==="

echo ">> Step 1: Modelli Classici (RandomForest, MLP)..."
python3 train_classic.py
if [ $? -ne 0 ]; then
    echo "Errore durante il training classico."
    exit 1
fi

echo ""
echo "=== TRAINING COMPLETATO CON SUCCESSO! ==="
echo "Ora puoi usare: python3 predict_race.py"
