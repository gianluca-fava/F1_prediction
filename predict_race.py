import pandas as pd
import joblib
import sys
import numpy as np
import os

def load_artifacts():
    try:
        rf = joblib.load('joblib/f1_rf_model.joblib')
        mlp = joblib.load('joblib/f1_mlp_model.joblib')
        scaler = joblib.load('scaler.joblib')
        le_driver = joblib.load('joblib/le_driver.joblib')
        le_circuit = joblib.load('joblib/le_circuit.joblib')
        
        return rf, mlp, scaler, le_driver, le_circuit
    except Exception as e:
        print(f"Error loading model files: {e}")
        print("Please ensure you have run 'train_f1_model.py' first.")
        sys.exit(1)

def get_user_input(le_driver, le_circuit):
    print("\n--- F1 Race Predictor (RF, MLP) ---")
    
    # --- Driver ---
    while True:
        driver = input("Inserisci codice Pilota (es. VER, HAM, LEC): ").upper()
        if driver in le_driver.classes_:
            break
        print(f"Pilota non trovato. Esempi: {', '.join(le_driver.classes_[:5])}...")

    # --- Circuit ---
    while True:
        circuit = input("Inserisci nome Circuito (es. Bahrain Grand Prix): ")
        if circuit in le_circuit.classes_:
            break
        print("Circuito non trovato.")
        print("Suggerimenti:", ', '.join(le_circuit.classes_[:5]), "...")
        if input("Lista completa? (s/n): ").lower() == 's':
            print(', '.join(le_circuit.classes_))

    # --- Grid Position ---
    while True:
        try:
            grid_pos = float(input("Posizione di partenza (Qualifica): "))
            if 1 <= grid_pos <= 20:
                break
            print("Deve essere tra 1 e 20.")
        except ValueError: pass

    # --- Humidity ---
    while True:
        try:
            humidity = float(input("Umidità (%) (es. 60.5): "))
            if 0 <= humidity <= 100:
                break
        except ValueError: pass

    return driver, circuit, grid_pos, humidity

def predict():
    rf, mlp, scaler, le_driver, le_circuit = load_artifacts()
    
    driver_in, circuit_in, grid_pos_in, humidity_in = get_user_input(le_driver, le_circuit)

    driver_enc = le_driver.transform([driver_in])[0]
    circuit_enc = le_circuit.transform([circuit_in])[0]

    # Input DataFrame
    X_new = pd.DataFrame({
        'Driver_Encoded': [driver_enc],
        'GridPosition': [grid_pos_in],
        'Humidity': [humidity_in],
        'Circuit_Encoded': [circuit_enc]
    })

    # Scaling
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    pred_rf = rf.predict(X_new)[0]
    pred_mlp = mlp.predict(X_new_scaled)[0]

    print("\n--- RISULTATI PREDIZIONE ---")
    print(f"Scenario: {driver_in} @ {circuit_in} (Partenza: {int(grid_pos_in)})")
    print("-" * 40)
    print(f"RandomForest:  Pos {pred_rf:.2f}")
    print(f"MLP (Rete):    Pos {pred_mlp:.2f}")
    print("-" * 40)
    
    # Simple Ensemble Average
    avg_pred = (pred_rf + pred_mlp) / 2
    print(f"MEDIA ENSEMBLE: Pos {avg_pred:.2f} -> {int(round(avg_pred))}")

if __name__ == "__main__":
    predict()
