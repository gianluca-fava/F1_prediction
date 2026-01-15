import pandas as pd
import joblib
import sys
import numpy as np
import os

import pandas as pd
import joblib
import sys
import numpy as np
import os

def load_artifacts():
    try:
        artifacts = {
            'rf': joblib.load('joblib/f1_rf_best_model.joblib'),
            'mlp': joblib.load('joblib/f1_mlp_model.joblib'),
            'le_driver': joblib.load('joblib/le_driver.joblib'),
            'le_team': joblib.load('joblib/le_team.joblib'),
            'le_circuit': joblib.load('joblib/le_circuit.joblib'),
            'le_compound': joblib.load('joblib/le_compound.joblib'),
            'scaler_x': joblib.load('joblib/scaler_mlp.joblib'),
            'scaler_y': joblib.load('joblib/scaler_y_mlp.joblib'),
            'mlp_cols': joblib.load('joblib/mlp_feature_cols.joblib')
        }
        return artifacts
    except Exception as e:
        print(f"Error loading model files: {e}")
        print("Please ensure you have run 'train.py' first.")
        sys.exit(1)

def get_user_input(art):
    print("\n--- F1 Race Predictor (RF, MLP) ---")
    
    # --- Driver ---
    while True:
        driver = input("Enter Driver code (e.g., VER, HAM, LEC): ").upper()
        if driver in art['le_driver'].classes_:
            break
        print(f"Driver not found. Examples: {', '.join(art['le_driver'].classes_[:5])}...")

    # --- Team ---
    while True:
        team = input("Enter Team name (e.g., Red Bull Racing, Ferrari): ")
        if team in art['le_team'].classes_:
            break
        print(f"Team not found. Examples: {', '.join(art['le_team'].classes_[:5])}...")

    # --- Circuit ---
    while True:
        circuit = input("Enter Circuit name (e.g., Bahrain Grand Prix): ")
        if circuit in art['le_circuit'].classes_:
            break
        print("Circuit not found.")
        print("Suggestions:", ', '.join(art['le_circuit'].classes_[:5]), "...")

    # --- Compound ---
    while True:
        compound = input("Enter Start Compound (e.g., SOFT, MEDIUM, HARD): ").upper()
        if compound in art['le_compound'].classes_:
            break
        print(f"Compound not found. Examples: {', '.join(art['le_compound'].classes_)}...")

    # --- Grid Position ---
    while True:
        try:
            grid_pos = float(input("Starting position (1-20): "))
            if 1 <= grid_pos <= 20:
                break
            print("Must be between 1 and 20.")
        except ValueError: pass

    # --- Humidity ---
    while True:
        try:
            humidity = float(input("Humidity (%) (e.g., 50.0): "))
            break
        except ValueError: pass

    # --- Temperature ---
    while True:
        try:
            temp = float(input("Track Temperature (°C) (e.g., 35.0): "))
            break
        except ValueError: pass

    # --- Wet Track ---
    is_wet = 1 if input("Is the track wet? (y/n): ").lower() == 'y' else 0

    return {
        'Driver': driver, 'Team': team, 'Circuit': circuit, 
        'Compound': compound, 'GridPosition': grid_pos, 
        'Humidity': humidity, 'Temperature': temp, 'is_wet': is_wet
    }

def predict():
    art = load_artifacts()
    u = get_user_input(art)

    # Encode inputs
    driver_enc = art['le_driver'].transform([u['Driver']])[0]
    team_enc = art['le_team'].transform([u['Team']])[0]
    circuit_enc = art['le_circuit'].transform([u['Circuit']])[0]
    compound_enc = art['le_compound'].transform([u['Compound']])[0]

    # Use defaults for complex features
    recent_form = u['GridPosition'] # Assume recent form is similar to grid position
    driver_elo = 1500.0 # Standard starting Elo

    # --- Random Forest Prediction ---
    # ['GridPosition', 'RecentForm', 'Team_Encoded', 'Compound_Label', 'Driver_Elo', 'is_wet']
    X_rf = pd.DataFrame([{
        'GridPosition': u['GridPosition'],
        'RecentForm': recent_form,
        'Team_Encoded': team_enc,
        'Compound_Label': compound_enc,
        'Driver_Elo': driver_elo,
        'is_wet': u['is_wet']
    }])
    
    pred_rf = art['rf'].predict(X_rf)[0]

    # --- MLP Prediction ---
    # ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition', 'Humidity', 'Temperature', 'RecentForm', 'is_wet'] + dummies
    base_mlp = {
        'Driver_Encoded': driver_enc,
        'Circuit_Encoded': circuit_enc,
        'GridPosition': u['GridPosition'],
        'Humidity': u['Humidity'],
        'Temperature': u['Temperature'],
        'RecentForm': recent_form,
        'is_wet': u['is_wet']
    }
    
    # Tyre Dummies
    for col in art['mlp_cols']:
        base_mlp[col] = 1 if f"tyre_{u['Compound']}" == col else 0
        
    X_mlp = pd.DataFrame([base_mlp])[art['scaler_x'].feature_names_in_]
    X_mlp_scaled = art['scaler_x'].transform(X_mlp)
    
    pred_mlp_scaled = art['mlp'].predict(X_mlp_scaled)
    pred_mlp = art['scaler_y'].inverse_transform(pred_mlp_scaled.reshape(-1, 1)).ravel()[0]

    print("\n--- PREDICTION RESULTS ---")
    print(f"Scenario: {u['Driver']} ({u['Team']}) @ {u['Circuit']}")
    print(f"Starting: {int(u['GridPosition'])} | Compound: {u['Compound']} | Wet: {'Yes' if u['is_wet'] else 'No'}")
    print("-" * 50)
    print(f"Random Forest Estimate:  Pos {pred_rf:.2f}")
    print(f"MLP (Neural) Estimate:   Pos {pred_mlp:.2f}")
    print("-" * 50)
    
    avg_pred = (pred_rf + pred_mlp) / 2
    final_pos = np.clip(np.round(avg_pred), 1, 21)
    print(f"FINAL PREDICTION: Pos {int(final_pos)}")

if __name__ == "__main__":
    predict()
