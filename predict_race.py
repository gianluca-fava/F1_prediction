import pandas as pd
import joblib
import sys
import numpy as np
import os

def load_artifacts():
    try:
        # Carichiamo i nuovi modelli V3
        hgb = joblib.load('f1_hgb_model.joblib')
        rf = joblib.load('f1_rf_best_model.joblib')
        mlp = joblib.load('f1_mlp_model.joblib')
        
        # Encoders
        le_driver = joblib.load('le_driver.joblib')
        le_circuit = joblib.load('le_circuit.joblib')
        le_team = joblib.load('le_team.joblib')
        le_compound = joblib.load('le_compound.joblib')
        
        # Scaler per MLP
        scaler = joblib.load('scaler_mlp.joblib')
        
        # Dataset per recupero feature storiche
        df_history = pd.read_parquet('f1_dataset_processed.parquet')
        
        return hgb, rf, mlp, le_driver, le_circuit, le_team, le_compound, scaler, df_history
    except Exception as e:
        print(f"❌ Errore caricamento file: {e}")
        print("Assicurati di aver eseguito 'train_classic.py' con successo.")
        sys.exit(1)

def get_historical_features(df, driver, circuit):
    # Recuperiamo l'ultimo stato noto del pilota
    driver_data = df[df['Driver'] == driver].sort_values(['Year', 'RecentForm'], ascending=False)
    
    if driver_data.empty:
        return None
    
    last_race = driver_data.iloc[0]
    
    # Recuperiamo l'Overtake Factor specifico del circuito (se noto)
    circuit_data = df[df['Circuit'] == circuit]
    if not circuit_data.empty:
        overtake_factor = circuit_data['CircuitOvertakeFactor'].iloc[0]
    else:
        overtake_factor = df['CircuitOvertakeFactor'].mean() # Fallback media globale
        
    return {
        'Team': last_race['Team'],
        'RecentForm': last_race['RecentForm'],
        'TeamForm': last_race['TeamForm'],
        'DriverStability': last_race['DriverStability'],
        'CircuitOvertakeFactor': overtake_factor
    }

def predict():
    hgb, rf, mlp, le_driver, le_circuit, le_team, le_compound, scaler, df_history = load_artifacts()
    
    print("\n" + "="*50)
    print("🏎️  F1 PREDICTOR V3 (HistGradientBoosting Mode)")
    print("="*50)

    # Input base
    driver_in = input(f"Codice Pilota (es. VER, HAM, LEC): ").upper()
    if driver_in not in le_driver.classes_:
        print(f"❌ Pilota '{driver_in}' non trovato nel database.")
        return

    circuit_in = input(f"Circuito (es. Italian Grand Prix): ")
    if circuit_in not in le_circuit.classes_:
        print(f"❌ Circuito '{circuit_in}' non trovato.")
        return

    try:
        grid_pos = float(input("Posizione in Griglia (1-20): "))
        temp = float(input("Temperatura Aria prevista (°C): "))
        humidity = float(input("Umidità prevista (%): "))
        is_wet = int(input("Gara bagnata? (1=Sì, 0=No): "))
    except ValueError:
        print("❌ Inserimento non valido.")
        return

    # Recupero feature avanzate dal passato
    hist = get_historical_features(df_history, driver_in, circuit_in)
    if not hist:
        print(f"❌ Impossibile recuperare dati storici per {driver_in}.")
        return

    # Preparazione feature per modelli
    # ['Driver_Encoded', 'Team_Encoded', 'Circuit_Encoded', 'Compound_Label',
    #  'GridPosition', 'RecentForm', 'TeamForm', 'DriverStability', 
    #  'CircuitOvertakeFactor', 'TeammateDiff', 'OutPosition', 'TrackGrip', 'is_wet']
    
    driver_enc = le_driver.transform([driver_in])[0]
    team_enc = le_team.transform([hist['Team']])[0]
    circuit_enc = le_circuit.transform([circuit_in])[0]
    
    # Assumiamo Soft come compound di default (label 0 o simile, recuperiamo dalla classe)
    compound_label = 0 # Default

    # Calcolo feature derivate V3
    teammate_diff = 0 # In assenza di info sul compagno nel weekend futuro, usiamo 0 (neutro)
    out_position = grid_pos - hist['RecentForm']
    track_grip = temp * (100 - humidity) / 100

    features_list = [
        driver_enc, team_enc, circuit_enc, compound_label,
        grid_pos, hist['RecentForm'], hist['TeamForm'], hist['DriverStability'],
        hist['CircuitOvertakeFactor'], teammate_diff, out_position, track_grip, is_wet
    ]

    X_new = pd.DataFrame([features_list], columns=[
        'Driver_Encoded', 'Team_Encoded', 'Circuit_Encoded', 'Compound_Label',
        'GridPosition', 'RecentForm', 'TeamForm', 'DriverStability', 
        'CircuitOvertakeFactor', 'TeammateDiff', 'OutPosition', 'TrackGrip', 'is_wet'
    ])

    # Predizione DELTA
    delta_hgb = hgb.predict(X_new)[0]
    delta_rf = rf.predict(X_new)[0]

    # Calcolo Posizione Finale (Grid + Delta)
    pos_hgb = np.clip(grid_pos + delta_hgb, 1, 20)
    pos_rf = np.clip(grid_pos + delta_rf, 1, 20)

    print("\n" + "-"*40)
    print(f"📊 ANALISI PER {driver_in} @ {circuit_in}")
    print(f"Partenza: {int(grid_pos)}° | Forma Recente: {hist['RecentForm']:.1f}")
    print("-"*40)
    
    print(f"🚀 HistGradientBoosting: Pos {pos_hgb:.2f} ({'guadagna' if delta_hgb < 0 else 'perde'} {abs(delta_hgb):.1f} pos)")
    print(f"🌲 Random Forest:        Pos {pos_rf:.2f}")
    print("-"*40)
    
    # Risultato finale consigliato (Media pesata o solo HGB visto il MAE 2.77)
    final_avg = (pos_hgb * 0.7 + pos_rf * 0.3)
    print(f"✨ PREVISIONE FINALE: Pos {int(round(final_avg))}")
    print("="*40 + "\n")

if __name__ == "__main__":
    predict()