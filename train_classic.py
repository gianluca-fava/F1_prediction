import pandas as pd
import numpy as np
import joblib
import sys
import os

# Configurazioni base
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler

def load_and_process_data():
    print("--- [CLASSIC] 1. CARICAMENTO DATI ---")
    import glob
    data_files = glob.glob(os.path.join('data', 'full_data', '*.parquet'))
    dfs = [] # list of dataset to concatenate all of them in 1 time
    for f in data_files:
        try:
            df_temp = pd.read_parquet(f)
            # Compatibility fix: raw files usually lack SessionType but are Race data
            if 'SessionType' not in df_temp.columns:
                df_temp['SessionType'] = 'R'
                
            dfs.append(df_temp) # add the df to the list of pieces of the df
            print(f"Caricato: {os.path.basename(f)} ({len(df_temp)} righe)")
        except FileNotFoundError:
            continue
    
    if not dfs:
        print("ERRORE: Nessun file dati trovato.")
        sys.exit(1) #stop if list is void
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    df = df[df['SessionType'] == 'R']
    
    # Aggregazione
    grouped = df.groupby(['Year', 'EventName', 'Driver'])
    data = []
    for (year, event, driver), group in grouped:
        group = group.sort_values('LapNumber')

        team = group['Team'].iloc[0] if 'Team' in group.columns else 'Unknown'

        # TOTAL LAPS OF THE RACE
        #Max lap number on the dataset for that race
        total_race_laps = df[(df['Year'] == year) & (df['EventName'] == event)]['LapNumber'].max()
        laps_completed = group['LapNumber'].max()
        is_classified = laps_completed >= (total_race_laps * 0.9)

        # START POSITION
        # if the starting pos is nan => the pilot has not started the race => we use 0 to say DNS (do not started)
        start_pos_raw = group.iloc[0]['Position']
        grid_pos = start_pos_raw if pd.notna(start_pos_raw) else 0
        if grid_pos == 0:
            print(f"🚩 [DNS] {driver} @ {event} ({year})")

        # DNF (Do Not Finish) - Last lap = NaN OR he hasn't completed 90% of the race laps => DNF
        final_pos_raw = group.iloc[-1]['Position']
        dnf_flag = 0
        if pd.isna(final_pos_raw) or not is_classified:
            final_pos = 21
            dnf_flag = 1
            print(f"🏳️ [DNF] {driver} @ {event} ({year})")
        else:
            final_pos = final_pos_raw

        # COMPOUND
        valid_compounds = group['Compound'].dropna()
        start_compound = valid_compounds.iloc[0] if not valid_compounds.empty else 'Unknown'

        # WEATHER
        humidity = group['AvgHumidity'].mean()
        temperature = group['AvgTrackTemp'].mean()

        data.append({
            'Year': year,
            'Driver': driver,
            'Team': team,
            'Circuit': event,
            'startCompound': start_compound,
            'GridPosition': grid_pos,
            'Humidity': humidity,
            'Temperature': temperature,
            'FinalPosition': final_pos,
            'is_DNF': dnf_flag
        })

    return pd.DataFrame(data)

def main():
    processed_data_path = 'f1_dataset_processed.parquet'

    # 1. CONTROLLO ESISTENZA DATASET ELABORATO
    if os.path.exists(processed_data_path):
        print(f"📦 Caricamento dataset elaborato da {processed_data_path}...")
        df = pd.read_parquet(processed_data_path)
    else:
        print("🛠 Dataset non trovato. Avvio elaborazione completa...")
        # Caricamento raw
        df = load_and_process_data()

        #If the pilot is using a compound for rain at the start => the race is wet
        df['is_wet'] = df['startCompound'].apply(lambda x: 1 if x in ['INTERMEDIATE', 'WET'] else 0)

        # --- RECENT FORM & STABILITY ---
        print("RecentForm (EWMA) and Stability calculation...")
        df = df.sort_values(by=['Year', 'Circuit', 'Team', 'Driver'])
        # EWMA: pesa di più l'ultima gara rispetto a 3 gare fa
        df['RecentForm'] = df.groupby('Driver')['FinalPosition'].transform(
            lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
        )
        # TeamForm: media degli ultimi 2 GP (4 auto)
        df['TeamForm'] = df.groupby('Team')['FinalPosition'].transform(
            lambda x: x.shift(2).rolling(window=4, min_periods=1).mean()
        )
        # DriverStability: deviazione standard delle ultime 5 gare
        df['DriverStability'] = df.groupby('Driver')['FinalPosition'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).std()
        ).fillna(0)

        # Fallback per valori mancanti
        df['RecentForm'] = df['RecentForm'].fillna(df['GridPosition'])
        df['TeamForm'] = df['TeamForm'].fillna(df['GridPosition'])

        # --- FEATURE ENGINEERING V3 ---
        print("Feature Engineering V3 (OvertakeFactor & TeammateDiff)...")
        # 1. Circuit Overtake Factor: variabilità storica delle posizioni in questo circuito
        # Circuiti con alta deviazione standard permettono più recuperi
        circuit_map = df.groupby('Circuit')['FinalPosition'].std().to_dict()
        df['CircuitOvertakeFactor'] = df['Circuit'].map(circuit_map).fillna(df['FinalPosition'].std())

        # 2. Teammate Difference: quanto il pilota è più in forma del compagno?
        # Confrontiamo la RecentForm del pilota con la media del team in quel weekend
        df['TeammateDiff'] = df.groupby(['Year', 'Circuit', 'Team'])['RecentForm'].transform(
            lambda x: x - x.mean()
        ).fillna(0)

        # 3. OutPosition: pilota veloce in griglia lenta (potenziale rimonta)
        # Se RecentForm (posizione media arrivo) < GridPosition => il pilota è "dietro" rispetto al suo valore
        df['OutPosition'] = df['GridPosition'] - df['RecentForm']

        # 4. Track Grip Interaction
        df['TrackGrip'] = df['Temperature'] * (100 - df['Humidity']) / 100

        df.to_parquet(processed_data_path, index=False)
        print(f"✅ Dataset salvato con successo in {processed_data_path}")

    print("\n--- [CLASSIC] 2. PREPARAZIONE DATASET ---")
    # --- TARGET ENCODING V4 ---
    print("Calcolo Target Encoding (Driver/Team Value)...")
    train_mask_enc = df['Year'] < 2025
    
    # Valore Pilota: media posizione finale storica (solo sul train per evitare leakage)
    driver_map = df[train_mask_enc].groupby('Driver')['FinalPosition'].mean().to_dict()
    df['Driver_Value'] = df['Driver'].map(driver_map).fillna(df['FinalPosition'].mean())
    
    # Valore Team: media posizione finale storica
    team_map = df[train_mask_enc].groupby('Team')['FinalPosition'].mean().to_dict()
    df['Team_Value'] = df['Team'].map(team_map).fillna(df['FinalPosition'].mean())

    le_circuit = LabelEncoder()
    df['Circuit_Encoded'] = le_circuit.fit_transform(df['Circuit'])
    le_compound = LabelEncoder()
    df['Compound_Label'] = le_compound.fit_transform(df['startCompound'])

    # --- 2. MODALITÀ PER MLP (One-Hot Encoding) ---
    df_dummies = pd.get_dummies(df['startCompound'], prefix='tyre')
    df = pd.concat([df, df_dummies], axis=1)

    # Save mapping per predizioni future
    joblib.dump(driver_map, 'driver_target_map.joblib')
    joblib.dump(team_map, 'team_target_map.joblib')
    joblib.dump(le_circuit, 'le_circuit.joblib')
    joblib.dump(le_compound, 'le_compound.joblib')

    features_rf = ['Driver_Value', 'Team_Value', 'Circuit_Encoded', 'Compound_Label',
                   'GridPosition', 'RecentForm', 'TeamForm', 'DriverStability', 
                   'CircuitOvertakeFactor', 'TeammateDiff', 'OutPosition', 'TrackGrip', 'is_wet']
    
    features_mlp = features_rf + list(df_dummies.columns)

    # --- PREPARAZIONE TARGET (DELTA MODE) ---
    # Prediciamo lo spostamento rispetto alla griglia: Final - Grid
    # Es: Parte 5°, arriva 3° => Delta = -2 (guadagna posizioni)
    # Es: Parte 2°, arriva 10° => Delta = +8 (perde posizioni)
    df['PositionDelta'] = df['FinalPosition'] - df['GridPosition']
    target = 'PositionDelta'

    # Split
    train_mask = (df['Year'] >= 2018) & (df['Year'] <= 2024)
    test_mask = (df['Year'] == 2025)

    # X e Y
    X_train_rf = df.loc[train_mask, features_rf]
    X_test_rf = df.loc[test_mask, features_rf]
    X_train_mlp = df.loc[train_mask, features_mlp]
    X_test_mlp = df.loc[test_mask, features_mlp]
    y_train = df.loc[train_mask, target]
    y_test_abs = df.loc[test_mask, 'FinalPosition'] # Per valutazione finale
    y_test_delta = df.loc[test_mask, target]
    grid_test = df.loc[test_mask, 'GridPosition']

    # --- SCALING (only MLP!) ---
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_mlp)
    X_test_mlp_scaled = scaler.transform(X_test_mlp)

    # --- TRAINING ---
    print("\n--- [CLASSIC] 3. ADDESTRAMENTO DIFFERENZIATO ---")

    # 1. DNF Classifier (Nuovo!)
    print("Training DNF Classifier...")
    dnf_model = HistGradientBoostingClassifier(random_state=42)
    dnf_model.fit(X_train_rf, df.loc[train_mask, 'is_DNF'])
    joblib.dump(dnf_model, 'f1_dnf_classifier.joblib')

    # 2. HistGradientBoosting (Regressore Delta)
    print("Training HistGradientBoosting Regressor...")
    hgb = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=500,
        learning_rate=0.03,
        max_depth=12,
        l2_regularization=0.5,
        random_state=42
    )
    hgb.fit(X_train_rf, y_train)
    joblib.dump(hgb, 'f1_hgb_model.joblib')

    # 3. Random Forest
    print("Training RandomForest...")
    rf = RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_leaf=7, random_state=42, n_jobs=-1)
    rf.fit(X_train_rf, y_train)
    joblib.dump(rf, 'f1_rf_best_model.joblib')

    # 4. MLP
    print("Training MLP (Neural)...")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='tanh', alpha=0.1, max_iter=5000, early_stopping=True, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    joblib.dump(mlp, 'f1_mlp_model.joblib')

    # --- EVALUATION ---
    print("\n--- [CLASSIC] RISULTATI TEST SET (2025) - TARGET: DELTA ---")

    # Predizioni Delta pure
    hgb_delta = hgb.predict(X_test_rf)
    rf_delta = rf.predict(X_test_rf)
    mlp_delta = mlp.predict(X_test_mlp_scaled)

    # Posizioni Finali (senza "sporcarle" con la probabilità di DNF)
    hgb_pred = np.clip(grid_test + hgb_delta, 1, 21)
    rf_pred = np.clip(grid_test + rf_delta, 1, 21)
    mlp_pred = np.clip(grid_test + mlp_delta, 1, 21)

    # Metriche
    mae_hgb = mean_absolute_error(y_test_abs, hgb_pred)
    mae_rf = mean_absolute_error(y_test_abs, rf_pred)
    mae_mlp = mean_absolute_error(y_test_abs, mlp_pred)

    r2_hgb = r2_score(y_test_abs, hgb_pred)
    r2_rf = r2_score(y_test_abs, rf_pred)
    r2_mlp = r2_score(y_test_abs, mlp_pred)

    print(f"{'METRICA':<20} | {'HGB (V4)':<12} | {'RF':<12} | {'MLP':<12}")
    print("-" * 65)
    print(f"{'R2 Score':<20} | {r2_hgb:<12.4f} | {r2_rf:<12.4f} | {r2_mlp:<12.4f}")
    print(f"{'MAE (Posizioni)':<20} | {mae_hgb:<12.2f} | {mae_rf:<12.2f} | {mae_mlp:<12.2f}")

    print("\n" + "-" * 40)
    print(f"🚀 MIGLIOR MODELLO: {min([('HGB', mae_hgb), ('RF', mae_rf), ('MLP', mae_mlp)], key=lambda x: x[1])[0]}")
    print("-" * 40)

    if hasattr(rf, 'feature_importances_'):
        print("\nIMPORTANZA FEATURE (Random Forest):")
        feat_imp = sorted(zip(features_rf, rf.feature_importances_), key=lambda x: x[1], reverse=True)
        for name, imp in feat_imp:
            print(f"🔹 {name:<20}: {imp:.4f}")


if __name__ == "__main__":
    main()
