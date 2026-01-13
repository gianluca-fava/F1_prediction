import os
import pandas as pd
import numpy as np
import joblib
import sys
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Configurazioni base
os.environ['OMP_NUM_THREADS'] = '1'
MODEL_DIR = 'joblib'
DATA_PATH = 'f1_dataset_processed.parquet'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_raw_data():
    print("📂 Caricamento file parquet raw...")
    data_files = glob.glob(os.path.join('data', 'full_data', '*.parquet'))
    if not data_files:
        print("❌ ERRORE: Nessun file trovato in data/full_data/")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in data_files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return df[df['SessionType'] == 'R']

def process_race_results(df):
    # Aggregazione
    grouped = df.groupby(['Year', 'EventName', 'Driver'])
    data = []
    for (year, event, driver), group in grouped:
        group = group.sort_values('LapNumber')

        # TOTAL LAPS OF THE RACE - Max lap number on the dataset for that race
        total_race_laps = df[(df['Year'] == year) & (df['EventName'] == event)]['LapNumber'].max()
        laps_completed = group['LapNumber'].max()
        is_classified = laps_completed >= (total_race_laps * 0.9)

        # DNS - if the starting pos is nan => the pilot has not started the race => we use 0 to say DNS (do not started)
        grid_pos = group.iloc[0]['Position'] if pd.notna(group.iloc[0]['Position']) else 0

        # DNF (Do Not Finish) - Last lap = NaN OR he hasn't completed 90% of the race laps => DNF
        final_pos_raw = group.iloc[-1]['Position']
        final_pos = final_pos_raw if (pd.notna(final_pos_raw) and is_classified) else 21

        # TYRE COMPOUND
        start_compound = group['Compound'].dropna().iloc[0] if not group['Compound'].dropna().empty else 'Unknown'

        # WETHER
        humidity = group['AvgHumidity'].mean()  # average humidity for all the race
        temperature = group['AvgTrackTemp'].mean()  # average temperature for all the race

        # TEAM
        team = group['Team'].iloc[0] if 'Team' in group.columns else 'Unknown'

        data.append({
            'Year': year,
            'Driver': driver,
            'Team': team,
            'Circuit': event,
            'startCompound': start_compound,
            'GridPosition': grid_pos,
            'Humidity': humidity,
            'Temperature': temperature,
            'FinalPosition': final_pos
        })

    return pd.DataFrame(data)


def compute_f1_elo(df, k_factor=10):
    print("📊 Calcolo Elo Rating dinamico...")
    df = df.sort_values(by=['Year', 'Circuit']).reset_index(drop=True) #ordinamento cronologico
    unique_drivers = df['Driver'].unique() # Inizializziamo i rating (tutti a 1500)
    current_ratings = {driver: 1500.0 for driver in unique_drivers}
    elo_before_race = []

    # 3. Iteriamo per ogni gara
    for (year, circuit), race_df in df.groupby(['Year', 'Circuit'], sort=False):
        for driver in race_df['Driver']:
            elo_before_race.append(current_ratings[driver])

        # Update ratings in batch per la gara
        new_ratings = current_ratings.copy()

        drivers_list = race_df.to_dict('records')
        new_ratings = current_ratings.copy()
        for i, p1 in enumerate(drivers_list):
            for p2 in drivers_list[i + 1:]:
                r1, r2 = current_ratings[p1['Driver']], current_ratings[p2['Driver']]
                e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                s1 = 1 if p1['FinalPosition'] < p2['FinalPosition'] else (
                    0 if p1['FinalPosition'] > p2['FinalPosition'] else 0.5)

                change = k_factor * (s1 - e1)
                new_ratings[p1['Driver']] += change
                new_ratings[p2['Driver']] -= change
        current_ratings = new_ratings
    return elo_before_race


def add_features(df):
    print("📈 Calcolo RecentForm e TeamForm...")
    # Is Wet logic
    wet_pattern = 'INTERMEDIATE|WET'
    df['is_wet'] = df.groupby(['Year', 'Circuit'])['startCompound'].transform(
        lambda x: (x.str.contains(wet_pattern, case=False, na=False).sum() >= 2).astype(int)
    )

    # Recent Form logic
    df = df.sort_values(by=['Year', 'Circuit', 'Team', 'Driver'])
    df['RecentForm'] = df.groupby('Driver')['FinalPosition'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(df['GridPosition'])

    df['Driver_Elo'] = compute_f1_elo(df)
    return df


def train_model_pipeline(df):
    """Accorpa preparazione feature, encoding e addestramento."""
    print("\n--- [1. PREPARAZIONE FEATURE & ENCODING] ---")

    # Inizializzazione e Fit Encoders
    le_team = LabelEncoder()
    le_compound = LabelEncoder()

    df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    df['Compound_Label'] = le_compound.fit_transform(df['startCompound'])

    # Salvataggio Encoders
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(le_team, os.path.join(MODEL_DIR, 'le_team.joblib'))
    joblib.dump(le_compound, os.path.join(MODEL_DIR, 'le_compound.joblib'))

    # Selezione Feature
    features_rf = ['GridPosition', 'RecentForm', 'Team_Encoded', 'Compound_Label', 'Driver_Elo', 'is_wet']
    target = 'FinalPosition'

    # Split Temporale (Train < 2025, Test == 2025)
    train_mask = df['Year'] < 2025
    test_mask = df['Year'] == 2025

    X_train = df.loc[train_mask, features_rf]
    y_train = df.loc[train_mask, target]
    X_test = df.loc[test_mask, features_rf]
    y_test = df.loc[test_mask, target]

    print(f"🚀 [2. TRAINING] Avvio RandomForest su {len(features_rf)} feature...")
    rf = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=15,
        min_samples_leaf=10,
        max_features=0.9,
        max_depth=25,
        ccp_alpha=0.1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Salvataggio Modello
    model_path = os.path.join(MODEL_DIR, 'f1_rf_best_model.joblib')
    joblib.dump(rf, model_path)
    print(f"✅ Modello salvato in: {model_path}")

    return rf, X_test, y_test, features_rf


def evaluate_model(model, X_test, y_test, feature_names):
    """Valutazione delle performance e Permutation Importance."""
    print("\n--- [3. VALUTAZIONE] ---")

    # Predizione con Clipping e Rounding
    raw_preds = model.predict(X_test)
    preds = np.round(np.clip(raw_preds, 1, 21))

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=" * 45)
    print(f"MAE:      {mae:.4f} posizioni")
    print(f"R2 Score: {r2:.4f}")
    print("=" * 45)

    print("\n📊 Calcolo Permutation Importance...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    sorted_idx = perm.importances_mean.argsort()[::-1]
    for i in sorted_idx:
        print(f"{feature_names[i]:<20} | {perm.importances_mean[i]:.4f} +/- {perm.importances_std[i]:.4f}")


def main():
    # --- GESTIONE DATASET ---
    if os.path.exists(DATA_PATH):
        print(f"📦 Caricamento dataset esistente: {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)
    else:
        print("🛠 Dataset non trovato. Avvio elaborazione completa...")
        # Assicurati che load_raw_data, process_race_results e add_features siano importate/definite
        df = load_raw_data()
        df = process_race_results(df)
        df = add_features(df)
        df.to_parquet(DATA_PATH, index=False)
        print(f"✅ Dataset salvato con successo.")

    # --- PIPELINE DI TRAINING ---
    # Accorpa preparazione dati, encoding e addestramento
    rf_model, X_test, y_test, features_rf = train_model_pipeline(df)

    # --- VALUTAZIONE ---
    evaluate_model(rf_model, X_test, y_test, features_rf)

if __name__ == "__main__":
    main()