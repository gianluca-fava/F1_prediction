


import os
import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV


# Configurazioni base
os.environ['OMP_NUM_THREADS'] = '1'

def load_and_process_data():
    print("--- [CLASSIC] 1. CARICAMENTO DATI ---")
    import glob
    data_files = glob.glob(os.path.join('data', 'full_data', '*.parquet'))
    dfs = []  # list of dataset to concatenate all of them in 1 time
    for f in data_files:
        try:
            df_temp = pd.read_parquet(f)
            # Compatibility fix: raw files usually lack SessionType but are Race data
            if 'SessionType' not in df_temp.columns:
                df_temp['SessionType'] = 'R'

            dfs.append(df_temp)  # add the df to the list of pieces of the df
            print(f"Caricato: {os.path.basename(f)} ({len(df_temp)} righe)")
        except FileNotFoundError:
            continue

    if not dfs:
        print("ERRORE: Nessun file dati trovato.")
        sys.exit(1)  # stop if list is void

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
        # Max lap number on the dataset for that race
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
        if pd.isna(final_pos_raw) or not is_classified:
            final_pos = 21
            print(f"🏳️ [DNF] {driver} @ {event} ({year})")
        else:
            final_pos = final_pos_raw

        # COMPOUND
        valid_compounds = group['Compound'].dropna()
        if not valid_compounds.empty:
            # first compound registred (lap 1 or 2)
            start_compound = valid_compounds.iloc[0]
        else:
            start_compound = 'Unknown'
            print(f"⚠️ [WARN] COMPOUND NOT FOUND {driver} @ {event} ({year})")

        # WETHER
        humidity = group['AvgHumidity'].mean()  # average humidity for all the race
        temperature = group['AvgTrackTemp'].mean()  # average temperature for all the race

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
    # 1. Ordiniamo cronologicamente
    df = df.sort_values(by=['Year', 'Circuit']).reset_index(drop=True)

    # 2. Inizializziamo i rating (tutti a 1500)
    unique_drivers = df['Driver'].unique()
    current_ratings = {driver: 1500.0 for driver in unique_drivers}

    # Lista per memorizzare i rating PRIMA della gara (quelli che userà il modello)
    elo_before_race = []

    # 3. Iteriamo per ogni gara
    grouped = df.groupby(['Year', 'Circuit'], sort=False)

    for (year, circuit), race_df in grouped:
        # Memorizziamo i rating attuali per questa gara
        race_drivers = race_df['Driver'].tolist()
        for driver in race_drivers:
            elo_before_race.append(current_ratings[driver])

        # Aggiorniamo i rating basandoci sui risultati della gara
        # Confrontiamo ogni coppia di piloti nella gara
        new_ratings = current_ratings.copy()

        drivers_list = race_df.to_dict('records')
        for i in range(len(drivers_list)):
            for j in range(i + 1, len(drivers_list)):
                p1 = drivers_list[i]
                p2 = drivers_list[j]

                d1_name = p1['Driver']
                d2_name = p2['Driver']

                # Rating attuali
                r1 = current_ratings[d1_name]
                r2 = current_ratings[d2_name]

                # Punteggio atteso
                e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                e2 = 1 - e1

                # Risultato reale (S=1 se arrivi prima, S=0.5 se pari, S=0 se dopo)
                if p1['FinalPosition'] < p2['FinalPosition']:
                    s1, s2 = 1, 0
                elif p1['FinalPosition'] > p2['FinalPosition']:
                    s1, s2 = 0, 1
                else:
                    s1, s2 = 0.5, 0.5

                # Update dei rating temporanei (accumuliamo i cambiamenti)
                new_ratings[d1_name] += k_factor * (s1 - e1)
                new_ratings[d2_name] += k_factor * (s2 - e2)

        # Applichiamo i nuovi rating per la prossima gara
        current_ratings = new_ratings

    return elo_before_race



'''def apply_advanced_form(df):
    print("📈 Calcolo della RecentDeltaForm (Weighted Position Gain)...")
    df = df.sort_values(by=['Driver', 'Year', 'Circuit'])

    # Delta: quante posizioni ha recuperato (positivo = rimonta, negativo = crollo)
    df['PosDelta'] = df['GridPosition'] - df['FinalPosition']

    def weighted_average(x):
        if len(x) == 0: return 0
        weights = np.array([0.5, 0.3, 0.2])[:len(x)]
        weights = weights / weights.sum()
        return np.sum(x * weights)

    # Applichiamo la media pesata sulle ultime 3 gare (esclusa la corrente)
    df['RecentDeltaForm'] = df.groupby('Driver')['PosDelta'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).apply(weighted_average)
    )
    return df.fillna({'RecentDeltaForm': 0})
'''
def main():
    processed_data_path = 'f1_dataset_processed.parquet'

    # --- 1. CARICAMENTO E PRE-PROCESSING ---
    if os.path.exists(processed_data_path):
        print(f"📦 Caricamento dataset elaborato da {processed_data_path}...")
        df = pd.read_parquet(processed_data_path)
    else:
        print("🛠 Dataset non trovato. Avvio elaborazione completa...")
        df = load_and_process_data()  # Assicurati che sia definita nel file

        # Logica IS_WET Full Race (Ottimizzata: almeno 2 piloti)
        wet_pattern = 'INTERMEDIATE|WET'
        df['driver_touched_wet'] = df['Compound'].str.contains(wet_pattern, case=False, na=False).astype(int)
        df['total_wet_drivers'] = df.groupby(['Year', 'Circuit'])['driver_touched_wet'].transform('sum')
        df['is_wet'] = (df['total_wet_drivers'] >= 2).astype(int)

        # --- RECENT FORM ---
        print("RecentForm calcoulation...")
        df = df.sort_values(by=['Year', 'Circuit', 'Team', 'Driver'])
        #mean on the last 3 races for the pilot (not the actual race at the moment)
        df['RecentForm'] = df.groupby('Driver')['FinalPosition'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        # TeamForm: mean on the 2 pilots on the last 2 racese (4 cars)
        df['TeamForm'] = df.groupby('Team')['FinalPosition'].transform(
        lambda x: x.shift(2).rolling(window=4, min_periods=1).mean()
        )
        # Fallback per valori mancanti
        df['RecentForm'] = df['RecentForm'].fillna(df['GridPosition'])
        df['TeamForm'] = df['TeamForm'].fillna(df['GridPosition'])


    print("\n--- [RF OPTIMIZED] 2. PREPARAZIONE DATASET ---")


    '''
    # Calcolo RecentForm (Media ultime 3 gare)
    df = apply_advanced_form(df)
    df['Driver_Elo'] = compute_f1_elo(df, k_factor=2)
    '''

    # --- 1. CALCOLO ELO RATING (Sostituisce Driver_Encoded) ---
    print("\n📊 Calcolo Elo Rating per i piloti...")
    df['Driver_Elo'] = compute_f1_elo(df, k_factor=2)  # K basso per stabilità in F1

    # Encoding variabili categoriche per RF
    le_team = LabelEncoder()
    df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    le_compound = LabelEncoder()
    df['Compound_Label'] = le_compound.fit_transform(df['startCompound'])

    # --- FEATURE SELECTION (Basata sui test precedenti) ---
    # Rimosse feature che causano rumore: Driver_Encoded, Humidity, Circuit_Encoded
    features_rf = [
        'GridPosition',
        'RecentForm',
        'Team_Encoded',
        'Compound_Label',
        'Driver_Elo',
        'is_wet'    ]

    target = 'FinalPosition'

    # Split temporale (Train: fino al 2024, Test: 2025)
    train_mask = (df['Year'] >= 2018) & (df['Year'] <= 2024)
    test_mask = (df['Year'] == 2025)

    X_train_rf = df.loc[train_mask, features_rf]
    X_test_rf = df.loc[test_mask, features_rf]
    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    # --- 3. TRAINING RANDOM FOREST ---
    print(f"\n🚀 Training RandomForest con {len(features_rf)} feature...")
    '''
    param_dist = {
        'n_estimators': [500, 800, 900, 1000, 1100, 1200, 1500, 1800],  # Numero di alberi (stabilità)
        'max_depth': [10, 15, 20, 25, 30, None],  # Profondità (capacità di apprendimento)
        'min_samples_leaf': [5, 10, 15, 20, 25],  # Foglie (controllo overfitting)
        'min_samples_split': [5, 10, 15, 20, 25, 30],  # Split (precisione dei rami)
        'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9],  # Feature per albero (biodiversità)
        'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'bootstrap': [True]
    }
    rf = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,  # Ridotto per velocità, puoi aumentarlo
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train_rf, y_train)
    best_params = rf.best_params_
    print(f"\n🏆 Migliori parametri con CCP: {best_params}")
    joblib.dump(rf, 'f1_rf_best_model.joblib')
    '''
    rf = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=15,
        min_samples_leaf=10,
        max_features=0.9,
        max_depth=25,
        ccp_alpha=0.1,
        bootstrap=True)
    rf.fit(X_train_rf, y_train)


import os
import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV


# Configurazioni base
os.environ['OMP_NUM_THREADS'] = '1'

def load_and_process_data():
    print("--- [CLASSIC] 1. CARICAMENTO DATI ---")
    import glob
    data_files = glob.glob(os.path.join('data', 'full_data', '*.parquet'))
    dfs = []  # list of dataset to concatenate all of them in 1 time
    for f in data_files:
        try:
            df_temp = pd.read_parquet(f)
            # Compatibility fix: raw files usually lack SessionType but are Race data
            if 'SessionType' not in df_temp.columns:
                df_temp['SessionType'] = 'R'

            dfs.append(df_temp)  # add the df to the list of pieces of the df
            print(f"Caricato: {os.path.basename(f)} ({len(df_temp)} righe)")
        except FileNotFoundError:
            continue

    if not dfs:
        print("ERRORE: Nessun file dati trovato.")
        sys.exit(1)  # stop if list is void

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
        # Max lap number on the dataset for that race
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
        if pd.isna(final_pos_raw) or not is_classified:
            final_pos = 21
            print(f"🏳️ [DNF] {driver} @ {event} ({year})")
        else:
            final_pos = final_pos_raw

        # COMPOUND
        valid_compounds = group['Compound'].dropna()
        if not valid_compounds.empty:
            # first compound registred (lap 1 or 2)
            start_compound = valid_compounds.iloc[0]
        else:
            start_compound = 'Unknown'
            print(f"⚠️ [WARN] COMPOUND NOT FOUND {driver} @ {event} ({year})")

        # WETHER
        humidity = group['AvgHumidity'].mean()  # average humidity for all the race
        temperature = group['AvgTrackTemp'].mean()  # average temperature for all the race

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
    # 1. Ordiniamo cronologicamente
    df = df.sort_values(by=['Year', 'Circuit']).reset_index(drop=True)

    # 2. Inizializziamo i rating (tutti a 1500)
    unique_drivers = df['Driver'].unique()
    current_ratings = {driver: 1500.0 for driver in unique_drivers}

    # Lista per memorizzare i rating PRIMA della gara (quelli che userà il modello)
    elo_before_race = []

    # 3. Iteriamo per ogni gara
    grouped = df.groupby(['Year', 'Circuit'], sort=False)

    for (year, circuit), race_df in grouped:
        # Memorizziamo i rating attuali per questa gara
        race_drivers = race_df['Driver'].tolist()
        for driver in race_drivers:
            elo_before_race.append(current_ratings[driver])

        # Aggiorniamo i rating basandoci sui risultati della gara
        # Confrontiamo ogni coppia di piloti nella gara
        new_ratings = current_ratings.copy()

        drivers_list = race_df.to_dict('records')
        for i in range(len(drivers_list)):
            for j in range(i + 1, len(drivers_list)):
                p1 = drivers_list[i]
                p2 = drivers_list[j]

                d1_name = p1['Driver']
                d2_name = p2['Driver']

                # Rating attuali
                r1 = current_ratings[d1_name]
                r2 = current_ratings[d2_name]

                # Punteggio atteso
                e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                e2 = 1 - e1

                # Risultato reale (S=1 se arrivi prima, S=0.5 se pari, S=0 se dopo)
                if p1['FinalPosition'] < p2['FinalPosition']:
                    s1, s2 = 1, 0
                elif p1['FinalPosition'] > p2['FinalPosition']:
                    s1, s2 = 0, 1
                else:
                    s1, s2 = 0.5, 0.5

                # Update dei rating temporanei (accumuliamo i cambiamenti)
                new_ratings[d1_name] += k_factor * (s1 - e1)
                new_ratings[d2_name] += k_factor * (s2 - e2)

        # Applichiamo i nuovi rating per la prossima gara
        current_ratings = new_ratings

    return elo_before_race



'''def apply_advanced_form(df):
    print("📈 Calcolo della RecentDeltaForm (Weighted Position Gain)...")
    df = df.sort_values(by=['Driver', 'Year', 'Circuit'])

    # Delta: quante posizioni ha recuperato (positivo = rimonta, negativo = crollo)
    df['PosDelta'] = df['GridPosition'] - df['FinalPosition']

    def weighted_average(x):
        if len(x) == 0: return 0
        weights = np.array([0.5, 0.3, 0.2])[:len(x)]
        weights = weights / weights.sum()
        return np.sum(x * weights)

    # Applichiamo la media pesata sulle ultime 3 gare (esclusa la corrente)
    df['RecentDeltaForm'] = df.groupby('Driver')['PosDelta'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).apply(weighted_average)
    )
    return df.fillna({'RecentDeltaForm': 0})
'''
def main():
    processed_data_path = 'f1_dataset_processed.parquet'

    # --- 1. CARICAMENTO E PRE-PROCESSING ---
    if os.path.exists(processed_data_path):
        print(f"📦 Caricamento dataset elaborato da {processed_data_path}...")
        df = pd.read_parquet(processed_data_path)
    else:
        print("🛠 Dataset non trovato. Avvio elaborazione completa...")
        df = load_and_process_data()  # Assicurati che sia definita nel file

        # Logica IS_WET Full Race (Ottimizzata: almeno 2 piloti)
        wet_pattern = 'INTERMEDIATE|WET'
        df['driver_touched_wet'] = df['Compound'].str.contains(wet_pattern, case=False, na=False).astype(int)
        df['total_wet_drivers'] = df.groupby(['Year', 'Circuit'])['driver_touched_wet'].transform('sum')
        df['is_wet'] = (df['total_wet_drivers'] >= 2).astype(int)

        # --- RECENT FORM ---
        print("RecentForm calcoulation...")
        df = df.sort_values(by=['Year', 'Circuit', 'Team', 'Driver'])
        #mean on the last 3 races for the pilot (not the actual race at the moment)
        df['RecentForm'] = df.groupby('Driver')['FinalPosition'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        # TeamForm: mean on the 2 pilots on the last 2 racese (4 cars)
        df['TeamForm'] = df.groupby('Team')['FinalPosition'].transform(
            lambda x: x.shift(2).rolling(window=4, min_periods=1).mean()
        )
        # Fallback per valori mancanti
        df['RecentForm'] = df['RecentForm'].fillna(df['GridPosition'])
        df['TeamForm'] = df['TeamForm'].fillna(df['GridPosition'])


    print("\n--- [RF OPTIMIZED] 2. PREPARAZIONE DATASET ---")


    '''
    # Calcolo RecentForm (Media ultime 3 gare)
    df = apply_advanced_form(df)
    df['Driver_Elo'] = compute_f1_elo(df, k_factor=2)
    '''

    # --- 1. CALCOLO ELO RATING (Sostituisce Driver_Encoded) ---
    print("\n📊 Calcolo Elo Rating per i piloti...")
    df['Driver_Elo'] = compute_f1_elo(df, k_factor=2)  # K basso per stabilità in F1

    # Encoding variabili categoriche per RF
    le_team = LabelEncoder()
    df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    le_compound = LabelEncoder()
    df['Compound_Label'] = le_compound.fit_transform(df['startCompound'])

    # --- FEATURE SELECTION (Basata sui test precedenti) ---
    # Rimosse feature che causano rumore: Driver_Encoded, Humidity, Circuit_Encoded
    features_rf = [
        'GridPosition',
        'RecentForm',
        'TeamForm',
        'Compound_Label',
        'Driver_Elo',
        'is_wet'
    ]

    target = 'FinalPosition'

    # Split temporale (Train: fino al 2024, Test: 2025)
    train_mask = (df['Year'] >= 2018) & (df['Year'] <= 2024)
    test_mask = (df['Year'] == 2025)

    X_train_rf = df.loc[train_mask, features_rf]
    X_test_rf = df.loc[test_mask, features_rf]
    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    # --- 3. TRAINING RANDOM FOREST ---
    print(f"\n🚀 Training RandomForest con {len(features_rf)} feature...")
    '''
    param_dist = {
        'n_estimators': [500, 800, 900, 1000, 1100, 1200, 1500, 1800],  # Numero di alberi (stabilità)
        'max_depth': [10, 15, 20, 25, 30, None],  # Profondità (capacità di apprendimento)
        'min_samples_leaf': [5, 10, 15, 20, 25],  # Foglie (controllo overfitting)
        'min_samples_split': [5, 10, 15, 20, 25, 30],  # Split (precisione dei rami)
        'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9],  # Feature per albero (biodiversità)
        'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'bootstrap': [True]
    }
    rf = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,  # Ridotto per velocità, puoi aumentarlo
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train_rf, y_train)
    best_params = rf.best_params_
    print(f"\n🏆 Migliori parametri con CCP: {best_params}")
    joblib.dump(rf, 'f1_rf_best_model.joblib')
    '''
    rf = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=15,
        min_samples_leaf=10,
        max_features=0.9,
        max_depth=25,
        ccp_alpha=0.1,
        bootstrap=True)
    rf.fit(X_train_rf, y_train)
    joblib.dump(rf, 'f1_rf_best_model.joblib')

    # --- 4. EVALUATION ---
    rf_pred = rf.predict(X_test_rf)

    # Arrotondamento e Clipping (F1 range 1-21)
    rf_pred = np.round(np.clip(rf_pred, 1, 21))

    mae_rf = mean_absolute_error(y_test, rf_pred)
    r2_rf = r2_score(y_test, rf_pred)

    print("\n" + "=" * 45)
    print(f"🔹 RISULTATI RANDOM FOREST (TEST 2025)")
    print("-" * 45)
    print(f"MAE:      {mae_rf:.4f} posizioni")
    print(f"R2 Score: {r2_rf:.4f}")
    print("=" * 45)

    # --- 5. PERMUTATION IMPORTANCE ---
    print("\n📊 Calcolo Permutation Importance (Test Set)...")
    # n_repeats=10 garantisce una stabilità statistica per la tua tesi
    perm_result = permutation_importance(rf, X_test_rf, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Ordinamento e Visualizzazione
    sorted_idx = perm_result.importances_mean.argsort()[::-1]

    print("\n🏆 Feature Ranking (Permutation):")
    for i in sorted_idx:
        print(f"{features_rf[i]:<20} | {perm_result.importances_mean[i]:.4f} +/- {perm_result.importances_std[i]:.4f}")

    # Grafico a barre per l'importanza
    plt.figure(figsize=(10, 6))
    plt.barh([features_rf[i] for i in sorted_idx[::-1]], perm_result.importances_mean[sorted_idx[::-1]])
    plt.xlabel("Permutation Importance (MAE Decrease)")
    plt.title("Feature Importance - Random Forest (Generalization on 2025)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    joblib.dump(rf, 'f1_rf_best_model.joblib')

    # --- 4. EVALUATION ---
    rf_pred = rf.predict(X_test_rf)

    # Arrotondamento e Clipping (F1 range 1-21)
    rf_pred = np.round(np.clip(rf_pred, 1, 21))

    mae_rf = mean_absolute_error(y_test, rf_pred)
    r2_rf = r2_score(y_test, rf_pred)

    print("\n" + "=" * 45)
    print(f"🔹 RISULTATI RANDOM FOREST (TEST 2025)")
    print("-" * 45)
    print(f"MAE:      {mae_rf:.4f} posizioni")
    print(f"R2 Score: {r2_rf:.4f}")
    print("=" * 45)

    # --- 5. PERMUTATION IMPORTANCE ---
    print("\n📊 Calcolo Permutation Importance (Test Set)...")
    # n_repeats=10 garantisce una stabilità statistica per la tua tesi
    perm_result = permutation_importance(rf, X_test_rf, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Ordinamento e Visualizzazione
    sorted_idx = perm_result.importances_mean.argsort()[::-1]

    print("\n🏆 Feature Ranking (Permutation):")
    for i in sorted_idx:
        print(f"{features_rf[i]:<20} | {perm_result.importances_mean[i]:.4f} +/- {perm_result.importances_std[i]:.4f}")

    # Grafico a barre per l'importanza
    plt.figure(figsize=(10, 6))
    plt.barh([features_rf[i] for i in sorted_idx[::-1]], perm_result.importances_mean[sorted_idx[::-1]])
    plt.xlabel("Permutation Importance (MAE Decrease)")
    plt.title("Feature Importance - Random Forest (Generalization on 2025)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()