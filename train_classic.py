import eli5
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Configurazioni base
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import pandas as pd
from treeinterpreter import treeinterpreter as ti

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
        humidity = group['AvgHumidity'].mean() #average humidity for all the race
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
        wet_pattern = 'INTERMEDIATE|WET'
        df['driver_touched_wet'] = df['Compound'].str.contains(wet_pattern, case=False, na=False).astype(int)
        df['total_wet_drivers'] = df.groupby(['Year', 'Circuit'])['driver_touched_wet'].transform('sum')
        df['is_wet'] = (df['total_wet_drivers'] >= 2).astype(int)
        df.drop(columns=['driver_touched_wet', 'total_wet_drivers'], inplace=True)
        wet_race = df[df['is_wet'] == 1][['Year', 'Circuit']].drop_duplicates().shape[0]
        print(f"✅ Analisi completata. Gare bagnate trovate: {wet_race}")

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

        df.to_parquet(processed_data_path, index=False)
        print(f"✅ Dataset salvato con successo in {processed_data_path}")

    print("\n--- [CLASSIC] 2. PREPARAZIONE DATASET ---")
    le_driver = LabelEncoder()
    df['Driver_Encoded'] = le_driver.fit_transform(df['Driver'])
    le_circuit = LabelEncoder()
    df['Circuit_Encoded'] = le_circuit.fit_transform(df['Circuit'])
    le_team = LabelEncoder()
    df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    le_compound = LabelEncoder()
    df['Compound_Label'] = le_compound.fit_transform(df['startCompound'])

    # --- 2. MODALITÀ PER MLP (One-Hot Encoding) ---
    df_dummies = pd.get_dummies(df['startCompound'], prefix='tyre')
    df = pd.concat([df, df_dummies], axis=1)

    # Save encoders
    joblib.dump(le_driver, 'joblib/le_driver.joblib')
    joblib.dump(le_circuit, 'joblib/le_circuit.joblib')
    joblib.dump(le_team, 'joblib/le_team.joblib')
    joblib.dump(le_compound, 'joblib/le_compound.joblib')

    joblib.dump(list(df_dummies.columns), 'joblib/mlp_feature_cols.joblib') #column names One-Hot for prediction

    features_rf = ['Driver_Encoded', 'Team_Encoded', 'Circuit_Encoded', 'Compound_Label',
                   'GridPosition', 'Humidity', 'Temperature', 'RecentForm', 'is_wet']
    features_mlp = ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition',
                    'Humidity', 'Temperature', 'RecentForm', 'is_wet'] + list(df_dummies.columns)


    target = 'FinalPosition'

    # Split
    train_mask = (df['Year'] >= 2018) & (df['Year'] <= 2024)
    test_mask = (df['Year'] == 2025)

    # X per Random Forest
    X_train_rf = df.loc[train_mask, features_rf]
    X_test_rf = df.loc[test_mask, features_rf]

    # X per MLP
    X_train_mlp = df.loc[train_mask, features_mlp]
    X_test_mlp = df.loc[test_mask, features_mlp]

    # Y
    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    # --- SCALING (only MLP!) ---
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_mlp)
    X_test_mlp_scaled = scaler.transform(X_test_mlp)


    # --- TRAINING ---
    print("\n--- [CLASSIC] 3. ADDESTRAMENTO DIFFERENZIATO ---")

    # RF
    #{'n_estimators': 800, 'min_samples_split': 16, 'min_samples_leaf': 10, 'max_features': 0.95, 'max_depth': 27, 'bootstrap': True}

    '''
    print("Training RandomForest (Label Mode)...")
    param_dist = {
        'n_estimators': [800, 900, 1000, 1100, 1200, 1500, 1800, 2000],  # Numero di alberi (stabilità)
        'max_depth': [ 25, 26, 27, 28, 29, 30,  None],  # Profondità (capacità di apprendimento)
        'min_samples_leaf': [2, 5, 7, 10, 12, 15],  # Foglie (controllo overfitting)
        'min_samples_split': [5, 10, 15, 20, 25],  # Split (precisione dei rami)
        'max_features': ['sqrt', 'log2', 0.9, 0.97, 0.98, 0.99],  # Feature per albero (biodiversità)
        'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05, 0.08, 0.1, 0.2],
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
    #2,97 => {'n_estimators': 800, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 0.99, 'max_depth': 29, 'ccp_alpha': 0.08, 'bootstrap': True}
    rf = RandomForestRegressor(
        n_estimators=800,
        min_samples_split=15,
        min_samples_leaf=7,
        max_features=0.99,
        max_depth=29,
        ccp_alpha=0.08,
        bootstrap=True)
    rf.fit(X_train_rf, y_train)
    joblib.dump(rf, 'joblib/f1_rf_best_model.joblib')


    # MLP
    #Migliori parametri MLP trovati: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (256, 128, 64), 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam'}

    print("Training MLP (One-Hot Mode)...")
    mlp_param_grid = {
        # 1. Architetture: Esploriamo varianti attorno alla (256, 128, 64)
        'hidden_layer_sizes': [
            (256, 128, 64),      # Il tuo attuale "best"
            (128, 64, 32),       # Più snella (meno rischio overfitting)
            (256, 256, 128, 64), # Più profonda all'inizio
            (512, 256, 64),      # Più "potente" nello strato di input
        ],
        # 2. Regolarizzazione: Testiamo valori vicini allo 0.1
        'alpha': [0.05, 0.1, 0.15, 0.2], 
        
        # 3. Velocità di apprendimento: Testiamo la sensibilità attorno allo 0.0005
        'learning_rate_init': [0.0005, 0.0003, 0.001],
        
        # 4. Funzioni di attivazione
        'activation': ['relu', 'tanh'],
        
        # 5. Batch size: Può influenzare molto la convergenza (stabilità del gradiente)
        'batch_size': [32, 64]
    }

    # Inizializziamo il regressore con Early Stopping per evitare sprechi di tempo

    mlp_search = GridSearchCV(
        estimator=MLPRegressor(max_iter=5000, validation_fraction=0.1, n_iter_no_change=20, early_stopping=True, random_state=42),
        param_grid=mlp_param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',  # Il nostro obiettivo primario
        n_jobs=-1,
        verbose=2
    )
    mlp_search.fit(X_train_scaled, y_train)
    mlp = mlp_search.best_estimator_
    print(f"Migliori parametri MLP trovati: {mlp_search.best_params_}")
    joblib.dump(mlp, 'f1_mlp_model.joblib')

    '''
    #{'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (256, 128, 64), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001}
    #{'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (256, 128, 64), 'learning_rate': 'adaptive', 'learning_rate_init': 0.0005}
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),  # Struttura a imbuto più profonda
        activation='relu',  # ReLU va bene, ma alpha deve essere corretto
        solver='adam',
        alpha=0.1,  # Aumentato per combattere l'overfitting (L2 penalty)
        learning_rate='adaptive',  # Riduce il learning rate se il loss non scende
        learning_rate_init=0.0005,  # Partiamo più cauti (default era 0.001)
        max_iter=5000,
        early_stopping=True,  # Fondamentale: usa un set di validazione interno
        validation_fraction=0.1,  # 10% dei dati per decidere quando fermarsi
        n_iter_no_change=20,  # Pazienza prima di fermarsi
        random_state=42,
        batch_size=32
    )
    mlp.fit(X_train_scaled, y_train)
    print(f"MLP terminata dopo {mlp.n_iter_} epoche.") # verify if stopped early
    joblib.dump(mlp, 'joblib/f1_mlp_model.joblib')

    '''

    # --- EVALUATION DIFFERENZIATA ---
    print("\n--- [CLASSIC] RISULTATI TEST SET ---")

    # Predizioni differenziate
    rf_pred = rf.predict(X_test_rf)
    mlp_pred = mlp.predict(X_test_mlp_scaled)

    # Metriche per Random Forest
    mae_rf = mean_absolute_error(y_test, rf_pred)
    r2_rf = r2_score(y_test, rf_pred)

    # Metriche per MLP
    mae_mlp = mean_absolute_error(y_test, mlp_pred)
    r2_mlp = r2_score(y_test, mlp_pred)


    print(f"{'R2 Score (Fit)':<20} | {r2_rf:<15.4f} | {r2_mlp:<15.4f}")

    print("\n" + "-" * 40)
    print(f"🔹 RANDOM FOREST: Errore medio di {mae_rf:.2f} posizioni")
    print(f"🔹 MLP (NEURAL):  Errore medio di {mae_mlp:.2f} posizioni")
    print("-" * 40)

if __name__ == "__main__":
    main()
