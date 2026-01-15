import os
import pandas as pd
import numpy as np
import joblib
import sys
import glob
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# Base configurations
os.environ['OMP_NUM_THREADS'] = '1'
MODEL_DIR = 'joblib'
DATA_PATH = 'f1_dataset_processed.parquet'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_raw_data():
    print("Loading raw parquet files")
    data_files = glob.glob(os.path.join('data', 'full_data', '*.parquet'))
    if not data_files:
        print("Error: No files found in data/full_data/")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in data_files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return df[df['SessionType'] == 'R']

def process_race_results(df):
    # Aggregation
    grouped = df.groupby(['Year', 'EventName', 'Driver'])
    data = []
    for (year, event, driver), group in grouped:
        group = group.sort_values('LapNumber')

        # TOTAL LAPS OF THE RACE - Max lap number on the dataset for that race
        total_race_laps = df[(df['Year'] == year) & (df['EventName'] == event)]['LapNumber'].max()
        laps_completed = group['LapNumber'].max()
        is_classified = laps_completed >= (total_race_laps * 0.9)

        # DNS - if the starting pos is nan => the pilot has not started the race => we use 0 to say DNS (did not start)
        grid_pos = group.iloc[0]['Position'] if pd.notna(group.iloc[0]['Position']) else 0

        # DNF (Do Not Finish) - Last lap = NaN OR he hasn't completed 90% of the race laps => DNF
        final_pos_raw = group.iloc[-1]['Position']
        final_pos = final_pos_raw if (pd.notna(final_pos_raw) and is_classified) else 21

        # TYRE COMPOUND
        start_compound = group['Compound'].dropna().iloc[0] if not group['Compound'].dropna().empty else 'Unknown'

        # WEATHER
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
    print("Calculating dynamic Elo Rating")
    df = df.sort_values(by=['Year', 'Circuit']).reset_index(drop=True) # chronological order
    unique_drivers = df['Driver'].unique() # Initialize ratings (all at 1500)
    current_ratings = {driver: 1500.0 for driver in unique_drivers}
    elo_before_race = []

    # 3. Iterate for each race
    for (year, circuit), race_df in df.groupby(['Year', 'Circuit'], sort=False):
        for driver in race_df['Driver']:
            elo_before_race.append(current_ratings[driver])

        # Update ratings in batch for the race
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
    print("Calculating RecentForm and TeamForm")
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
    """Combines feature preparation, encoding, and training."""
    print("\n--- [1. FEATURE PREPARATION & ENCODING] ---")

    # --- ENCODING ---
    # Initialize all necessary encoders
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()
    le_compound = LabelEncoder()

    df['Driver_Encoded'] = le_driver.fit_transform(df['Driver'])
    df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    df['Circuit_Encoded'] = le_circuit.fit_transform(df['Circuit'])
    df['Compound_Label'] = le_compound.fit_transform(df['startCompound'])

    # Saving Encoders (ALL those needed for MLP and RF)
    for name, le in zip(['driver', 'team', 'circuit', 'compound'], [le_driver, le_team, le_circuit, le_compound]):
        joblib.dump(le, os.path.join(MODEL_DIR, f'le_{name}.joblib'))

    # Temporal Split (Train < 2025, Test == 2025)
    train_mask = df['Year'] < 2025
    test_mask = df['Year'] == 2025

    # Feature Selection
    features_rf = ['GridPosition', 'RecentForm', 'Team_Encoded', 'Compound_Label', 'Driver_Elo', 'is_wet']
    target = 'FinalPosition'
    X_train_rf = df.loc[train_mask, features_rf]
    X_test_rf = df.loc[test_mask, features_rf]
    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    print(f"Training RandomForest")
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
    rf.fit(X_train_rf, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, 'f1_rf_best_model.joblib'))
    print(f"Random Forest model saved")

    # --- MLP SECTION (One-Hot & Scaling) ---
    print("Preparing MLP")
    df_dummies = pd.get_dummies(df['startCompound'], prefix='tyre')
    df_mlp = pd.concat([df, df_dummies], axis=1)

    # Save dummy columns for consistency during inference (one hot encoding)
    joblib.dump(list(df_dummies.columns), os.path.join(MODEL_DIR, 'mlp_feature_cols.joblib'))

    features_mlp = ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition', 'Humidity', 'Temperature', 'RecentForm', 'is_wet'] + list(df_dummies.columns)

    X_train_mlp = df_mlp.loc[train_mask, features_mlp]
    X_test_mlp = df_mlp.loc[test_mask, features_mlp]

    # Scaling Input
    scaler_x = RobustScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_mlp)
    X_test_mlp_scaled = scaler_x.transform(X_test_mlp)
    joblib.dump(scaler_x, os.path.join(MODEL_DIR, 'scaler_mlp.joblib'))

    # Target Scaling (MinMaxScaler to bring 1-21 into range 0-1)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y_mlp.joblib'))

    print(f"Training MLP")
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128, 64),
        activation='tanh',
        solver='adam',
        alpha=0.15,
        learning_rate='adaptive',
        learning_rate_init=0.0005,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        batch_size=32
    )
    mlp.fit(X_train_scaled, y_train_scaled)
    joblib.dump(mlp, os.path.join(MODEL_DIR, 'f1_mlp_best_model.joblib'))
    print(f"MLP model saved")

    return rf, mlp, X_test_rf, X_test_mlp_scaled, y_test, scaler_y, features_rf

def evaluate_model(rf, mlp, X_test_rf, X_test_mlp, y_test, scaler_y, feature_names):
    """Performance evaluation and Permutation Importance."""
    print("\n--- [3. EVALUATION] ---")

    # RF Evaluation
    rf_preds = np.round(np.clip(rf.predict(X_test_rf), 1, 21))
    mae_rf = mean_absolute_error(y_test, rf_preds)
    r2_rf = r2_score(y_test, rf_preds)

    # MLP Evaluation
    mlp_pred_scaled = mlp.predict(X_test_mlp)
    # Bring values back from 0-1 to 1-21
    mlp_preds = scaler_y.inverse_transform(mlp_pred_scaled.reshape(-1, 1)).ravel()
    mae_mlp = mean_absolute_error(y_test, mlp_preds)
    r2_mlp = r2_score(y_test, mlp_preds)

    print(f"{'MODEL':<20} | {'R2 Score':<15} | {'MAE':<15}")
    print("-" * 55)
    print(f"{'Random Forest':<20} | {r2_rf:<15.4f} | {mae_rf:<15.2f}")
    print(f"{'MLP (Neural)':<20} | {r2_mlp:<15.4f} | {mae_mlp:<15.2f}")

    print("\nCalculating Permutation Importance")
    perm = permutation_importance(rf, X_test_rf, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    sorted_idx = perm.importances_mean.argsort()[::-1]
    for i in sorted_idx:
        print(f"{feature_names[i]:<20} | {perm.importances_mean[i]:.4f} +/- {perm.importances_std[i]:.4f}")

def main():
    # --- DATASET MANAGEMENT ---
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset: {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)
    else:
        print("Dataset not found. Starting full processing")
        # Ensure load_raw_data, process_race_results, and add_features are imported/defined
        df = load_raw_data()
        df = process_race_results(df)
        df = add_features(df)
        df.to_parquet(DATA_PATH, index=False)
        print(f"Dataset saved successfully.")

    # --- TRAINING PIPELINE ---
    # Combines data preparation, encoding, and training
    rf, mlp, X_test_rf, X_test_mlp, y_test, scaler_y, features_rf = train_model_pipeline(df)

    # --- EVALUATION ---
    evaluate_model(rf, mlp, X_test_rf, X_test_mlp, y_test, scaler_y, features_rf)

if __name__ == "__main__":
    main()