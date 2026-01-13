import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# --- CONFIGURAZIONI ---
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# --- FUNZIONI DI SUPPORTO ---

def load_artifacts():
    print("📦 Caricamento modelli e artefatti dalla cartella 'joblib'...")
    base_path = "joblib"
    artifacts = {}
    files_to_load = {
        'rf': 'f1_rf_best_model.joblib',
        'mlp': 'f1_mlp_best_model.joblib',
        'le_driver': 'le_driver.joblib',  # Usato da MLP
        'le_team': 'le_team.joblib',  # Usato da entrambi
        'le_circuit': 'le_circuit.joblib',  # Usato da MLP
        'le_compound': 'le_compound.joblib',  # Usato da entrambi
        'scaler': 'scaler_mlp.joblib',
        'mlp_cols': 'mlp_feature_cols.joblib'
    }
    try:
        for key, filename in files_to_load.items():
            full_path = os.path.join(base_path, filename)
            artifacts[key] = joblib.load(full_path)
        print("✅ Tutto caricato correttamente.")
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        raise e
    return artifacts


def prepare_data(df, artifacts):
    print("⚙️ Preparazione dati di test (Stagione 2025)...")
    df_2025 = df[df['Year'] == 2025].copy()
    if df_2025.empty: return None, None, None

    # --- 1. ENCODING PER MLP ---
    # MLP usa ancora Driver e Circuit encoded
    df_2025['Driver_Encoded'] = artifacts['le_driver'].transform(df_2025['Driver'])
    df_2025['Circuit_Encoded'] = artifacts['le_circuit'].transform(df_2025['Circuit'])

    # --- 2. ENCODING COMUNE (RF & MLP) ---
    df_2025['Team_Encoded'] = artifacts['le_team'].transform(df_2025['Team'])
    df_2025['Compound_Label'] = artifacts['le_compound'].transform(df_2025['startCompound'])

    # --- 3. FEATURES RF (Versione Ottimizzata) ---
    # Usiamo esattamente la lista del tuo nuovo training
    features_rf = [
        'GridPosition',
        'RecentForm',
        'Team_Encoded',
        'Compound_Label',
        'Driver_Elo',
        'is_wet'
    ]
    X_rf = df_2025[features_rf]

    # --- 4. FEATURES MLP (One-Hot + Base) ---
    df_dummies = pd.get_dummies(df_2025['startCompound'], prefix='tyre')
    df_mlp_full = pd.concat([df_2025, df_dummies], axis=1)

    base_mlp_feats = ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition',
                      'Humidity', 'Temperature', 'RecentForm', 'is_wet']

    tyre_cols = artifacts['mlp_cols']
    full_features_mlp = base_mlp_feats + tyre_cols

    for col in tyre_cols:
        if col not in df_mlp_full.columns:
            df_mlp_full[col] = 0

    X_mlp_final = df_mlp_full[full_features_mlp]
    X_mlp_scaled = artifacts['scaler'].transform(X_mlp_final)

    return df_2025, X_rf, X_mlp_scaled


# --- FUNZIONI DI PLOTTING (Invariate) ---
def plot_scatter(y_true, y_pred, model_name, ax):
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, ax=ax)
    ax.plot([1, 21], [1, 21], 'r--', lw=2, label='Ideale')
    ax.set_title(f'{model_name}: Reale vs Predetto')
    ax.set_xlim(0.5, 21.5);
    ax.set_ylim(0.5, 21.5)
    ax.invert_yaxis()


def plot_residuals(y_true, y_pred, model_name, ax):
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, bins=20, ax=ax, color='orange')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'{model_name}: Residui')


def plot_race_comparison(df, model_name, pred_col, filename, selected_races=None):
    races_in_df = df['Circuit'].unique()
    if selected_races is None:
        selected_races = np.random.choice(races_in_df, min(3, len(races_in_df)), replace=False)
    fig, axes = plt.subplots(1, len(selected_races), figsize=(18, 6), sharey=True)
    if len(selected_races) == 1: axes = [axes]
    for i, race in enumerate(selected_races):
        race_data = df[df['Circuit'] == race].sort_values('FinalPosition')
        ax = axes[i]
        x = range(len(race_data))
        ax.plot(x, race_data['FinalPosition'], 'o-', label='Reale', color='black')
        ax.plot(x, np.clip(race_data[pred_col], 1, 21), 'x--', label='Predetto', color='red')
        ax.set_xticks(x)
        ax.set_xticklabels(race_data['Driver'], rotation=90)
        ax.set_title(race)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)


def plot_best_worst_races(df, model_name, pred_col, filename):
    races = df['Circuit'].unique()
    metrics = []
    for race in races:
        rd = df[df['Circuit'] == race]
        mae = np.mean(np.abs(rd['FinalPosition'] - rd[pred_col]))
        metrics.append((race, mae))
    metrics.sort(key=lambda x: x[1])
    best, worst = metrics[0][0], metrics[-1][0]
    plot_race_comparison(df, model_name, pred_col, filename, selected_races=[best, worst])


# --- MAIN ---
def main():
    if not os.path.exists('f1_dataset_processed.parquet'):
        print("❌ Dataset mancante!")
        return

    df = pd.read_parquet('f1_dataset_processed.parquet')
    artifacts = load_artifacts()

    # 3. Prepara Dati
    df_test, X_rf, X_mlp_scaled = prepare_data(df, artifacts)
    if 'Driver_Elo' not in df.columns:
        print("⚠️ Colonna 'Driver_Elo' non trovata. La sto calcolando ora...")
        # Se hai la funzione compute_f1_elo importala o incollala qui
        # df['Driver_Elo'] = compute_f1_elo(df, k_factor=2)
        # Se non vuoi ricalcolarla qui, devi rieseguire il training script
        print("❌ Errore: Riesegui il training script per generare il dataset con i ratings Elo.")
        return
    if df_test is None: return

    y_true = df_test['FinalPosition']

    print("🔮 Generazione predizioni...")
    # Predizione RF + Post-processing (arrotondamento e clipping come nel training)
    rf_raw_pred = artifacts['rf'].predict(X_rf)
    df_test['Pred_RF'] = np.round(np.clip(rf_raw_pred, 1, 21))

    # Predizione MLP
    df_test['Pred_MLP'] = artifacts['mlp'].predict(X_mlp_scaled)

    # --- 1. EVALUATION METRICS ---
    mae_rf = mean_absolute_error(y_true, df_test['Pred_RF'])
    r2_rf = r2_score(y_true, df_test['Pred_RF'])
    mae_mlp = mean_absolute_error(y_true, df_test['Pred_MLP'])
    r2_mlp = r2_score(y_true, df_test['Pred_MLP'])

    print("\n" + "=" * 55)
    print(f"📊 PERFORMANCE EVALUATION (TEST 2025)")
    print("-" * 55)
    print(f"{'Modello':<20} | {'MAE':<10} | {'R2':<10}")
    print("-" * 55)
    print(f"{'RF Optimized':<20} | {mae_rf:<10.4f} | {r2_rf:<10.4f}")
    print(f"{'MLP Neural':<20} | {mae_mlp:<10.4f} | {r2_mlp:<10.4f}")
    print("=" * 55)

    # --- 2. PERMUTATION IMPORTANCE (Random Forest) ---
    print("\n📊 Calcolo Permutation Importance (RF - Optimized)...")
    perm_rf = permutation_importance(artifacts['rf'], X_rf, y_true, n_repeats=10, random_state=42, n_jobs=-1)

    features_rf_names = X_rf.columns
    sorted_idx_rf = perm_rf.importances_mean.argsort()[::-1]

    print("\n🏆 Feature Ranking - RANDOM FOREST (Elo Mode):")
    print("-" * 55)
    for i in sorted_idx_rf:
        print(f"{features_rf_names[i]:<20} | {perm_rf.importances_mean[i]:.4f} +/- {perm_rf.importances_std[i]:.4f}")

    # --- 3. PERMUTATION IMPORTANCE (MLP) ---
    print("\n📊 Calcolo Permutation Importance (MLP)...")
    perm_mlp = permutation_importance(artifacts['mlp'], X_mlp_scaled, y_true, n_repeats=10, random_state=42, n_jobs=-1)

    base_mlp_feats = ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition', 'Humidity', 'Temperature', 'RecentForm',
                      'is_wet']
    full_features_mlp = base_mlp_feats + artifacts['mlp_cols']
    sorted_idx_mlp = perm_mlp.importances_mean.argsort()[::-1]

    print("\n🏆 Feature Ranking - MLP (Top 10):")
    print("-" * 55)
    for i in sorted_idx_mlp[:10]:
        print(f"{full_features_mlp[i]:<20} | {perm_mlp.importances_mean[i]:.4f} +/- {perm_mlp.importances_std[i]:.4f}")

    # --- 4. PLOTTING ---
    os.makedirs('plots', exist_ok=True)
    print("\n🎨 Generazione grafici...")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plot_scatter(y_true, df_test['Pred_RF'], "RF Optimized", axs[0, 0])
    plot_residuals(y_true, df_test['Pred_RF'], "RF Optimized", axs[0, 1])
    plot_scatter(y_true, df_test['Pred_MLP'], "MLP Neural", axs[1, 0])
    plot_residuals(y_true, df_test['Pred_MLP'], "MLP Neural", axs[1, 1])
    plt.tight_layout()
    plt.savefig('plots/results_scatter_residuals.png')

    rf_races = ['Australian Grand Prix', 'Qatar Grand Prix', 'British Grand Prix']
    plot_race_comparison(df_test, "Random Forest", 'Pred_RF', 'plots/results_rf_races.png', selected_races=rf_races)
    plot_race_comparison(df_test, "MLP", 'Pred_MLP', 'plots/results_mlp_races.png')
    plot_best_worst_races(df_test, "Random Forest", 'Pred_RF', 'plots/results_rf_best_worst.png')
    plot_best_worst_races(df_test, "MLP", 'Pred_MLP', 'plots/results_mlp_best_worst.png')

    print("✅ Fine. Grafici salvati in /plots")


if __name__ == "__main__":
    main()