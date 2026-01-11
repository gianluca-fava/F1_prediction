import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurazione Plot
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_artifacts():
    print("📦 Caricamento modelli e artefatti...")
    artifacts = {}
    try:
        artifacts['rf'] = joblib.load('f1_rf_best_model.joblib')
        artifacts['mlp'] = joblib.load('f1_mlp_model.joblib')
        artifacts['le_driver'] = joblib.load('le_driver.joblib')
        artifacts['le_team'] = joblib.load('le_team.joblib')
        artifacts['le_circuit'] = joblib.load('le_circuit.joblib')
        artifacts['le_compound'] = joblib.load('le_compound.joblib')
        artifacts['scaler'] = joblib.load('scaler_mlp.joblib')
        artifacts['scaler_y'] = joblib.load('scaler_y_mlp.joblib')
        artifacts['mlp_cols'] = joblib.load('mlp_feature_cols.joblib')
        print("✅ Tutto caricato correttamente.")
    except FileNotFoundError as e:
        print(f"❌ Errore: File mancante -> {e}")
        exit(1)
    return artifacts

def prepare_data(df, artifacts):
    print("⚙️ Preparazione dati di test (Stagione 2025)...")
    
    # Filtra solo 2025
    df = df[df['Year'] == 2025].copy()
    
    if df.empty:
        print("❌ Nessun dato trovato per l'anno 2025.")
        exit(1)

    # 1. Label Encoding (gestione etichette non viste con fallback)
    def safe_transform(encoder, series):
        # Se ci sono label non viste, le mappiamo a -1 o al primo valore (meglio gestire eccezioni)
        # Qui assumiamo che il training abbia visto tutto o usiamo un try/except riga per riga se necessario
        # Dato che encoders sono fittati su tutto il dataset nel training script, dovrebbe andare bene.
        return encoder.transform(series)

    df['Driver_Encoded'] = safe_transform(artifacts['le_driver'], df['Driver'])
    df['Team_Encoded'] = safe_transform(artifacts['le_team'], df['Team'])
    df['Circuit_Encoded'] = safe_transform(artifacts['le_circuit'], df['Circuit'])
    df['Compound_Label'] = safe_transform(artifacts['le_compound'], df['startCompound'])

    # 2. Features per RF
    features_rf = ['Driver_Encoded', 'Team_Encoded', 'Circuit_Encoded', 'Compound_Label',
                   'GridPosition', 'Humidity', 'Temperature', 'RecentForm', 'is_wet']
    
    X_rf = df[features_rf]

    # 3. Features per MLP (One-Hot + Scaling)
    # Ricrea dummies
    df_dummies = pd.get_dummies(df['startCompound'], prefix='tyre')
    
    # Assicura che ci siano tutte le colonne usate in training
    for col in artifacts['mlp_cols']:
        if col not in df_dummies.columns:
            df_dummies[col] = 0
    
    # Seleziona solo quelle giuste nell'ordine giusto
    df_dummies = df_dummies[artifacts['mlp_cols']]
    
    base_mlp_feats = ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition',
                      'Humidity', 'Temperature', 'RecentForm', 'is_wet']
    
    X_mlp = pd.concat([df[base_mlp_feats], df_dummies], axis=1)
    
    # Scaling
    X_mlp_scaled = artifacts['scaler'].transform(X_mlp)

    return df, X_rf, X_mlp_scaled

def plot_scatter(y_true, y_pred, model_name, ax):
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None, ax=ax)
    
    # Linea ideale
    ax.plot([1, 21], [1, 21], 'r--', lw=2, label='Perfetta Previsione')
    
    # Configurazione Assi per gestire DNF e Scala
    ax.set_title(f'{model_name}: Reale vs Predetto')
    ax.set_xlabel('Posizione Reale')
    ax.set_ylabel('Posizione Predetta')
    
    # Forza i limiti per evitare scale enormi (es. 70)
    ax.set_xlim(0.5, 22)
    ax.set_ylim(0.5, 22)
    
    # Etichette personalizzate per DNF
    ticks = [1, 5, 10, 15, 20, 21]
    labels = ['1', '5', '10', '15', '20', 'DNF']
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    
    ax.legend(loc='upper left')

def plot_residuals(y_true, y_pred, model_name, ax):
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, bins=20, ax=ax, color='orange')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f'{model_name}: Distribuzione Errori (Residui)')
    ax.set_xlabel('Errore (Posizioni)')

def plot_race_comparison(df, model_name, pred_col, filename, selected_races=None):
    """Visualizza 3 gare specifiche o Monaco + casuali"""
    races_in_df = df['Circuit'].unique()
    
    if selected_races is None:
        selected_races = []
        # 1. Cerca Monaco
        monaco_names = ['Monaco Grand Prix', 'Monaco']
        for m in monaco_names:
            if m in races_in_df:
                selected_races.append(m)
                break
                
        # 2. Riempi con altri random
        remaining_races = [r for r in races_in_df if r not in selected_races]
        n_needed = 3 - len(selected_races)
        
        if len(remaining_races) >= n_needed:
            random_picks = np.random.choice(remaining_races, n_needed, replace=False)
            selected_races.extend(random_picks)
        else:
            selected_races.extend(remaining_races)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Dettaglio Gara: {model_name} (Classifica Piloti)', fontsize=16)

    for i, race in enumerate(selected_races):
        if race not in races_in_df:
            print(f"⚠️ Avviso: La gara '{race}' non è presente nei dati 2025.")
            continue
            
        # Ordina per posizione REALE
        race_data = df[df['Circuit'] == race].sort_values('FinalPosition')
        
        ax = axes[i]
        
        x_indices = range(len(race_data))
        
        # Clip delle predizioni per il plot (così non vanno fuori grafico)
        y_pred_clipped = np.clip(race_data[pred_col], 1, 21)
        
        ax.plot(x_indices, race_data['FinalPosition'], 'o-', label='Reale', color='black', markersize=8)
        ax.plot(x_indices, y_pred_clipped, 'x--', label='Predetto', color='red', markersize=8)
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(race_data['Driver'], rotation=90, fontsize=9)
        ax.set_title(f"GP: {race}")
        
        # Gestione asse Y con DNF
        ax.set_ylim(0.5, 21.5)
        ticks = [1, 5, 10, 15, 20, 21]
        labels = ['1', '5', '10', '15', '20', 'DNF']
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.invert_yaxis() # 1st is top
        
        if i == 0:
            ax.set_ylabel("Posizione Finale")
            ax.legend()
        
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 Grafico salvato: {filename}")
    # plt.show()

def plot_best_worst_races(df, model_name, pred_col, filename):
    """Identifica e plotta la gara con l'errore minore (Best) e maggiore (Worst)"""
    races = df['Circuit'].unique()
    race_metrics = []

    for race in races:
        race_data = df[df['Circuit'] == race]
        # Calcolo MAE (Mean Absolute Error) per questa gara
        mae = np.mean(np.abs(race_data['FinalPosition'] - race_data[pred_col]))
        race_metrics.append((race, mae))

    # Ordina per errore: [0] è Best, [-1] è Worst
    race_metrics.sort(key=lambda x: x[1])
    
    best_race_name, best_mae = race_metrics[0]
    worst_race_name, worst_mae = race_metrics[-1]
    
    selected_races = [best_race_name, worst_race_name]
    titles = [f"BEST: {best_race_name} (Err: {best_mae:.2f})", f"WORST: {worst_race_name} (Err: {worst_mae:.2f})"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f'{model_name}: Best vs Worst Prediction Performance', fontsize=16)

    for i, race in enumerate(selected_races):
        # Ordina per posizione REALE
        race_data = df[df['Circuit'] == race].sort_values('FinalPosition')
        ax = axes[i]
        x_indices = range(len(race_data))
        
        # Clip predizioni
        y_pred_clipped = np.clip(race_data[pred_col], 1, 21)
        
        ax.plot(x_indices, race_data['FinalPosition'], 'o-', label='Reale', color='green', markersize=8)
        ax.plot(x_indices, y_pred_clipped, 'x--', label='Predetto', color='red', markersize=8)
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(race_data['Driver'], rotation=90, fontsize=9)
        ax.set_title(titles[i], fontweight='bold')
        
        # Gestione Assi
        ax.set_ylim(0.5, 21.5)
        ticks = [1, 5, 10, 15, 20, 21]
        labels = ['1', '5', '10', '15', '20', 'DNF']
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        
        if i == 0:
            ax.set_ylabel("Posizione Finale")
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 Grafico salvato: {filename}")

def main():
    # 1. Carica Dataset
    if not os.path.exists('f1_dataset_processed.parquet'):
        print("Dataset non trovato. Esegui prima train_classic.py per generarlo.")
        return
        
    df = pd.read_parquet('f1_dataset_processed.parquet')
    
    # 2. Carica Modelli
    artifacts = load_artifacts()
    
    # 3. Prepara Dati
    df_test, X_rf, X_mlp = prepare_data(df, artifacts)
    y_true = df_test['FinalPosition']
    
    # 4. Predizioni
    print("🔮 Generazione predizioni...")
    pred_rf = artifacts['rf'].predict(X_rf)
    
    # MLP Pred (Inverse Transform)
    pred_mlp_scaled = artifacts['mlp'].predict(X_mlp)
    pred_mlp = artifacts['scaler_y'].inverse_transform(pred_mlp_scaled.reshape(-1, 1)).ravel()
    
    df_test['Pred_RF'] = pred_rf
    df_test['Pred_MLP'] = pred_mlp

    # 5. Plotting Generale
    print("📊 Generazione grafici...")
    
    # Figura 1: Scatter e Residui
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_scatter(y_true, pred_rf, "Random Forest", axs[0, 0])
    plot_residuals(y_true, pred_rf, "Random Forest", axs[0, 1])
    
    plot_scatter(y_true, pred_mlp, "MLP (Neural Net)", axs[1, 0])
    plot_residuals(y_true, pred_mlp, "MLP (Neural Net)", axs[1, 1])
    
    plt.tight_layout()
    plt.savefig('plots/results_scatter_residuals.png')
    print("💾 Grafico salvato: plots/results_scatter_residuals.png")
    # plt.show()
    
    # Figura 2: Dettaglio Gare RF
    rf_races = ['Australian Grand Prix', 'Qatar Grand Prix', 'British Grand Prix']
    plot_race_comparison(df_test, "Random Forest", 'Pred_RF', 'plots/results_rf_races.png', selected_races=rf_races)
    
    # Figura 3: Dettaglio Gare MLP
    plot_race_comparison(df_test, "MLP", 'Pred_MLP', 'plots/results_mlp_races.png')

    # Figura 4: Best vs Worst (RF)
    plot_best_worst_races(df_test, "Random Forest", 'Pred_RF', 'plots/results_rf_best_worst.png')

    # Figura 5: Best vs Worst (MLP)
    plot_best_worst_races(df_test, "MLP", 'Pred_MLP', 'plots/results_mlp_best_worst.png')

if __name__ == "__main__":
    main()
