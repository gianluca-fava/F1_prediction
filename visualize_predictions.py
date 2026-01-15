import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Plot Configuration
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_artifacts():
    print("📦 Loading models and artifacts...")
    artifacts = {}
    try:
        artifacts['rf'] = joblib.load('joblib/f1_rf_best_model.joblib')
        artifacts['mlp'] = joblib.load('joblib/f1_mlp_model.joblib')
        artifacts['le_driver'] = joblib.load('joblib/le_driver.joblib')
        artifacts['le_team'] = joblib.load('joblib/le_team.joblib')
        artifacts['le_circuit'] = joblib.load('joblib/le_circuit.joblib')
        artifacts['le_compound'] = joblib.load('joblib/le_compound.joblib')
        artifacts['scaler'] = joblib.load('joblib/scaler_mlp.joblib')
        artifacts['scaler_y'] = joblib.load('joblib/scaler_y_mlp.joblib')
        artifacts['mlp_cols'] = joblib.load('joblib/mlp_feature_cols.joblib')
        print("✅ Models loaded")
    except FileNotFoundError as e:
        print(f"❌ Error: Missing file: {e}")
        exit(1)
    return artifacts

def prepare_data(df, artifacts):
    print("⚙️ Preparing test data (2025 Season)...")
    
    # Filter only 2025
    df = df[df['Year'] == 2025].copy()
    
    if df.empty:
        print("❌ No data found for the year 2025.")
        exit(1)

    # 1. Label Encoding (handling unseen labels with fallback)
    def safe_transform(encoder, series):
        # If there are unseen labels, we map them to -1 or the first value (better to handle exceptions)
        # Here we assume that training saw everything or we use a row-by-row try/except if necessary
        # Since encoders are fitted on the entire dataset in the training script, it should be fine.
        return encoder.transform(series)

    df['Driver_Encoded'] = safe_transform(artifacts['le_driver'], df['Driver'])
    df['Team_Encoded'] = safe_transform(artifacts['le_team'], df['Team'])
    df['Circuit_Encoded'] = safe_transform(artifacts['le_circuit'], df['Circuit'])
    df['Compound_Label'] = safe_transform(artifacts['le_compound'], df['startCompound'])

    # 2. Features for RF
    features_rf = ['GridPosition', 'RecentForm', 'Team_Encoded', 'Compound_Label', 'Driver_Elo', 'is_wet']
    
    X_rf = df[features_rf]

    # 3. Features for MLP (One-Hot + Scaling)
    # Recreate dummies
    df_dummies = pd.get_dummies(df['startCompound'], prefix='tyre')
    
    # Ensure all columns used in training are present
    for col in artifacts['mlp_cols']:
        if col not in df_dummies.columns:
            df_dummies[col] = 0
    
    # Select only the correct ones in the correct order
    df_dummies = df_dummies[artifacts['mlp_cols']]
    
    base_mlp_feats = ['Driver_Encoded', 'Circuit_Encoded', 'GridPosition',
                      'Humidity', 'Temperature', 'RecentForm', 'is_wet']
    
    X_mlp = pd.concat([df[base_mlp_feats], df_dummies], axis=1)
    
    # Scaling
    X_mlp_scaled = artifacts['scaler'].transform(X_mlp)

    return df, X_rf, X_mlp_scaled

def plot_scatter(y_true, y_pred, model_name, ax):
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None, ax=ax)
    
    # Ideal line
    ax.plot([1, 21], [1, 21], 'r--', lw=2, label='Perfect Prediction')
    
    # Axis configuration to handle DNF and Scale
    ax.set_title(f'{model_name}: Real vs Predicted')
    ax.set_xlabel('Real Position')
    ax.set_ylabel('Predicted Position')
    
    # Force limits to avoid huge scales (e.g. 70)
    ax.set_xlim(0.5, 22)
    ax.set_ylim(0.5, 22)
    
    # Custom labels for DNF
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
    ax.set_title(f'{model_name}: Error Distribution (Residuals)')
    ax.set_xlabel('Error (Positions)')

def plot_race_comparison(df, model_name, pred_col, filename, selected_races=None):
    """Visualizes 3 specific races or Monaco + random ones"""
    races_in_df = df['Circuit'].unique()
    
    if selected_races is None:
        selected_races = []
        # 1. Search for Monaco
        monaco_names = ['Monaco Grand Prix', 'Monaco']
        for m in monaco_names:
            if m in races_in_df:
                selected_races.append(m)
                break
                
        # 2. Fill with other random ones
        remaining_races = [r for r in races_in_df if r not in selected_races]
        n_needed = 3 - len(selected_races)
        
        if len(remaining_races) >= n_needed:
            random_picks = np.random.choice(remaining_races, n_needed, replace=False)
            selected_races.extend(random_picks)
        else:
            selected_races.extend(remaining_races)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Race Detail: {model_name} (Driver Standings)', fontsize=16)

    for i, race in enumerate(selected_races):
        if race not in races_in_df:
            print(f"⚠️ Warning: The race '{race}' is not present in 2025 data.")
            continue
            
        # Sort by REAL position
        race_data = df[df['Circuit'] == race].sort_values('FinalPosition')
        
        ax = axes[i]
        
        x_indices = range(len(race_data))
        
        # Clip predictions for the plot (so they don't go out of the graph)
        y_pred_clipped = np.clip(race_data[pred_col], 1, 21)
        
        ax.plot(x_indices, race_data['FinalPosition'], 'o-', label='Real', color='black', markersize=8)
        ax.plot(x_indices, y_pred_clipped, 'x--', label='Predicted', color='red', markersize=8)
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(race_data['Driver'], rotation=90, fontsize=9)
        ax.set_title(f"GP: {race}")
        
        # Y axis management with DNF
        ax.set_ylim(0.5, 21.5)
        ticks = [1, 5, 10, 15, 20, 21]
        labels = ['1', '5', '10', '15', '20', 'DNF']
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.invert_yaxis() # 1st is top
        
        if i == 0:
            ax.set_ylabel("Final Position")
            ax.legend()
        
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 Plot saved: {filename}")
    # plt.show()

def plot_best_worst_races(df, model_name, pred_col, filename):
    """Identifies and plots the race with the smallest error (Best) and largest error (Worst)"""
    races = df['Circuit'].unique()
    race_metrics = []

    for race in races:
        race_data = df[df['Circuit'] == race]
        # MAE (Mean Absolute Error) calculation for this race
        mae = np.mean(np.abs(race_data['FinalPosition'] - race_data[pred_col]))
        race_metrics.append((race, mae))

    # Sort by error: [0] is Best, [-1] is Worst
    race_metrics.sort(key=lambda x: x[1])
    
    best_race_name, best_mae = race_metrics[0]
    worst_race_name, worst_mae = race_metrics[-1]
    
    selected_races = [best_race_name, worst_race_name]
    titles = [f"BEST: {best_race_name} (Err: {best_mae:.2f})", f"WORST: {worst_race_name} (Err: {worst_mae:.2f})"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f'{model_name}: Best vs Worst Prediction Performance', fontsize=16)

    for i, race in enumerate(selected_races):
        # Sort by REAL position
        race_data = df[df['Circuit'] == race].sort_values('FinalPosition')
        ax = axes[i]
        x_indices = range(len(race_data))
        
        # Clip predictions
        y_pred_clipped = np.clip(race_data[pred_col], 1, 21)
        
        ax.plot(x_indices, race_data['FinalPosition'], 'o-', label='Real', color='green', markersize=8)
        ax.plot(x_indices, y_pred_clipped, 'x--', label='Predicted', color='red', markersize=8)
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(race_data['Driver'], rotation=90, fontsize=9)
        ax.set_title(titles[i], fontweight='bold')
        
        # Axis Management
        ax.set_ylim(0.5, 21.5)
        ticks = [1, 5, 10, 15, 20, 21]
        labels = ['1', '5', '10', '15', '20', 'DNF']
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        
        if i == 0:
            ax.set_ylabel("Final Position")
            ax.legend()
            
        # Prepare and print the comparison table for the best/worst race
        comparison_df = race_data[['Driver', 'FinalPosition', pred_col]].copy()
        comparison_df.rename(columns={'FinalPosition': 'Real Result', pred_col: 'Predicted Result'}, inplace=True)
        
        print(f"\n--- {model_name} - Comparison Table for {titles[i]} ---")
        print(comparison_df.to_string(index=False))

        # Save to CSV
        race_type = "best" if i == 0 else "worst"
        model_clean = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        csv_filename = f"plots/{model_clean}_{race_type}_race.csv"
        comparison_df.to_csv(csv_filename, index=False)
        print(f"💾 Table saved: {csv_filename}")
            
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 Plot saved: {filename}")

def main():
    # 1. Load Dataset
    if not os.path.exists('f1_dataset_processed.parquet'):
        print("Dataset not found. Run train_classic.py first to generate it.")
        return
        
    df = pd.read_parquet('f1_dataset_processed.parquet')
    
    # 2. Load Models
    artifacts = load_artifacts()
    
    # 3. Prepare Data
    df_test, X_rf, X_mlp = prepare_data(df, artifacts)
    y_true = df_test['FinalPosition']
    
    # 4. Predictions
    print("🔮 Generating predictions...")
    pred_rf = artifacts['rf'].predict(X_rf)
    
    # MLP Pred (Inverse Transform)
    pred_mlp_scaled = artifacts['mlp'].predict(X_mlp)
    pred_mlp = artifacts['scaler_y'].inverse_transform(pred_mlp_scaled.reshape(-1, 1)).ravel()
    
    df_test['Pred_RF'] = pred_rf
    df_test['Pred_MLP'] = pred_mlp

    # 5. General Plotting
    print("📊 Generating plots...")
    
    # Figure 1: Scatter and Residuals
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    plot_scatter(y_true, pred_rf, "Random Forest", axs[0, 0])
    plot_residuals(y_true, pred_rf, "Random Forest", axs[0, 1])
    
    plot_scatter(y_true, pred_mlp, "MLP (Neural Net)", axs[1, 0])
    plot_residuals(y_true, pred_mlp, "MLP (Neural Net)", axs[1, 1])
    
    plt.tight_layout()
    plt.savefig('plots/results_scatter_residuals.png')
    print("💾 Plot saved: plots/results_scatter_residuals.png")
    # plt.show()
    
    # Figure 2: RF Race Detail
    rf_races = ['Australian Grand Prix', 'Qatar Grand Prix', 'British Grand Prix']
    plot_race_comparison(df_test, "Random Forest", 'Pred_RF', 'plots/results_rf_races.png', selected_races=rf_races)
    
    # Figure 3: MLP Race Detail
    plot_race_comparison(df_test, "MLP", 'Pred_MLP', 'plots/results_mlp_races.png')

    # Figure 4: Best vs Worst (RF)
    plot_best_worst_races(df_test, "Random Forest", 'Pred_RF', 'plots/results_rf_best_worst.png')

    # Figure 5: Best vs Worst (MLP)
    plot_best_worst_races(df_test, "MLP", 'Pred_MLP', 'plots/results_mlp_best_worst.png')

if __name__ == "__main__":
    main()
