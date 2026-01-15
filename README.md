# F1 Race Prediction System

This framework predict Formula 1 race results. It utilizes **Random Forest** and **Multi-Layer Perceptron (MLP)** models, leveraging historical race data, weather conditions, and derived features like Elo ratings and driver form.

## Project Structure

- **`download_data.py`**: Script to download raw F1 race data using the `fastf1` library.
- **`train.py`**: Main pipeline script. It processes raw data, computes features (Elo, RecentForm), trains the RF and MLP models, and evaluates them.
- **`predict_race.py`**: Interactive CLI tool to predict the finish position of a specific driver given race conditions.
- **`visualize_predictions.py`**: Generates performance plots (Scatter, Residuals, Best/Worst races) based on the test set (2025 season).
- **`data/`**: Directory storing raw and processed parquet data files.
- **`joblib/`**: Directory storing trained models (`.joblib`) and scalers/encoders.
- **`plots/`**: Directory where visualization results are saved.

## Prerequisites

Ensure you have Python installed. You can install the required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

## Usage Workflow

### 1. Data Ingestion
Download the historical race data. The script is configured to download data from 2022 to 2023 by default (configurable in the script).

```bash
python download_data.py
```
*Note: This creates raw parquet files in the `data/` directory.*

### 2. Model Training
Process the data, generate features, and train the models. This script also evaluates the models on the test set (default: 2025 season).

```bash
python train.py
```
*   **Features Computed**: `Driver_Elo`, `RecentForm`, `is_wet`, `Team_Encoded`, etc.
*   **Outputs**:
    *   Processed dataset: `f1_dataset_processed.parquet`
    *   Trained Models & Artifacts: Saved in `joblib/`

### 3. Visualization & Evaluation
Generate graphs to visualize model performance, including Real vs. Predicted scatter plots and detailed race comparisons.

```bash
python visualize_predictions.py
```
*Check the `plots/` directory for the generated images.*

### 4. Interactive Prediction
Make predictions for a specific scenario by providing inputs interactively (Driver, Circuit, Grid Position, Humidity).

```bash
python predict_race.py
```

## Model Details

The system uses two regression models:

1.  **Random Forest Regressor**:
    *   Robust to outliers and non-linear data.
    *   Key Features: `GridPosition`, `RecentForm`, `Team`, `Driver_Elo`.

2.  **MLP Regressor (Neural Network)**:
    *   Deep learning approach for capturing complex patterns.
    *   Architecture: 4 Hidden Layers (256, 256, 128, 64), Tanh activation.
    *   Inputs are scaled using `RobustScaler` (features) and `MinMaxScaler` (target).

## Features

*   **Elo Rating**: Dynamic rating updated race-by-race based on head-to-head performance.
*   **Recent Form**: Rolling average of final positions over the last 3 races.
*   **Weather**: Average track/air temperature and humidity.
*   **Context**: Grid position, Tyre compound, Circuit, Team.
