# model_development.py
# Phase 2: ML Model Development & Optimization for Predictive Vehicle Maintenance

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# 1. Load processed data (assuming data_preprocessing.py has created *_processed.csv files)
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
processed_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
dfs = {os.path.splitext(f)[0]: pd.read_csv(os.path.join(data_dir, f)) for f in processed_files}

# 2. Example: Use vehicle_sensor_data_processed for classification (adjust as needed)
df = dfs.get('vehicle_sensor_data_processed')
if df is None:
    raise FileNotFoundError('vehicle_sensor_data_processed.csv not found in data directory.')

# Assume 'failure_type' is the target for classification (update as per your data)
if 'failure_type' not in df.columns:
    raise ValueError('Expected target column "failure_type" not found.')

X = df.drop(columns=['failure_type'])
y = df['failure_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Classification Models: XGBoost & LightGBM
mlflow.set_experiment('vehicle_maintenance_classification')

with mlflow.start_run(run_name="XGBoost_Classification"):
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    grid = RandomizedSearchCV(xgb_model, param_grid, n_iter=5, scoring='f1_weighted', cv=3, random_state=42)
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    y_pred = best_xgb.predict(X_test)
    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })
    mlflow.sklearn.log_model(best_xgb, "xgboost_model")

with mlflow.start_run(run_name="LightGBM_Classification"):
    lgb_model = lgb.LGBMClassifier()
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    grid = RandomizedSearchCV(lgb_model, param_grid, n_iter=5, scoring='f1_weighted', cv=3, random_state=42)
    grid.fit(X_train, y_train)
    best_lgb = grid.best_estimator_
    y_pred = best_lgb.predict(X_test)
    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })
    mlflow.sklearn.log_model(best_lgb, "lightgbm_model")

# 4. Anomaly Detection: Isolation Forest (example)
mlflow.set_experiment('vehicle_maintenance_anomaly_detection')
with mlflow.start_run(run_name="IsolationForest_Anomaly"):
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_train)
    y_pred = iso.predict(X_test)
    # -1 for anomaly, 1 for normal
    anomaly_score = (y_pred == -1).sum() / len(y_pred)
    mlflow.log_metric('anomaly_rate', anomaly_score)
    mlflow.sklearn.log_model(iso, "isolation_forest_model")

# 5. Time-Series Forecasting: LSTM (Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

mlflow.set_experiment('vehicle_maintenance_lstm_forecasting')
with mlflow.start_run(run_name="LSTM_Forecasting"):
    # Example: assume 'failure_count' is the target time series (adjust as needed)
    # This is a placeholder; adjust for your real time-series data
    ts_df = dfs.get('engine_data_processed')
    if ts_df is not None and 'failure_count' in ts_df.columns:
        series = ts_df['failure_count'].values.reshape(-1, 1)
        # Normalize
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series)
        # Prepare sequences
        def create_sequences(data, seq_length=10):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        X_seq, y_seq = create_sequences(series_scaled, seq_length=10)
        X_train, X_test = X_seq[:-20], X_seq[-20:]
        y_train, y_test = y_seq[:-20], y_seq[-20:]
        # LSTM model
        model = Sequential([
            LSTM(32, input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=3)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[es], verbose=0)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mape', mape)
        model.save('lstm_model.h5')
        mlflow.log_artifact('lstm_model.h5')
    else:
        print('Time-series data for LSTM not found or target column missing.')
