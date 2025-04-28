# model_development.py
# Phase 2: ML Model Development & Optimization for Predictive Vehicle Maintenance

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# 1. Load processed data
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
processed_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
dfs = {os.path.splitext(f)[0]: pd.read_csv(os.path.join(data_dir, f)) for f in processed_files}

# 2. Clean models directory before training
import glob
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)
for f in glob.glob(os.path.join(models_dir, '*.pkl')):
    try:
        os.remove(f)
        print(f"Removed old model file: {f}")
    except Exception as e:
        print(f"Could not remove {f}: {e}")

# 3. Use vehicle_sensor_data_synthetic.csv for classification
df = pd.read_csv(os.path.join(data_dir, 'vehicle_sensor_data_synthetic.csv'))
# Normalize column names: replace spaces with underscores for consistency
df.columns = [col.replace(' ', '_') for col in df.columns]
if 'failure_type' not in df.columns:
    raise ValueError('Expected target column "failure_type" not found in synthetic data.')

# 4. Prepare features and target
non_feature_cols = ['failure_type', 'Maintenance_Type']
X = df.drop(columns=[col for col in non_feature_cols if col in df.columns])
y = df['failure_type']

# 5. Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
print('Class distribution after SMOTE:', Counter(y_resampled))
print('Label mapping:', dict(zip(le.classes_, le.transform(le.classes_))))

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print('Class distribution in y_train:', Counter(y_train))
print('Class distribution in y_test:', Counter(y_test))

# 8. LightGBM Model Training, Saving, and Verification
mlflow.set_experiment('vehicle_maintenance_classification')
with mlflow.start_run(run_name="LightGBM_Classification"):
    lgb_model = lgb.LGBMClassifier(class_weight='balanced')
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        grid = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=5, scoring='f1_weighted',
            cv=skf, random_state=42
        )
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
        # Save and verify
        import joblib
        lgb_path = os.path.join(models_dir, 'lightgbm_model.pkl')
        joblib.dump(best_lgb, lgb_path)
        print(f"Saved LightGBM model to {lgb_path}")
        # Verify by loading and predicting
        loaded_lgb = joblib.load(lgb_path)
        dummy = np.array([X_test.iloc[0].values]).reshape(1, -1)
        try:
            _ = loaded_lgb.predict(dummy)
            print("LightGBM model is fitted and can predict.")
        except Exception as v_e:
            print(f"Verification failed: {v_e}")
    except Exception as e:
        print(f"LightGBM training failed: {e}")
        print(f"Unique classes in y_train: {set(y_train)}")
        print(f"Class counts: {Counter(y_train)})")
        pass
# --- End LightGBM Only ---

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

# CLEAN models directory before training
import glob
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)
for f in glob.glob(os.path.join(models_dir, '*.pkl')):
    try:
        os.remove(f)
        print(f"Removed old model file: {f}")
    except Exception as e:
        print(f"Could not remove {f}: {e}")

# 2. Use vehicle_sensor_data_synthetic.csv for classification
df = pd.read_csv(os.path.join(data_dir, 'vehicle_sensor_data_synthetic.csv'))
# Normalize column names: replace spaces with underscores for consistency
df.columns = [col.replace(' ', '_') for col in df.columns]

# Use the synthetic 'failure_type' column as target
if 'failure_type' not in df.columns:
    raise ValueError('Expected target column "failure_type" not found in synthetic data.')

# Drop non-numeric columns for ML models
non_feature_cols = ['failure_type', 'Maintenance_Type']
X = df.drop(columns=[col for col in non_feature_cols if col in df.columns])
y = df['failure_type']

from collections import Counter
from imblearn.over_sampling import SMOTE

# Encode y labels as integers for ML models
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Use SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
print('Class distribution after SMOTE:', Counter(y_resampled))

# Print label mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Label mapping:', label_mapping)

# Hybrid: You could add undersampling or SMOTE here for more advanced balancing

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
print('Class distribution in y_train:', Counter(y_train))
print('Class distribution in y_test:', Counter(y_test))

# 3. Classification Models: XGBoost & LightGBM
mlflow.set_experiment('vehicle_maintenance_classification')

from sklearn.model_selection import StratifiedKFold
# Only LightGBM section remains
with mlflow.start_run(run_name="LightGBM_Classification"):
    lgb_model = lgb.LGBMClassifier(class_weight='balanced')
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        grid = RandomizedSearchCV(lgb_model, param_grid, n_iter=5, scoring='f1_weighted', cv=skf, random_state=42)
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
        # Save and verify
        import joblib
        lgb_path = os.path.join(models_dir, 'lightgbm_model.pkl')
        joblib.dump(best_lgb, lgb_path)
        print(f"Saved LightGBM model to {lgb_path}")
        # Verify by loading and predicting
        loaded_lgb = joblib.load(lgb_path)
        import numpy as np
        dummy = np.array([X_test.iloc[0].values]).reshape(1, -1)
        try:
            _ = loaded_lgb.predict(dummy)
            print("LightGBM model is fitted and can predict.")
        except Exception as v_e:
            print(f"Verification failed: {v_e}")
    except Exception as e:
        print(f"LightGBM training failed: {e}")
        print(f"Unique classes in y_train: {set(y_train)}")
        print(f"Class counts: {Counter(y_train)})")
        pass
# --- End LightGBM Only ---
# --- Time Series Forecasting: LSTM Model Training ---
import torch
import joblib
from lstm_pytorch_utils import train_lstm
import mlflow
import mlflow.pytorch
print("Starting LSTM time-series training...")
mlflow.set_experiment('vehicle_maintenance_timeseries')
with mlflow.start_run(run_name='LSTM_TimeSeries'):
    mlflow.log_param('seq_length', 10)
    mlflow.log_param('epochs', 20)
    mlflow.log_param('lr', 0.001)
    mlflow.log_param('batch_size', 16)
    mlflow.log_param('hidden_size', 32)
    mlflow.log_param('num_layers', 1)
    try:
        # Use Engine_Temperature for forecasting
        series = df["Engine_Temperature_(°C)"].values.astype(float)
        model_ts, scaler_ts, rmse, mape, _, _ = train_lstm(
            series, seq_length=10, epochs=20, lr=0.001,
            batch_size=16, hidden_size=32, num_layers=1
        )
        # Save model state_dict and scaler
        lstm_model_path = os.path.join(models_dir, 'lstm_model.pt')
        torch.save(model_ts.state_dict(), lstm_model_path)
        scaler_path = os.path.join(models_dir, 'lstm_scaler.pkl')
        joblib.dump(scaler_ts, scaler_path)
        print(f"Saved LSTM state_dict to {lstm_model_path}")
        print(f"Saved LSTM scaler to {scaler_path}")
        print(f"LSTM RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
        mlflow.log_metric('lstm_rmse', rmse)
        mlflow.log_metric('lstm_mape', mape)
        mlflow.pytorch.log_model(model_ts, 'lstm_model')
    except Exception as e:
        print(f"Time series LSTM training failed: {e}")
