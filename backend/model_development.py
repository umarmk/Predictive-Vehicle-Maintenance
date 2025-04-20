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

# 2. Use vehicle_sensor_data_synthetic.csv for classification
df = pd.read_csv(os.path.join(data_dir, 'vehicle_sensor_data_synthetic.csv'))

# Use the synthetic 'failure_type' column as target
if 'failure_type' not in df.columns:
    raise ValueError('Expected target column "failure_type" not found in synthetic data.')

# Drop non-numeric columns for ML models
non_feature_cols = ['failure_type', 'Maintenance Type']
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
with mlflow.start_run(run_name="XGBoost_Classification"):
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, class_weight='balanced')
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        grid = RandomizedSearchCV(xgb_model, param_grid, n_iter=5, scoring='f1_weighted', cv=skf, random_state=42)
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
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        print(f"Unique classes in y_train: {set(y_train)}")
        print(f"Class counts: {Counter(y_train)})")

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
    except Exception as e:
        print(f"LightGBM training failed: {e}")
        print(f"Unique classes in y_train: {set(y_train)}")
        print(f"Class counts: {Counter(y_train)})")

# 4. Ensemble Model: VotingClassifier (XGBoost + LightGBM)
from sklearn.ensemble import VotingClassifier

mlflow.set_experiment('vehicle_maintenance_ensemble')
with mlflow.start_run(run_name="Voting_Ensemble_XGB_LGBM"):
    xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, class_weight='balanced')
    lgb_clf = lgb.LGBMClassifier(class_weight='balanced')
    ensemble = VotingClassifier(estimators=[('xgb', xgb_clf), ('lgb', lgb_clf)], voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })
    mlflow.sklearn.log_model(ensemble, "ensemble_model")
    print('Ensemble VotingClassifier metrics:')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print('Recall:', recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print('F1 Score:', f1_score(y_test, y_pred, average='weighted', zero_division=0))

# 5. Anomaly Detection: Isolation Forest (example)
mlflow.set_experiment('vehicle_maintenance_anomaly_detection')
with mlflow.start_run(run_name="IsolationForest_Anomaly"):
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_train)
    y_pred = iso.predict(X_test)
    # -1 for anomaly, 1 for normal
    anomaly_score = (y_pred == -1).sum() / len(y_pred)
    mlflow.log_metric('anomaly_rate', anomaly_score)
    mlflow.sklearn.log_model(iso, "isolation_forest_model")

# 5. Time-Series Forecasting: LSTM (PyTorch)
from lstm_pytorch_utils import train_lstm

mlflow.set_experiment('vehicle_maintenance_lstm_forecasting')
import itertools
with mlflow.start_run(run_name="LSTM_Forecasting_Tuned"):
    ts_df = dfs.get('engine_data_processed')
    if ts_df is not None and 'Engine Condition' in ts_df.columns:
        series = ts_df['Engine Condition'].values
        # Hyperparameter grid
        param_grid = {
            'seq_length': [5, 10, 20],
            'epochs': [20, 40],
            'lr': [0.001, 0.0005],
            'hidden_size': [16, 32],
            'num_layers': [1, 2]
        }
        best_rmse = float('inf')
        best_params = None
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            try:
                model, scaler, rmse, mape, y_test, y_pred = train_lstm(series, **param_dict)
                param_str = '_'.join([f"{k}{v}" for k, v in param_dict.items()])
                print(f"LSTM run {param_str}: RMSE={rmse}, MAPE={mape}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = param_dict
            except Exception as e:
                print(f"LSTM training failed for {param_dict}: {e}")
        if best_params is not None:
            print(f"Best LSTM RMSE: {best_rmse}")
            print(f"Best LSTM params: {best_params}")
            mlflow.log_metric('best_lstm_rmse', float(best_rmse))
            mlflow.log_params(best_params)
        else:
            print("All LSTM hyperparameter runs failed. No best parameters to log.")
    else:
        print('Time-series data for LSTM not found or target column missing.')
