#!/usr/bin/env python3
"""
Script to create missing ML models for the RideAhead system.
This script generates XGBoost and Random Forest models based on the existing LightGBM model structure.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic vehicle sensor data for training models."""
    np.random.seed(42)
    
    # Generate base features
    engine_temp = np.random.normal(90, 15, n_samples)
    tire_pressure = np.random.normal(32, 3, n_samples)
    brake_pad = np.random.normal(8, 2, n_samples)
    
    # Ensure realistic ranges
    engine_temp = np.clip(engine_temp, 60, 130)
    tire_pressure = np.clip(tire_pressure, 20, 45)
    brake_pad = np.clip(brake_pad, 2, 15)
    
    # Generate anomaly indication based on conditions
    anomaly_indication = np.zeros(n_samples)
    
    # Create failure labels based on realistic conditions
    labels = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Engine failure (overheating)
        if engine_temp[i] > 110:
            labels[i] = 1
            anomaly_indication[i] = 1
        # Brake failure (low pad thickness)
        elif brake_pad[i] < 4:
            labels[i] = 2
            anomaly_indication[i] = 1
        # Tire pressure issues
        elif tire_pressure[i] < 25 or tire_pressure[i] > 40:
            labels[i] = 4
            anomaly_indication[i] = 1
        # General maintenance (random cases)
        elif np.random.random() < 0.1:
            labels[i] = 5
            anomaly_indication[i] = 1
        # Battery issues (random cases)
        elif np.random.random() < 0.05:
            labels[i] = 3
            anomaly_indication[i] = 1
    
    # Create additional binary features
    is_engine_failure = (labels == 1).astype(int)
    is_brake_failure = (labels == 2).astype(int)
    is_battery_failure = (labels == 3).astype(int)
    is_low_tire_pressure = (labels == 4).astype(int)
    is_maintenance_required = (labels == 5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Engine_Temperature_(°C)': engine_temp,
        'Tire_Pressure_(PSI)': tire_pressure,
        'Brake_Pad_Thickness_(mm)': brake_pad,
        'Anomaly_Indication': anomaly_indication,
        'is_engine_failure': is_engine_failure,
        'is_brake_failure': is_brake_failure,
        'is_battery_failure': is_battery_failure,
        'is_low_tire_pressure': is_low_tire_pressure,
        'is_maintenance_required': is_maintenance_required
    })
    
    return data, labels

def create_xgboost_model():
    """Create and train XGBoost model."""
    print("Creating XGBoost model...")
    
    # Generate training data
    X, y = generate_synthetic_data(10000)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'xgboost_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"XGBoost model saved to {model_path}")
    print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    
    return model

def create_random_forest_model():
    """Create and train Random Forest model."""
    print("Creating Random Forest model...")
    
    # Generate training data
    X, y = generate_synthetic_data(10000)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"Random Forest model saved to {model_path}")
    print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    
    return model

def create_label_encoder():
    """Create and save label encoder."""
    print("Creating label encoder...")
    
    # Create label encoder with failure types
    encoder = LabelEncoder()
    failure_types = [
        'No Failure',
        'Engine Failure', 
        'Brake System Issues',
        'Battery Problems',
        'Tire Pressure Warning',
        'General Maintenance Required'
    ]
    
    encoder.fit(failure_types)
    
    # Save the encoder
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    joblib.dump(encoder, encoder_path)
    
    print(f"Label encoder saved to {encoder_path}")
    return encoder

def create_feature_names():
    """Create and save feature names."""
    print("Creating feature names...")
    
    feature_names = [
        'Engine_Temperature_(°C)',
        'Tire_Pressure_(PSI)',
        'Brake_Pad_Thickness_(mm)',
        'Anomaly_Indication',
        'is_engine_failure',
        'is_brake_failure',
        'is_battery_failure',
        'is_low_tire_pressure',
        'is_maintenance_required'
    ]
    
    # Save feature names
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    joblib.dump(feature_names, features_path)
    
    print(f"Feature names saved to {features_path}")
    return feature_names

def main():
    """Main function to create all missing models."""
    print("🚗 RideAhead - Creating Missing ML Models")
    print("=" * 50)
    
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Create XGBoost model
        create_xgboost_model()
        print()
        
        # Create Random Forest model
        create_random_forest_model()
        print()
        
        # Create label encoder
        create_label_encoder()
        print()
        
        # Create feature names
        create_feature_names()
        print()
        
        print("✅ All models created successfully!")
        print("\nAvailable models:")
        print("- LightGBM (existing)")
        print("- XGBoost (created)")
        print("- Random Forest (created)")
        print("- LSTM Time Series (existing)")
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
