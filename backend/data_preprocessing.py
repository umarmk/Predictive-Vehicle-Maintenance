# data_preprocessing.py
# Data Preprocessing Script for Predictive Vehicle Maintenance System
# Loads, cleans, normalizes, and performs basic feature engineering on all datasets in the data folder

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_datasets(data_dir):
    data_paths = {
        'ai4i2020': os.path.join(data_dir, 'ai4i2020.csv'),
        'cleaned_vehicle_data': os.path.join(data_dir, 'cleaned_vehicle_data.csv'),
        'engine_data': os.path.join(data_dir, 'engine_data.csv'),
        'vehicle_sensor_data': os.path.join(data_dir, 'vehicle_sensor_data.csv')
    }
    dfs = {name: pd.read_csv(path) for name, path in data_paths.items() if os.path.exists(path)}
    return dfs

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    # Fill missing values: median for numeric, mode for categorical
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def detect_outliers(df):
    # Simple Z-score method, returns count of outliers per column
    num_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
    outlier_counts = (z_scores > 3).sum()
    return outlier_counts

def normalize_data(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    return df_scaled

def feature_engineering(df):
    # Example: flag for high mileage and temperature above average
    if 'mileage' in df.columns:
        df['high_mileage'] = (df['mileage'] > 100000).astype(int)
    if 'temperature' in df.columns:
        df['temp_above_avg'] = (df['temperature'] > df['temperature'].mean()).astype(int)
    return df

def preprocess_all(data_dir):
    dfs = load_datasets(data_dir)
    processed = {}
    for name, df in dfs.items():
        print(f'Processing {name}...')
        df = clean_data(df)
        outliers = detect_outliers(df)
        print('Potential outliers per column:')
        print(outliers)
        df = feature_engineering(df)
        df_scaled = normalize_data(df)
        processed[name] = df_scaled
        # Optionally, save processed file
        df_scaled.to_csv(os.path.join(data_dir, f'{name}_processed.csv'), index=False)
        print(f'Saved: {name}_processed.csv')
    return processed

if __name__ == "__main__":
    # Set your data directory relative to this script
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    preprocess_all(data_dir)
