# generate_synthetic_labels.py
# Adds synthetic failure_type and other labels to vehicle_sensor_data.csv using industry-standard thresholds
import pandas as pd
import numpy as np
import os

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/vehicle_sensor_data.csv'))
df = pd.read_csv(data_path)

# Example thresholds (adjust as needed for your data)
def label_failure(row):
    try:
        # More permissive thresholds for demonstration
        if 'Engine Temperature (°C)' in row and row['Engine Temperature (°C)'] > 90:
            return 'Engine Failure'
        elif 'Tire Pressure (PSI)' in row and row['Tire Pressure (PSI)'] < 33:
            return 'Low Tire Pressure'
        elif 'Brake Pad Thickness (mm)' in row and row['Brake Pad Thickness (mm)'] < 8:
            return 'Brake Failure'
        elif 'Battery Voltage (V)' in row and row['Battery Voltage (V)'] < 12.2:
            return 'Battery Failure'
        elif 'Anomaly Indication' in row and row['Anomaly Indication'] == 1:
            return 'Maintenance Required'
        else:
            return 'No Failure'
    except Exception as e:
        print(f"Error labeling row: {e}")
        return 'No Failure'

# Apply the function to create the synthetic label
df['failure_type'] = df.apply(label_failure, axis=1)

# Optionally, create binary columns for each failure type
df['is_engine_failure'] = (df['failure_type'] == 'Engine Failure').astype(int)
df['is_brake_failure'] = (df['failure_type'] == 'Brake Failure').astype(int)
df['is_battery_failure'] = (df['failure_type'] == 'Battery Failure').astype(int)
df['is_low_tire_pressure'] = (df['failure_type'] == 'Low Tire Pressure').astype(int)
df['is_maintenance_required'] = (df['failure_type'] == 'Maintenance Required').astype(int)

# Save new file for downstream use
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/vehicle_sensor_data_synthetic.csv'))
df.to_csv(out_path, index=False)
print(f'Synthetic labels generated and saved to {out_path}')

# Print label distribution for debug
try:
    print('Label distribution:')
    print(df['failure_type'].value_counts())
except Exception as e:
    print(f'Error printing label distribution: {e}')
