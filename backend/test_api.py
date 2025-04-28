import requests

# Test /predict endpoint for LightGBM and timeseries
base_url = "http://127.0.0.1:5000"

def test_predict_lightgbm():
    payload = {
        "Engine_Temperature_(°C)": 90.0,
        "Brake_Pad_Thickness_(mm)": 10.0,
        "Tire_Pressure_(PSI)": 32.0,
        "Anomaly_Indication": 0,
        "is_engine_failure": 1,
        "is_brake_failure": 0,
        "is_battery_failure": 0,
        "is_low_tire_pressure": 0,
        "is_maintenance_required": 0
    }
    r = requests.post(f"{base_url}/predict", json=payload)
    try:
        print(f"/predict (lightgbm):", r.status_code, r.json())
    except Exception:
        print(f"/predict (lightgbm):", r.status_code, r.text)

def test_predict_timeseries():
    payload = {
        "series": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    }
    r = requests.post(f"{base_url}/predict/timeseries", json=payload)
    print("/predict/timeseries:", r.status_code, r.json())

def test_predict_missing_feature():
    # Incomplete payload: missing required features
    payload = {"Engine_Temperature_(°C)": 90.0}
    r = requests.post(f"{base_url}/predict", json=payload)
    print("/predict missing features:", r.status_code, r.json())

def test_explain_shap():
    # Valid explain request using SHAP
    payload = {
        "input": {
            "Engine_Temperature_(°C)": 90.0,
            "Brake_Pad_Thickness_(mm)": 10.0,
            "Tire_Pressure_(PSI)": 32.0,
            "Anomaly_Indication": 0,
            "is_engine_failure": 1,
            "is_brake_failure": 0,
            "is_battery_failure": 0,
            "is_low_tire_pressure": 0,
            "is_maintenance_required": 0
        },
        "method": "shap"
    }
    r = requests.post(f"{base_url}/explain", json=payload)
    print("/explain shap:", r.status_code, r.json())

def test_explain_missing_input():
    # Missing input key
    r = requests.post(f"{base_url}/explain", json={})
    print("/explain missing input:", r.status_code, r.json())

# New Phase 5 tests for multi-step forecasting and history
def test_predict_timeseries_horizon_1():
    payload = {
        "series": list(range(1, 21)),
        "horizon": 1,
        "seq_length": 10
    }
    r = requests.post(f"{base_url}/predict/timeseries", json=payload)
    print("/predict/timeseries horizon=1:", r.status_code, r.json())

def test_predict_timeseries_horizon_3():
    payload = {
        "series": list(range(1, 21)),
        "horizon": 3,
        "seq_length": 10
    }
    r = requests.post(f"{base_url}/predict/timeseries", json=payload)
    print("/predict/timeseries horizon=3:", r.status_code, r.json())

def test_history_endpoint():
    r = requests.get(f"{base_url}/history")
    print("/history:", r.status_code, r.json())

if __name__ == "__main__":
    test_predict_lightgbm()
    test_predict_missing_feature()
    test_predict_timeseries()
    test_predict_timeseries_horizon_1()
    test_predict_timeseries_horizon_3()
    test_history_endpoint()
    test_explain_shap()
    test_explain_missing_input()
