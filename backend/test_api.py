"""Basic API integration tests using :mod:`unittest`.

These tests assume the Flask application is already running locally on port
``5000``. They exercise both the ``/predict`` endpoint for the LightGBM model
and the ``/predict/timeseries`` endpoint used for LSTM forecasting. Only the
most important aspects are validated: successful HTTP responses and minimal
structure of the returned JSON payloads.
"""

import unittest
from typing import Any, Dict

import requests


BASE_URL = "http://127.0.0.1:5000"


class APITestCase(unittest.TestCase):
    """Collection of integration tests for the Flask API."""

    def _post_json(self, path: str, payload: Dict[str, Any]) -> requests.Response:
        """Helper to POST JSON data to an endpoint."""
        return requests.post(f"{BASE_URL}{path}", json=payload)

    def test_predict_lightgbm(self) -> None:
        """``/predict`` should return predictions for a valid payload."""

        payload = {
            "Engine_Temperature_(°C)": 90.0,
            "Brake_Pad_Thickness_(mm)": 10.0,
            "Tire_Pressure_(PSI)": 32.0,
            "Anomaly_Indication": 0,
            "is_engine_failure": 1,
            "is_brake_failure": 0,
            "is_battery_failure": 0,
            "is_low_tire_pressure": 0,
            "is_maintenance_required": 0,
        }

        response = self._post_json("/predict", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)

    def test_predict_missing_feature(self) -> None:
        """``/predict`` should reject incomplete payloads."""

        payload = {"Engine_Temperature_(°C)": 90.0}
        response = self._post_json("/predict", payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_timeseries_default(self) -> None:
        """``/predict/timeseries`` with a simple series should succeed."""

        payload = {"series": list(range(1, 21))}
        response = self._post_json("/predict/timeseries", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(
            "prediction" in data or "predictions" in data,
            msg="Timeseries endpoint should return prediction(s)",
        )

    def test_predict_timeseries_horizon_one(self) -> None:
        """Multi-step forecasting with ``horizon`` set to 1."""

        payload = {
            "series": list(range(1, 21)),
            "horizon": 1,
            "seq_length": 10,
        }
        response = self._post_json("/predict/timeseries", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("prediction", data)

    def test_predict_timeseries_horizon_three(self) -> None:
        """Multi-step forecasting with ``horizon`` greater than 1."""

        payload = {
            "series": list(range(1, 21)),
            "horizon": 3,
            "seq_length": 10,
        }
        response = self._post_json("/predict/timeseries", payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 3)

    def test_history_endpoint(self) -> None:
        """``/history`` should respond successfully."""

        response = requests.get(f"{BASE_URL}/history")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)


if __name__ == "__main__":
    unittest.main()

