# Predictive Vehicle Maintenance System

A comprehensive web-based predictive vehicle maintenance system using advanced machine learning and a modern frontend. The system predicts key maintenance issues (Engine Failure, Brake Failure, Battery Failure, Low Tire Pressure, and Maintenance/Service Requirements) using sensor datasets.

## Project Structure

```
Predictive-Vehicle-Maintenance/
├── backend/               # Backend code (APIs, ML models, etc.)
├── data/                  # Raw and processed datasets
├── docker/                # Docker and deployment configs
├── docs/                  # Project documentation
├── frontend/              # Frontend web application
├── instance/              # (For Flask or similar frameworks)
├── notebooks/             # Jupyter notebooks and preprocessing scripts
│   ├── data_preprocessing.ipynb
│   ├── eda.ipynb
│   └── data_preprocessing.py
└── README.md              # Project overview and instructions
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/umarmk/Predictive-Vehicle-Maintenance.git
   cd Predictive-Vehicle-Maintenance
   ```

2. **Install dependencies:**
   ```bash
   pip install flask flask-cors pandas numpy scikit-learn imbalanced-learn lightgbm torch joblib mlflow shap lime requests
   ```

3. **Data Preprocessing:**
   ```bash
   cd backend
   python data_preprocessing.py
   ```
   Or explore with the Jupyter notebooks in `/notebooks`.

4. **Model Training:**
   ```bash
   python model_development.py
   ```

5. **Start the API server:**
   ```bash
   python app.py
   ```

6. **Run tests:**
   ```bash
   python test_api.py
   ```

## Project Phases
- Phase 1: Data Preparation & EDA (complete)
- Phase 2: Model Development & Optimization (complete)
- Phase 3: API Integration & Automated Testing (complete)
- Phase 4: Explainability Module (SHAP/LIME) (complete)
- Phase 5: Time Series Forecasting Improvements (in progress)
- Phase 6: Deployment & Dockerization (planned)
- Phase 7: Frontend Dashboard (React) (planned)
- Phase 8: CI/CD & Monitoring (planned)

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, flask, flask-cors, imbalanced-learn, lightgbm, torch, joblib, mlflow, shap, lime, requests

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
