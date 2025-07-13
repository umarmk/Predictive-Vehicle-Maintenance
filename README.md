# RideAhead - Predictive Vehicle Maintenance System

A comprehensive machine learning-based system for predicting vehicle maintenance needs using sensor data and historical patterns. Built with modern web technologies and advanced ML algorithms for real-world vehicle maintenance applications.

## Features

### **Advanced Prediction Capabilities**

- **Multi-Model Classification**: LightGBM, XGBoost, and Random Forest models for failure type prediction
- **Hybrid Time Series Forecasting**: LSTM-based engine temperature prediction with intelligent trend analysis for extreme temperatures
- **Multiple Issue Detection**: Simultaneous detection of engine, brake, and tire issues
- **Intelligent Anomaly Handling**: Smart anomaly indication that only triggers maintenance when other values are normal
- **Safety-Critical Temperature Analysis**: Emergency detection for dangerous engine temperatures above 120°C

### **User-Friendly Interface**

- **Interactive Web Dashboard**: React-based frontend with real-time predictions
- **Visual Status Indicators**: Color-coded alerts with gradient backgrounds and icons
- **Quick Test Scenarios**: One-click sample data for immediate testing
- **Comprehensive Recommendations**: Actionable advice for each detected issue
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### **Robust Authentication System**

- **User Registration/Login**: Secure user account management
- **Session Management**: Token-based authentication with automatic expiry
- **Protected Routes**: Secure access to prediction features
- **User Profile Display**: Username and email shown in header

### **Enhanced Analytics**

- **Prediction History**: Track and analyze past predictions with detailed statistics
- **Performance Metrics**: Visual charts showing vehicle health trends
- **Advanced Temperature Visualization**: Interactive charts with color-coded safety zones and reference lines
- **Risk Assessment**: Intelligent severity classification (low/medium/high/emergency)
- **Real-time Safety Indicators**: Immediate warnings for critical and emergency temperature conditions

## Technology Stack

### Backend

- **Python 3.8+** - Core programming language
- **Flask** - Web framework with CORS support
- **scikit-learn** - Machine learning models
- **LightGBM** - Primary gradient boosting framework
- **XGBoost** - Alternative gradient boosting model
- **PyTorch** - Deep learning for LSTM models
- **SQLite** - User authentication database
- **pandas & numpy** - Data manipulation and numerical computing

### Frontend

- **React 18** - Modern UI framework
- **TypeScript** - Type-safe JavaScript development
- **Tailwind CSS** - Utility-first CSS framework
- **Recharts** - Interactive data visualization
- **React Router** - Client-side routing
- **Context API** - State management for authentication

### Security & Authentication

- **PBKDF2** - Secure password hashing
- **Session Tokens** - JWT-like token management
- **Protected Routes** - Frontend route protection
- **CORS Configuration** - Secure cross-origin requests

## Installation & Setup

### Prerequisites

- **Python 3.8+** installed
- **Node.js 16+** installed
- **Git** for version control

### Quick Start

1. **Clone the repository:**

```bash
git clone https://github.com/umarmk/Predictive-Vehicle-Maintenance.git
cd Predictive-Vehicle-Maintenance
```

2. **Backend Setup:**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

3. **Frontend Setup (new terminal):**

```bash
cd frontend
npm install
npm run dev
```

4. **Access the application:**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:5000`

## Usage Guide

### **Getting Started**

1. Navigate to `http://localhost:5173`
2. Create a new account or sign in
3. Access the dashboard to see vehicle overview

### **Classification Predictions**

- Enter vehicle sensor data manually or use quick test scenarios:
  - **Normal Vehicle**: Healthy baseline parameters
  - **Engine Issue**: High temperature scenario (115°C)
  - **Brake Issue**: Low brake pad thickness (3mm)
  - **Tire Issue**: Low tire pressure (20 PSI)

### **Engine Temperature Forecasting**

- Input historical temperature readings or use sample patterns:
  - **Normal Operation**: Stable temperature around 85°C with LSTM predictions
  - **Rising Temperature**: Gradual increase pattern with trend analysis
  - **Rapid Overheating**: Emergency temperature escalation with safety warnings
  - **Cooling Down**: Temperature reduction pattern with recovery analysis
- **Interactive Charts**: Color-coded temperature zones with reference lines for Normal (75-95°C), Warning (95-105°C), Critical (105-120°C), and Emergency (120°C+)
- **Safety Features**: Immediate emergency alerts for dangerous temperatures with actionable recommendations

### **Dashboard Analytics**

- View vehicle health overview
- Monitor prediction history
- Analyze performance trends
- Track maintenance recommendations

## Model Information

### Classification Models

| Model             | Purpose            | Accuracy | Features         |
| ----------------- | ------------------ | -------- | ---------------- |
| **LightGBM**      | Primary classifier | High     | Fast, efficient  |
| **XGBoost**       | Alternative model  | High     | Robust, reliable |
| **Random Forest** | Ensemble backup    | Good     | Interpretable    |

**Prediction Classes:**

- **No Failure** (0) - Vehicle operating normally
- **Engine Failure** (1) - Overheating or mechanical issues
- **Brake System Issues** (2) - Worn pads or pressure problems
- **Battery Problems** (3) - Low voltage or charging issues
- **Tire Pressure Warning** (4) - Under/over-inflated tires
- **General Maintenance Required** (5) - Routine service needs

### Time Series Model

- **Architecture**: Hybrid LSTM Neural Network with intelligent trend analysis
- **Input**: Historical engine temperature sequences (configurable length)
- **Output**: Future temperature forecasts with safety classifications
- **Features**:
  - Configurable sequence length and forecast horizon
  - Automatic switching between LSTM (normal temps) and trend-based prediction (extreme temps)
  - Real-time safety analysis with emergency detection
  - Temperature validation and anomaly handling

## Project Structure

```
Predictive-Vehicle-Maintenance/
├── backend/
│   ├── models/              # Trained ML models (.pkl, .pt files)
│   ├── app.py              # Main Flask application
│   ├── auth.py             # Authentication system
│   ├── time_series.py      # LSTM model implementation
│   ├── requirements.txt    # Python dependencies
│   └── users.db           # SQLite user database
├── frontend/
│   ├── src/
│   │   ├── components/     # Reusable React components
│   │   ├── pages/          # Page components
│   │   ├── contexts/       # React contexts (Theme, Auth)
│   │   ├── api/            # API client configuration
│   │   └── types/          # TypeScript type definitions
│   ├── package.json        # Node.js dependencies
│   └── tailwind.config.js  # Tailwind CSS configuration
├── README.md               # This file
├── ProjectPlan.md          # Detailed project documentation
└── .gitignore             # Git ignore rules
```

## 🔧 API Reference

### Authentication Endpoints

- `POST /api/auth/register` - Create new user account
- `POST /api/auth/login` - User authentication
- `POST /api/auth/logout` - End user session
- `GET /api/auth/verify` - Verify session token

### Prediction Endpoints

- `GET /api/info` - API information and available models
- `POST /api/predict` - Classification predictions
- `POST /api/predict/timeseries` - Time series forecasting
- `GET /api/history` - Prediction history

### Sample Request (Classification)

```json
{
  "model": "lightgbm_model",
  "data": {
    "Engine_Temperature_(°C)": 115,
    "Tire_Pressure_(PSI)": 32,
    "Brake_Pad_Thickness_(mm)": 3,
    "Anomaly_Indication": 1
  }
}
```

## Key Improvements Made

### **Enhanced Prediction Logic**

- Multiple failure detection in single prediction
- Smart anomaly indication handling
- Hybrid time series forecasting with extreme temperature handling
- Comprehensive recommendation system with emergency protocols

### **Visual Enhancements**

- Advanced interactive charts with temperature safety zones
- Color-coded severity indicators with emergency alerts
- Professional temperature visualization with reference lines
- User-friendly tooltips with safety information

### **Safety Features**

- Emergency temperature detection above 120°C
- Real-time safety warnings and actionable recommendations
- Intelligent temperature validation and anomaly handling
- Critical temperature progression analysis

### **Security Features**

- Complete user authentication system
- Protected routes and session management
- Secure password hashing (PBKDF2)
- User profile display in header

### **User Experience**

- One-click sample data testing with realistic patterns
- Enhanced chart readability with proper axis formatting
- Improved error handling and validation
- Comprehensive safety documentation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built for predictive maintenance research and development
- Uses state-of-the-art machine learning techniques
- Designed for real-world vehicle maintenance applications
- Implements modern web development best practices

---
