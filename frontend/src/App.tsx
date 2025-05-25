import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import SimpleLayout from './components/layout/SimpleLayout';
import ProtectedRoute from './components/auth/ProtectedRoute';
import AuthPage from './pages/AuthPage';
import Card from './components/ui/Card';
import Button from './components/ui/Button';
import Input from './components/ui/Input';
import TimeSeriesPage from './pages/TimeSeriesPage';
import HistoryPage from './pages/HistoryPage';
import Dashboard from './pages/Dashboard';
import api from './api/client';
import type { ClassificationInput, HistoryItem } from './types';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Sector
} from 'recharts';

// Colors for the pie chart
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#FF6B6B'];

// Home page component
const HomePage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Card title="Welcome">
      <p className="text-gray-600 dark:text-gray-300 mb-4">
        Welcome to the Predictive Vehicle Maintenance System. This application helps predict vehicle maintenance needs
        using machine learning algorithms.
      </p>
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div
          className="bg-primary-50 dark:bg-primary-900 p-4 rounded-lg cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => navigate('/classification')}
        >
          <h3 className="font-semibold text-primary-700 dark:text-primary-300">Classification</h3>
          <p className="text-sm text-primary-600 dark:text-primary-400">
            Predict potential vehicle failures based on sensor data.
          </p>
        </div>
        <div
          className="bg-secondary-50 dark:bg-secondary-900 p-4 rounded-lg cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => navigate('/dashboard')}
        >
          <h3 className="font-semibold text-secondary-700 dark:text-secondary-300">Dashboard</h3>
          <p className="text-sm text-secondary-600 dark:text-secondary-400">
            View analytics and insights about your vehicle's health.
          </p>
        </div>
      </div>
    </Card>
  );
};

// Dashboard page component
const DashboardPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [historyData, setHistoryData] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Fetch history data for the dashboard
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const response = await api.getHistory(10);
        setHistoryData(response.history);
        setError(null);
      } catch (err: any) {
        console.error('Error fetching history:', err);
        setError('Failed to load prediction history. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  // Format the timestamp
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  // Map prediction code to human-readable failure type
  const getFailureType = (code: number): string => {
    const failureTypes: Record<number, string> = {
      0: 'No Failure',
      1: 'Engine Failure',
      2: 'Brake System Issues',
      3: 'Battery Problems',
      4: 'Tire Pressure Warning',
      5: 'General Maintenance Required'
    };

    return failureTypes[code] || `Unknown (${code})`;
  };

  // Generate data for the performance metrics chart
  const generatePerformanceData = () => {
    // This would ideally come from an API, but we'll generate mock data for now
    return [
      { name: 'Jan', engineTemp: 85, tirePressure: 32, brakePad: 9 },
      { name: 'Feb', engineTemp: 88, tirePressure: 31, brakePad: 8.5 },
      { name: 'Mar', engineTemp: 90, tirePressure: 30, brakePad: 8 },
      { name: 'Apr', engineTemp: 92, tirePressure: 31, brakePad: 7.5 },
      { name: 'May', engineTemp: 95, tirePressure: 30, brakePad: 7 },
      { name: 'Jun', engineTemp: 98, tirePressure: 29, brakePad: 6.5 },
    ];
  };

  // Generate data for the prediction distribution chart
  const generatePredictionDistribution = () => {
    // Count the number of each prediction type
    const counts = {
      'No Failure': 0,
      'Engine Failure': 0,
      'Brake System Issues': 0,
      'Battery Problems': 0,
      'Tire Pressure Warning': 0,
      'General Maintenance Required': 0
    };

    // Count from history data
    historyData.forEach(item => {
      if (item.type === 'classification' && item.prediction) {
        const failureType = getFailureType(item.prediction[0]);
        counts[failureType] = (counts[failureType] || 0) + 1;
      }
    });

    // Convert to array for the chart
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  };

  const handleRefresh = async () => {
    try {
      setLoading(true);
      const response = await api.getHistory(10);
      setHistoryData(response.history);
      setError(null);
    } catch (err: any) {
      console.error('Error refreshing history:', err);
      setError('Failed to refresh data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Custom label renderer for the pie chart
  const renderCustomizedLabel = (props: any) => {
    const { cx, cy, midAngle, innerRadius, outerRadius, percent, index, name, value } = props;

    // Only show labels for segments with non-zero values
    if (value === 0) return null;

    // Calculate the position of the label
    const RADIAN = Math.PI / 180;
    // Adjust radius based on the segment's position to avoid overlapping
    let radius = outerRadius + 35;

    // Adjust radius for specific angles to prevent overlap
    if (midAngle > 45 && midAngle < 135) {
      radius += 15; // Top labels
    } else if (midAngle > 225 && midAngle < 315) {
      radius += 15; // Bottom labels
    }

    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    // Get the first word of the name (e.g., "Battery" from "Battery Failure")
    const shortName = name.split(' ')[0];

    // Determine text anchor based on position
    const textAnchor = x > cx ? 'start' : 'end';

    return (
      <g>
        {/* Draw a small line from pie to label */}
        <path
          d={`M${cx + (outerRadius * 0.95) * Math.cos(-midAngle * RADIAN)},${cy + (outerRadius * 0.95) * Math.sin(-midAngle * RADIAN)}L${x - (x > cx ? 5 : -5)},${y}`}
          stroke={COLORS[index % COLORS.length]}
          fill="none"
          strokeWidth={1}
        />
        {/* Draw the label text */}
        <text
          x={x}
          y={y}
          fill={COLORS[index % COLORS.length]}
          textAnchor={textAnchor}
          dominantBaseline="central"
          fontSize={14}
          fontWeight="bold"
        >
          {`${shortName}: ${(percent * 100).toFixed(0)}%`}
        </text>
      </g>
    );
  };

  return (
    <div className="space-y-6">
      <Card title="Vehicle Overview">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-success-50 dark:bg-success-900 p-4 rounded-lg">
            <h3 className="font-semibold text-success-700 dark:text-success-300">Vehicle Status</h3>
            <p className="text-3xl font-bold text-success-600 dark:text-success-400">Good</p>
          </div>
          <div className="bg-warning-50 dark:bg-warning-900 p-4 rounded-lg">
            <h3 className="font-semibold text-warning-700 dark:text-warning-300">Maintenance Due</h3>
            <p className="text-3xl font-bold text-warning-600 dark:text-warning-400">15 days</p>
          </div>
          <div className="bg-primary-50 dark:bg-primary-900 p-4 rounded-lg">
            <h3 className="font-semibold text-primary-700 dark:text-primary-300">Total Predictions</h3>
            <p className="text-3xl font-bold text-primary-600 dark:text-primary-400">
              {historyData.length}
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card
          title="Performance Metrics"
          headerAction={
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              isLoading={loading}
            >
              Refresh
            </Button>
          }
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={generatePerformanceData()}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="engineTemp" stroke="#8884d8" name="Engine Temp (°C)" />
                <Line type="monotone" dataKey="tirePressure" stroke="#82ca9d" name="Tire Pressure (PSI)" />
                <Line type="monotone" dataKey="brakePad" stroke="#ffc658" name="Brake Pad (mm)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Prediction Distribution">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart margin={{ top: 30, right: 50, left: 50, bottom: 30 }}>
                <Pie
                  data={generatePredictionDistribution()}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomizedLabel}
                  outerRadius={70}
                  innerRadius={0}
                  paddingAngle={3}
                  fill="#8884d8"
                  dataKey="value"
                  isAnimationActive={false} // Disable animation for better label positioning
                >
                  {generatePredictionDistribution().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value} predictions`, '']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card
        title="Recent Predictions"
        headerAction={
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            isLoading={loading}
          >
            Refresh
          </Button>
        }
      >
        {loading ? (
          <div className="py-8 text-center">
            <svg className="animate-spin h-8 w-8 text-primary-500 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="mt-2 text-gray-600 dark:text-gray-400">Loading prediction history...</p>
          </div>
        ) : error ? (
          <div className="py-8 text-center text-danger-600 dark:text-danger-400">
            {error}
          </div>
        ) : historyData.length === 0 ? (
          <div className="py-8 text-center text-gray-600 dark:text-gray-400">
            No predictions found. Start by making a prediction.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Date
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Type
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Result
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {historyData.map((item, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatTimestamp(item.timestamp)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        item.type === 'classification'
                          ? 'bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200'
                          : 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
                      }`}>
                        {item.type === 'classification' ? 'Classification' : 'Time Series'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {item.type === 'classification'
                        ? getFailureType(item.prediction?.[0] || 0)
                        : `${item.predictions?.length || 1} forecast values`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
};

// Classification page component
const ClassificationPage: React.FC = () => {
  const [formData, setFormData] = useState<Partial<ClassificationInput>>({
    "Engine_Temperature_(°C)": 90,
    "Tire_Pressure_(PSI)": 32,
    "Brake_Pad_Thickness_(mm)": 8,
    "Anomaly_Indication": 0,
    "is_engine_failure": 0,
    "is_brake_failure": 0,
    "is_battery_failure": 0,
    "is_low_tire_pressure": 0,
    "is_maintenance_required": 0
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (name: string, value: number) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      // Make sure all required fields are present
      const input = formData as ClassificationInput;

      // Ensure Anomaly_Indication is properly set
      if (input["Anomaly_Indication"] === 1) {
        console.log('Anomaly Indication is set to 1, this should trigger a failure prediction');

        // If no specific issue type is selected, default to general maintenance
        if (!input["selected_issue_type"]) {
          console.log('No specific issue type selected, defaulting to general maintenance');
          input["is_maintenance_required"] = 1;
        }
      }

      const response = await api.predict(input);
      console.log('Prediction response:', response);
      console.log('Setting result to:', response.predictions);
      setResult(response.predictions);
    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.error || 'Failed to make prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Analyze input data for potential issues
  const analyzeInputData = (inputData: any) => {
    const issues = [];
    const engineTemp = inputData["Engine_Temperature_(°C)"] || 0;
    const tirePressure = inputData["Tire_Pressure_(PSI)"] || 0;
    const brakePad = inputData["Brake_Pad_Thickness_(mm)"] || 0;

    // Engine temperature analysis
    if (engineTemp > 110) {
      issues.push({
        type: 'Engine Failure',
        severity: 'high',
        details: `Engine temperature is critically high (${engineTemp}°C). Normal range: 75-95°C`,
        icon: '🔥',
        color: 'danger'
      });
    } else if (engineTemp > 100) {
      issues.push({
        type: 'Engine Failure',
        severity: 'medium',
        details: `Engine temperature is elevated (${engineTemp}°C). Monitor closely`,
        icon: '⚠️',
        color: 'warning'
      });
    }

    // Tire pressure analysis
    if (tirePressure < 25 || tirePressure > 40) {
      issues.push({
        type: 'Low Tire Pressure',
        severity: tirePressure < 20 ? 'high' : 'medium',
        details: `Tire pressure is ${tirePressure < 25 ? 'too low' : 'too high'} (${tirePressure} PSI). Recommended: 30-35 PSI`,
        icon: '🛞',
        color: tirePressure < 20 ? 'danger' : 'warning'
      });
    }

    // Brake pad analysis
    if (brakePad < 4) {
      issues.push({
        type: 'Brake Failure',
        severity: 'high',
        details: `Brake pad thickness is critically low (${brakePad}mm). Replace immediately`,
        icon: '🛑',
        color: 'danger'
      });
    } else if (brakePad < 6) {
      issues.push({
        type: 'Brake Failure',
        severity: 'medium',
        details: `Brake pad thickness is low (${brakePad}mm). Schedule replacement soon`,
        icon: '⚠️',
        color: 'warning'
      });
    }

    return issues;
  };

  // Enhanced prediction analysis with multiple failure detection
  const analyzePrediction = (result: any, inputData: any) => {
    console.log('analyzePrediction called with result:', result, 'inputData:', inputData);
    if (!result || result.length === 0) {
      console.log('No result or empty result, returning null');
      return null;
    }

    // The result is an array where result[0] is the prediction code
    const predictionCode = result[0];
    const predictionType = getFailureType(predictionCode);
    console.log('Prediction code:', predictionCode, 'Type:', predictionType);

    // Analyze input data for potential issues
    const inputIssues = analyzeInputData(inputData);

    // Combine input analysis with model prediction
    const allIssues = [...inputIssues];

    // Handle anomaly indication logic
    if (inputData["Anomaly_Indication"] === 1) {
      // Only add maintenance required if no other specific issues are detected
      const hasSpecificIssues = allIssues.length > 0;

      if (!hasSpecificIssues) {
        allIssues.push({
          type: 'Maintenance Required',
          severity: 'medium',
          details: 'Vehicle anomaly detected. General maintenance recommended',
          icon: '🔧',
          color: 'warning'
        });
      }
    }

    // If no issues detected from input analysis and model predicts failure, add model prediction
    if (allIssues.length === 0 && predictionCode !== 0) {
      const severity = predictionCode === 1 ? 'high' : 'medium'; // Engine failure is high priority
      allIssues.push({
        type: predictionType,
        severity: severity,
        details: `Model prediction: ${predictionType}`,
        icon: predictionCode === 1 ? '🔥' : predictionCode === 2 ? '🛑' : predictionCode === 4 ? '🛞' : '🔧',
        color: severity === 'high' ? 'danger' : 'warning'
      });
    }

    return {
      issues: allIssues,
      modelPrediction: predictionType,
      modelPredictionCode: predictionCode,
      overallStatus: allIssues.length === 0 ? 'healthy' :
                    allIssues.some(issue => issue.severity === 'high') ? 'critical' : 'warning'
    };
  };

  // Map prediction code to human-readable failure type
  const getFailureType = (code: number): string => {
    const failureTypes: Record<number, string> = {
      0: 'No Failure',
      1: 'Engine Failure',
      2: 'Brake System Issues',
      3: 'Battery Problems',
      4: 'Tire Pressure Warning',
      5: 'General Maintenance Required'
    };

    return failureTypes[code] || `Unknown (${code})`;
  };

  // Determine if the result indicates a failure
  const isFailure = (code: number): boolean => {
    return code !== 0;
  };

  // Get severity level based on input values
  const getSeverityLevel = (): 'low' | 'medium' | 'high' => {
    const engineTemp = formData["Engine_Temperature_(°C)"] || 0;
    const tirePressure = formData["Tire_Pressure_(PSI)"] || 0;
    const brakePad = formData["Brake_Pad_Thickness_(mm)"] || 0;
    const anomalyIndication = formData["Anomaly_Indication"] || 0;

    // If anomaly indication is set to 1, at least medium severity
    if (anomalyIndication === 1) {
      return 'medium';
    }
    // Check for high severity conditions
    else if (engineTemp > 110 || tirePressure < 25 || tirePressure > 40 || brakePad < 4) {
      return 'high';
    }
    // Check for medium severity conditions
    else if (engineTemp > 100 || tirePressure < 28 || tirePressure > 38 || brakePad < 6) {
      return 'medium';
    }
    // Otherwise, low severity
    else {
      return 'low';
    }
  };

  // Get color class based on severity
  const getSeverityColorClass = (severity: 'low' | 'medium' | 'high'): string => {
    switch (severity) {
      case 'high':
        return 'text-danger-600 dark:text-danger-400';
      case 'medium':
        return 'text-warning-600 dark:text-warning-400';
      case 'low':
        return 'text-success-600 dark:text-success-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  // Explainability functions removed

  return (
    <div className="space-y-6">
      <Card title="Vehicle Failure Prediction">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-700 dark:text-gray-300">Input Parameters</h3>
            <Input
              label="Engine Temperature (°C)"
              type="number"
              value={formData["Engine_Temperature_(°C)"]}
              onChange={(e) => handleInputChange("Engine_Temperature_(°C)", Number(e.target.value))}
              min={0}
              max={150}
            />
            <Input
              label="Tire Pressure (PSI)"
              type="number"
              value={formData["Tire_Pressure_(PSI)"]}
              onChange={(e) => handleInputChange("Tire_Pressure_(PSI)", Number(e.target.value))}
              min={0}
              max={50}
            />
            <Input
              label="Brake Pad Thickness (mm)"
              type="number"
              value={formData["Brake_Pad_Thickness_(mm)"]}
              onChange={(e) => handleInputChange("Brake_Pad_Thickness_(mm)", Number(e.target.value))}
              min={0}
              max={20}
            />
            <Input
              label="Anomaly Indication"
              type="number"
              value={formData["Anomaly_Indication"]}
              onChange={(e) => handleInputChange("Anomaly_Indication", Number(e.target.value))}
              min={0}
              max={1}
              helperText="0 for normal, 1 for anomaly"
            />

            {/* Sample Data Buttons */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Quick Test Scenarios
              </label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFormData({
                      "Engine_Temperature_(°C)": 85,
                      "Tire_Pressure_(PSI)": 32,
                      "Brake_Pad_Thickness_(mm)": 8,
                      "Anomaly_Indication": 0,
                      "is_engine_failure": 0,
                      "is_brake_failure": 0,
                      "is_battery_failure": 0,
                      "is_low_tire_pressure": 0,
                      "is_maintenance_required": 0
                    });
                  }}
                >
                  Normal Vehicle
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFormData({
                      "Engine_Temperature_(°C)": 115,
                      "Tire_Pressure_(PSI)": 32,
                      "Brake_Pad_Thickness_(mm)": 8,
                      "Anomaly_Indication": 1,
                      "is_engine_failure": 1,
                      "is_brake_failure": 0,
                      "is_battery_failure": 0,
                      "is_low_tire_pressure": 0,
                      "is_maintenance_required": 0
                    });
                  }}
                >
                  Engine Issue
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFormData({
                      "Engine_Temperature_(°C)": 90,
                      "Tire_Pressure_(PSI)": 32,
                      "Brake_Pad_Thickness_(mm)": 3,
                      "Anomaly_Indication": 1,
                      "is_engine_failure": 0,
                      "is_brake_failure": 1,
                      "is_battery_failure": 0,
                      "is_low_tire_pressure": 0,
                      "is_maintenance_required": 0
                    });
                  }}
                >
                  Brake Issue
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFormData({
                      "Engine_Temperature_(°C)": 90,
                      "Tire_Pressure_(PSI)": 20,
                      "Brake_Pad_Thickness_(mm)": 8,
                      "Anomaly_Indication": 1,
                      "is_engine_failure": 0,
                      "is_brake_failure": 0,
                      "is_battery_failure": 0,
                      "is_low_tire_pressure": 1,
                      "is_maintenance_required": 0
                    });
                  }}
                >
                  Tire Issue
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Risk Level
                </label>
                <div className={`text-lg font-semibold ${getSeverityColorClass(getSeverityLevel())}`}>
                  {getSeverityLevel() === 'low' ? 'Low' : getSeverityLevel() === 'medium' ? 'Medium' : 'High'}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Recommended Action
                </label>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {getSeverityLevel() === 'low'
                    ? 'Regular maintenance'
                    : getSeverityLevel() === 'medium'
                      ? 'Schedule check-up soon'
                      : 'Immediate attention required'}
                </div>
              </div>
            </div>
            <Button
              variant="primary"
              fullWidth
              isLoading={loading}
              onClick={handlePredict}
            >
              Predict Failure
            </Button>
          </div>

          <div>
            <h3 className="font-semibold text-gray-700 dark:text-gray-300 mb-4">Prediction Results</h3>
            {error && (
              <div className="bg-danger-50 dark:bg-danger-900 p-6 rounded-lg text-center">
                <p className="text-danger-700 dark:text-danger-300">
                  {error}
                </p>
              </div>
            )}
            {result && !error ? (() => {
              const analysis = analyzePrediction(result, formData);
              if (!analysis) {
                // Fallback to simple display if analysis fails
                return (
                  <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg text-center">
                    <p className="text-gray-600 dark:text-gray-400">
                      Prediction: {getFailureType(result[0])}
                    </p>
                  </div>
                );
              }

              return (
                <div className="space-y-4">
                  {/* Overall Status Card */}
                  <div className={`p-6 rounded-lg text-center ${
                    analysis.overallStatus === 'healthy'
                      ? 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900 dark:to-emerald-900 border border-green-200 dark:border-green-700'
                      : analysis.overallStatus === 'critical'
                      ? 'bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900 dark:to-orange-900 border border-red-200 dark:border-red-700'
                      : 'bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900 dark:to-amber-900 border border-yellow-200 dark:border-yellow-700'
                  }`}>
                    <div className="flex items-center justify-center mb-3">
                      <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl ${
                        analysis.overallStatus === 'healthy'
                          ? 'bg-green-100 dark:bg-green-800'
                          : analysis.overallStatus === 'critical'
                          ? 'bg-red-100 dark:bg-red-800'
                          : 'bg-yellow-100 dark:bg-yellow-800'
                      }`}>
                        {analysis.overallStatus === 'healthy' ? '✅' : analysis.overallStatus === 'critical' ? '🚨' : '⚠️'}
                      </div>
                    </div>
                    <h3 className={`text-2xl font-bold mb-2 ${
                      analysis.overallStatus === 'healthy'
                        ? 'text-green-700 dark:text-green-300'
                        : analysis.overallStatus === 'critical'
                        ? 'text-red-700 dark:text-red-300'
                        : 'text-yellow-700 dark:text-yellow-300'
                    }`}>
                      {analysis.overallStatus === 'healthy'
                        ? 'Vehicle Status: Healthy'
                        : analysis.overallStatus === 'critical'
                        ? 'Critical Issues Detected'
                        : 'Attention Required'}
                    </h3>
                    <p className={`${
                      analysis.overallStatus === 'healthy'
                        ? 'text-green-600 dark:text-green-400'
                        : analysis.overallStatus === 'critical'
                        ? 'text-red-600 dark:text-red-400'
                        : 'text-yellow-600 dark:text-yellow-400'
                    }`}>
                      {analysis.overallStatus === 'healthy'
                        ? 'Your vehicle is operating within normal parameters.'
                        : analysis.overallStatus === 'critical'
                        ? 'Immediate attention required. Do not drive until issues are resolved.'
                        : 'Some issues detected. Schedule maintenance soon.'}
                    </p>
                  </div>

                  {/* Issues List */}
                  {analysis.issues.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-700 dark:text-gray-300 flex items-center">
                        <span className="mr-2">🔍</span>
                        Detected Issues ({analysis.issues.length})
                      </h4>
                      {analysis.issues.map((issue, index) => (
                        <div key={index} className={`p-4 rounded-lg border-l-4 ${
                          issue.color === 'danger'
                            ? 'bg-red-50 dark:bg-red-900/20 border-red-500'
                            : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500'
                        }`}>
                          <div className="flex items-start">
                            <span className="text-2xl mr-3 mt-1">{issue.icon}</span>
                            <div className="flex-1">
                              <h5 className={`font-medium ${
                                issue.color === 'danger'
                                  ? 'text-red-700 dark:text-red-300'
                                  : 'text-yellow-700 dark:text-yellow-300'
                              }`}>
                                {issue.type}
                              </h5>
                              <p className={`text-sm mt-1 ${
                                issue.color === 'danger'
                                  ? 'text-red-600 dark:text-red-400'
                                  : 'text-yellow-600 dark:text-yellow-400'
                              }`}>
                                {issue.details}
                              </p>
                              <span className={`inline-block mt-2 px-2 py-1 rounded-full text-xs font-medium ${
                                issue.severity === 'high'
                                  ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100'
                                  : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100'
                              }`}>
                                {issue.severity === 'high' ? 'High Priority' : 'Medium Priority'}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Recommendations */}
                  {analysis.issues.length > 0 && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700">
                      <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-3 flex items-center">
                        <span className="mr-2">💡</span>
                        Recommended Actions
                      </h4>
                      <div className="space-y-2">
                        {analysis.issues.map((issue, index) => (
                          <div key={index} className="text-sm text-blue-600 dark:text-blue-400">
                            <strong>{issue.type}:</strong>
                            <ul className="list-disc pl-5 mt-1 space-y-1">
                              {issue.type === 'Engine Failure' && (
                                <>
                                  <li>Check engine coolant levels immediately</li>
                                  <li>Inspect radiator and cooling system</li>
                                  <li>Verify thermostat functionality</li>
                                  <li>Schedule emergency service if temperature is critical</li>
                                </>
                              )}
                              {issue.type === 'Brake Failure' && (
                                <>
                                  <li>Inspect brake pads and rotors</li>
                                  <li>Check brake fluid levels and quality</li>
                                  <li>Test brake system pressure</li>
                                  <li>Replace brake pads if thickness is below 4mm</li>
                                </>
                              )}
                              {issue.type === 'Low Tire Pressure' && (
                                <>
                                  <li>Check all tires with a pressure gauge</li>
                                  <li>Inflate to recommended PSI (30-35 PSI)</li>
                                  <li>Inspect for punctures or damage</li>
                                  <li>Monitor pressure weekly</li>
                                </>
                              )}
                              {issue.type === 'Maintenance Required' && (
                                <>
                                  <li>Schedule comprehensive vehicle inspection</li>
                                  <li>Check all fluid levels</li>
                                  <li>Inspect belts, hoses, and filters</li>
                                  <li>Review manufacturer's maintenance schedule</li>
                                </>
                              )}
                            </ul>
                          </div>
                        ))}
                      </div>
                      <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-800 rounded text-sm text-blue-700 dark:text-blue-300">
                        <strong>⚠️ Important:</strong> This system provides predictive insights based on sensor data.
                        Always consult with qualified technicians for actual vehicle maintenance and repairs.
                      </div>
                    </div>
                  )}
                </div>
              );
            })() : (
              <div className="flex items-center justify-center h-64 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
                <p className="text-gray-500 dark:text-gray-400">
                  {loading ? 'Processing...' : 'No prediction yet'}
                </p>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Explanation card removed */}

      <Card title="How It Works">
        <div className="prose dark:prose-invert max-w-none">
          <p>
            Our classification model analyzes vehicle sensor data to predict potential failures before they occur.
            The model has been trained on thousands of vehicle maintenance records to identify patterns that precede
            different types of failures.
          </p>
          <h4>Supported Prediction Types:</h4>
          <ul>
            <li><strong>Engine Failure</strong> - Overheating, coolant issues, or mechanical problems</li>
            <li><strong>Brake System Issues</strong> - Worn brake pads, low fluid, or pressure problems</li>
            <li><strong>Battery Problems</strong> - Low voltage, corrosion, or charging issues</li>
            <li><strong>Tire Pressure Warning</strong> - Under/over-inflated tires or punctures</li>
            <li><strong>General Maintenance Required</strong> - Routine service needs</li>
          </ul>
          <h4>How to Use:</h4>
          <ul>
            <li>Enter your vehicle's current sensor readings</li>
            <li>Use the "Quick Test Scenarios" for sample data</li>
            <li>Click "Predict Failure" to get instant analysis</li>
            <li>Follow the recommended actions for any detected issues</li>
          </ul>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
            <strong>Note:</strong> This system provides predictive insights based on sensor data patterns.
            Always consult with qualified technicians for actual vehicle maintenance and repairs.
          </p>
        </div>
      </Card>
    </div>
  );
};

// App component
const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Public route */}
            <Route path="/auth" element={<AuthPage />} />

            {/* Protected routes */}
            <Route path="/" element={
              <ProtectedRoute>
                <SimpleLayout>
                  <HomePage />
                </SimpleLayout>
              </ProtectedRoute>
            } />
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <SimpleLayout>
                  <Dashboard />
                </SimpleLayout>
              </ProtectedRoute>
            } />
            <Route path="/classification" element={
              <ProtectedRoute>
                <SimpleLayout>
                  <ClassificationPage />
                </SimpleLayout>
              </ProtectedRoute>
            } />
            <Route path="/timeseries" element={
              <ProtectedRoute>
                <SimpleLayout>
                  <TimeSeriesPage />
                </SimpleLayout>
              </ProtectedRoute>
            } />
            <Route path="/history" element={
              <ProtectedRoute>
                <SimpleLayout>
                  <HistoryPage />
                </SimpleLayout>
              </ProtectedRoute>
            } />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;
