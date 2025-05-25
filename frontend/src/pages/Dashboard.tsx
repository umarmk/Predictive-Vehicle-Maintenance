import React, { useEffect, useState } from 'react';
import Card from '../components/ui/Card';
import api from '../api/client';
import Button from '../components/ui/Button';

// Simple Dashboard component
const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [modelStatus, setModelStatus] = useState<'loaded' | 'error' | 'checking'>('checking');
  const [checkingStatus, setCheckingStatus] = useState(false);

  // Function to check API and model status
  const checkApiStatus = async () => {
    setCheckingStatus(true);
    try {
      // Check API health
      try {
        await api.getHealthStatus();
        setApiStatus('online');
        setModelStatus('loaded');
      } catch (err) {
        console.error('API health check failed:', err);
        setApiStatus('offline');
        setModelStatus('error');
      }
    } catch (err) {
      console.error('Error checking API status:', err);
      setApiStatus('offline');
      setModelStatus('error');
    } finally {
      setCheckingStatus(false);
      setLoading(false);
    }
  };

  // Initial data loading
  useEffect(() => {
    checkApiStatus();
  }, []);

  return (
    <div className="space-y-6">
      {/* Vehicle Health Dashboard */}
      <Card title="Vehicle Health Status">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Overall Health</h3>
            <span className="px-3 py-1 rounded-full text-sm font-medium bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200">
              85%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
            <div
              className="h-2.5 rounded-full bg-success-500"
              style={{ width: '85%' }}
            ></div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Engine */}
          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Engine</h3>
            <div className="text-xl font-bold text-success-600 dark:text-success-400">
              90%
            </div>
          </div>

          {/* Brakes */}
          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Brakes</h3>
            <div className="text-xl font-bold text-primary-600 dark:text-primary-400">
              75%
            </div>
          </div>

          {/* Battery */}
          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Battery</h3>
            <div className="text-xl font-bold text-success-600 dark:text-success-400">
              95%
            </div>
          </div>

          {/* Tires */}
          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Tires</h3>
            <div className="text-xl font-bold text-warning-600 dark:text-warning-400">
              65%
            </div>
          </div>

          {/* Oil */}
          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Oil</h3>
            <div className="text-xl font-bold text-warning-600 dark:text-warning-400">
              60%
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <h4 className="text-md font-semibold text-gray-800 dark:text-gray-200 mb-2">Maintenance Recommendations</h4>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start">
              <span className="inline-block w-2 h-2 rounded-full mt-1.5 mr-2 bg-warning-500"></span>
              <span className="text-gray-700 dark:text-gray-300">Tire tread is wearing. Rotation recommended within 5,000 miles.</span>
            </li>
            <li className="flex items-start">
              <span className="inline-block w-2 h-2 rounded-full mt-1.5 mr-2 bg-warning-500"></span>
              <span className="text-gray-700 dark:text-gray-300">Oil change needed soon. Schedule service within 2 weeks.</span>
            </li>
          </ul>
        </div>
      </Card>

      {/* System Status Card */}
      <Card
        title="System Status"
        headerAction={
          <Button
            variant="outline"
            size="sm"
            onClick={checkApiStatus}
            isLoading={checkingStatus}
          >
            Refresh
          </Button>
        }
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex justify-between items-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">API Status</span>
            {apiStatus === 'checking' ? (
              <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200">
                Checking...
              </span>
            ) : apiStatus === 'online' ? (
              <span className="px-2 py-1 text-xs rounded-full bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200">
                Online
              </span>
            ) : (
              <span className="px-2 py-1 text-xs rounded-full bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200">
                Offline
              </span>
            )}
          </div>
          <div className="flex justify-between items-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Classification Model</span>
            {modelStatus === 'checking' ? (
              <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200">
                Checking...
              </span>
            ) : modelStatus === 'loaded' ? (
              <span className="px-2 py-1 text-xs rounded-full bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200">
                Loaded
              </span>
            ) : (
              <span className="px-2 py-1 text-xs rounded-full bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200">
                Error
              </span>
            )}
          </div>
          <div className="flex justify-between items-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Time Series Model</span>
            {modelStatus === 'checking' ? (
              <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200">
                Checking...
              </span>
            ) : modelStatus === 'loaded' ? (
              <span className="px-2 py-1 text-xs rounded-full bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200">
                Loaded
              </span>
            ) : (
              <span className="px-2 py-1 text-xs rounded-full bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200">
                Error
              </span>
            )}
          </div>
        </div>
      </Card>

      {/* Quick Actions Card */}
      <Card title="Quick Actions">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <a
            href="/classification"
            className="flex items-center p-4 bg-primary-50 dark:bg-primary-900 rounded-lg hover:bg-primary-100 dark:hover:bg-primary-800 transition-colors"
          >
            <div className="rounded-full bg-primary-100 dark:bg-primary-800 p-3 mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-primary-600 dark:text-primary-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-primary-700 dark:text-primary-300">Predict Failures</h3>
              <p className="text-sm text-primary-600 dark:text-primary-400">
                Run classification models to predict potential vehicle failures
              </p>
            </div>
          </a>
          <a
            href="/timeseries"
            className="flex items-center p-4 bg-secondary-50 dark:bg-secondary-900 rounded-lg hover:bg-secondary-100 dark:hover:bg-secondary-800 transition-colors"
          >
            <div className="rounded-full bg-secondary-100 dark:bg-secondary-800 p-3 mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-secondary-600 dark:text-secondary-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-secondary-700 dark:text-secondary-300">Forecast Metrics</h3>
              <p className="text-sm text-secondary-600 dark:text-secondary-400">
                Use time series models to forecast future component metrics
              </p>
            </div>
          </a>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
