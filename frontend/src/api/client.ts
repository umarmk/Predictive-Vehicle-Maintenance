import axios from 'axios';
import type { AxiosError, AxiosResponse, AxiosRequestConfig } from 'axios';
import type {
  ClassificationInput,
  ClassificationResult,
  TimeSeriesInput,
  TimeSeriesResult,
  ExplainInput,
  ExplainResult,
  HistoryResponse
} from '../types';

// Create axios instance with base URL
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

// Add request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.method?.toUpperCase()} ${response.config.url}`);
    return response;
  },
  (error: AxiosError) => {
    const errorMessage = error.response?.data?.error || error.message;
    const status = error.response?.status;
    const url = error.config?.url;
    const method = error.config?.method?.toUpperCase();

    console.error(`API Error (${status}) [${method} ${url}]:`, errorMessage);

    // Return a more structured error object
    return Promise.reject({
      status,
      message: errorMessage,
      originalError: error,
      isNetworkError: !error.response && error.code === 'ECONNABORTED',
      isServerError: status && status >= 500,
      isClientError: status && status >= 400 && status < 500
    });
  }
);

// Retry mechanism for failed requests
const withRetry = async <T>(
  apiCall: () => Promise<T>,
  retries = 3,
  delay = 1000
): Promise<T> => {
  try {
    return await apiCall();
  } catch (error: any) {
    // Only retry on network errors or server errors (5xx)
    if (retries > 0 && (error.isNetworkError || error.isServerError)) {
      console.log(`Retrying API call... (${retries} attempts left)`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return withRetry(apiCall, retries - 1, delay * 1.5); // Exponential backoff
    }
    throw error;
  }
};

// API functions
const api = {
  // Classification prediction
  predict: async (input: ClassificationInput): Promise<ClassificationResult> => {
    return withRetry(async () => {
      console.log('API Request: POST /api/predict with payload:', {
        model: 'lightgbm_model',
        data: input
      });

      const response: AxiosResponse<any> = await apiClient.post('/api/predict', {
        model: 'lightgbm_model',
        data: input
      });

      console.log('API Response:', response.status, 'POST /api/predict', response.data);

      // Transform the response to match the expected ClassificationResult type
      // Get the prediction from the backend response
      let prediction = response.data.prediction;

      console.log('Original prediction from backend:', prediction);

      // Handle different response formats
      // If prediction is an array, use the first element
      if (Array.isArray(prediction)) {
        prediction = prediction[0];
        console.log('Using first element of prediction array:', prediction);
      }

      // Ensure prediction is a number
      if (typeof prediction === 'string') {
        prediction = parseInt(prediction, 10);
        console.log('Converted string prediction to number:', prediction);
        if (isNaN(prediction)) {
          prediction = 0; // Default to 'No Failure' if parsing fails
          console.log('Invalid prediction value, defaulting to 0 (No Failure)');
        }
      }

      // Determine failure type based on input parameters
      // This is a fallback logic that will only be used if the backend prediction is 0 (No Failure)
      if (prediction === 0) {
        // Only override the prediction if Anomaly_Indication is 1 or if there are specific failure indicators
        if (input["Anomaly_Indication"] === 1) {
          // Check if a specific issue type was selected
          if (input["selected_issue_type"]) {
            const issueType = input["selected_issue_type"] as string;
            console.log(`Using selected issue type: ${issueType}`);

            switch (issueType) {
              case 'engine_failure':
                prediction = 1; // Engine Failure
                console.log('Selected issue type is engine_failure, setting prediction to 1 (Engine Failure)');
                break;
              case 'brake_failure':
                prediction = 2; // Brake System Issues
                console.log('Selected issue type is brake_failure, setting prediction to 2 (Brake System Issues)');
                break;
              case 'battery_failure':
                prediction = 3; // Battery Problems
                console.log('Selected issue type is battery_failure, setting prediction to 3 (Battery Problems)');
                break;
              case 'tire_pressure':
                prediction = 4; // Tire Pressure Warning
                console.log('Selected issue type is tire_pressure, setting prediction to 4 (Tire Pressure Warning)');
                break;
              case 'general_maintenance':
                prediction = 5; // General Maintenance Required
                console.log('Selected issue type is general_maintenance, setting prediction to 5 (General Maintenance Required)');
                break;
              default:
                prediction = 5; // Default to General Maintenance Required
                console.log('Unknown issue type, defaulting to 5 (General Maintenance Required)');
            }
          } else {
            prediction = 5; // General Maintenance Required
            console.log('Anomaly Indication is set to 1 but no issue type selected, setting prediction to 5 (General Maintenance Required)');
          }
        }
        // Only check these conditions if Anomaly_Indication is 1 or if the failure indicators are explicitly set
        else if (input["is_engine_failure"] === 1 ||
                (input["Engine_Temperature_(°C)"] !== undefined && input["Engine_Temperature_(°C)"] > 95)) {
          prediction = 1; // Engine Failure
          console.log('Detected engine failure condition, setting prediction to 1 (Engine Failure)');
        }
        else if (input["is_brake_failure"] === 1 ||
                (input["Brake_Pad_Thickness_(mm)"] !== undefined && input["Brake_Pad_Thickness_(mm)"] < 5)) {
          prediction = 2; // Brake System Issues
          console.log('Detected brake system issue, setting prediction to 2 (Brake System Issues)');
        }
        else if (input["is_battery_failure"] === 1) {
          prediction = 3; // Battery Problems
          console.log('Battery failure indicator is set, setting prediction to 3 (Battery Problems)');
        }
        else if (input["is_low_tire_pressure"] === 1 ||
                (input["Tire_Pressure_(PSI)"] !== undefined &&
                (input["Tire_Pressure_(PSI)"] < 25 || input["Tire_Pressure_(PSI)"] > 40))) {
          prediction = 4; // Tire Pressure Warning
          console.log('Detected tire pressure issue, setting prediction to 4 (Tire Pressure Warning)');
        }
        else if (input["is_maintenance_required"] === 1) {
          prediction = 5; // General Maintenance Required
          console.log('Maintenance required indicator is set, setting prediction to 5 (General Maintenance Required)');
        }
        // If Anomaly_Indication is 0 and no other conditions are met, keep prediction as 0 (No Failure)
        else {
          console.log('No anomaly or failure conditions detected, keeping prediction as 0 (No Failure)');
        }
      }

      console.log('Final prediction after processing:', prediction);

      const result: ClassificationResult = {
        prediction: prediction,
        predictions: Array.isArray(response.data.prediction)
          ? response.data.prediction
          : [prediction],
        probabilities: response.data.probability,
        success: response.data.success || true,
        timestamp: new Date().toISOString()
      };

      console.log('Transformed result:', result);
      return result;
    });
  },

  // Time series prediction
  predictTimeSeries: async (input: TimeSeriesInput): Promise<TimeSeriesResult> => {
    return withRetry(async () => {
      console.log('API Request: POST /api/predict/timeseries with payload:', {
        model: 'engine_temp_predictor',
        series: input.series,
        seq_length: input.seq_length || 10,
        horizon: input.horizon || 5
      });

      const response: AxiosResponse<any> = await apiClient.post('/api/predict/timeseries', {
        model: 'engine_temp_predictor',
        series: input.series,
        seq_length: input.seq_length || 10,
        horizon: input.horizon || 5
      });

      console.log('API Response:', response.status, 'POST /api/predict/timeseries', response.data);

      // Transform the response to match the expected TimeSeriesResult type
      const result: TimeSeriesResult = {
        predictions: response.data.predictions || (response.data.prediction ? [response.data.prediction] : []),
        prediction: response.data.prediction,
        success: response.data.success || true,
        timestamp: new Date().toISOString()
      };

      console.log('Transformed time series result:', result);
      return result;
    });
  },

  // Explainability functionality removed

  // History
  getHistory: async (limit?: number, type?: 'classification' | 'timeseries'): Promise<HistoryResponse> => {
    return withRetry(async () => {
      const params: Record<string, string | number> = {};
      if (limit) params.limit = limit;
      if (type) params.type = type;

      const response: AxiosResponse<HistoryResponse> = await apiClient.get('/api/history', { params });
      return response.data;
    });
  },

  // Get API health status
  getHealthStatus: async (): Promise<boolean> => {
    try {
      await apiClient.get('/api/info');
      return true;
    } catch (error) {
      console.error('API health check failed:', error);
      return false;
    }
  },

  // Get model information
  getModels: async () => {
    return withRetry(async () => {
      const response = await apiClient.get('/api/models');
      return response.data;
    });
  }
};

export default api;
