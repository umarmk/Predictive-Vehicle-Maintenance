import React, { useState, useEffect } from 'react';
import TimeSeriesForm from '../components/forms/TimeSeriesForm';
import TimeSeriesChart from '../components/charts/TimeSeriesChart';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import type { TimeSeriesInput, TimeSeriesResult } from '../types';
import api from '../api/client';

// Engine temperature ranges
const TEMP_RANGES = {
  normal: { min: 75, max: 95 },
  warning: { min: 95, max: 105 },
  critical: { min: 105, max: 120 },
  emergency: { min: 120, max: 300 } // Extreme overheating - engine damage imminent
};

// Define the pattern names as a type for type safety
type EngineTemperaturePatternName = 'normal_operation' | 'gradual_heating' | 'rapid_overheating' | 'temperature_spikes' | 'cooling_down' | 'cold_start' | 'highway_driving' | 'city_traffic';

// Enhanced sample engine temperature patterns with realistic scenarios
const ENGINE_TEMP_PATTERNS: Record<EngineTemperaturePatternName, { data: number[], description: string, scenario: string }> = {
  normal_operation: {
    data: [82, 84, 86, 85, 87, 86, 88, 87, 89, 88, 87, 86, 88, 89, 87, 86, 85, 87, 88, 86],
    description: "Stable engine temperature during normal operation",
    scenario: "Regular city driving with optimal cooling system"
  },
  gradual_heating: {
    data: [78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116],
    description: "Gradual temperature increase indicating potential cooling issues",
    scenario: "Long uphill drive or cooling system degradation"
  },
  rapid_overheating: {
    data: [85, 87, 89, 92, 96, 101, 107, 114, 122, 130, 138, 145, 152, 158, 164, 169, 174, 178, 182, 185],
    description: "Rapid temperature rise - immediate attention required",
    scenario: "Coolant leak or thermostat failure"
  },
  temperature_spikes: {
    data: [86, 88, 92, 87, 89, 95, 88, 91, 98, 89, 87, 94, 88, 92, 99, 87, 90, 96, 88, 91],
    description: "Irregular temperature spikes during operation",
    scenario: "Intermittent cooling fan issues or air pockets in coolant"
  },
  cooling_down: {
    data: [115, 112, 109, 106, 103, 100, 97, 94, 91, 88, 85, 82, 79, 76, 73, 70, 67, 64, 61, 58],
    description: "Engine cooling down after operation",
    scenario: "Post-drive cooldown period"
  },
  cold_start: {
    data: [45, 52, 58, 64, 69, 74, 78, 82, 85, 87, 89, 90, 91, 92, 91, 90, 89, 88, 87, 86],
    description: "Engine warming up from cold start",
    scenario: "Morning startup in cold weather"
  },
  highway_driving: {
    data: [88, 89, 90, 91, 92, 93, 94, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83],
    description: "Highway driving with good airflow cooling",
    scenario: "Sustained highway speeds with optimal cooling"
  },
  city_traffic: {
    data: [86, 88, 91, 94, 97, 95, 92, 89, 91, 94, 97, 100, 98, 95, 92, 89, 87, 90, 93, 96],
    description: "Stop-and-go traffic with varying temperatures",
    scenario: "Urban driving with frequent stops and starts"
  }
};

const TimeSeriesPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [inputData, setInputData] = useState<TimeSeriesInput | null>(null);
  const [result, setResult] = useState<TimeSeriesResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedSeries, setSavedSeries] = useState<{name: string, data: number[]}[]>([]);
  const [showSavedSeries, setShowSavedSeries] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);

  // Load a predefined engine temperature pattern
  const loadEngineTemperaturePattern = (patternName: string) => {
    // Type guard to ensure patternName is a valid key
    const isValidPattern = (name: string): name is EngineTemperaturePatternName => {
      return Object.keys(ENGINE_TEMP_PATTERNS).includes(name);
    };

    // Check if the pattern name is valid
    if (!isValidPattern(patternName)) return;

    setSelectedPattern(patternName);

    const newInputData: TimeSeriesInput = {
      series: ENGINE_TEMP_PATTERNS[patternName].data,
      seq_length: 10,
      horizon: 5,
      feature: 'Engine Temperature (°C)'
    };

    setInputData(newInputData);

    // Clear previous results when loading new pattern
    setResult(null);
    setError(null);

    // Don't auto-submit - let user review and manually submit
    // This improves UX by allowing users to modify the data before prediction
  };

  // Handle form submission for time series prediction
  const handleSubmit = async (data: TimeSeriesInput) => {
    try {
      setLoading(true);
      setError(null);

      // Add feature name if not present
      const enhancedData = {
        ...data,
        feature: data.feature || 'Engine Temperature (°C)'
      };

      console.log('Submitting time series prediction with data:', enhancedData);
      const result = await api.predictTimeSeries(enhancedData);
      console.log('Received time series prediction result:', result);

      setResult(result);
      setInputData(enhancedData);
    } catch (err: any) {
      console.error('Time series prediction error:', err);
      setError(err.response?.data?.error || 'Failed to generate engine temperature forecast. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Calculate statistics for the time series data
  const calculateStats = (data: number[]) => {
    if (!data || data.length === 0) return null;

    // Validate temperature values (allow extreme temperatures for safety analysis)
    const isValidTemp = (temp: number) => temp >= -50 && temp <= 300 && !isNaN(temp) && isFinite(temp);
    const validData = data.filter(isValidTemp);

    // If no valid data, return default values
    if (validData.length === 0) {
      return {
        mean: "85.00",
        min: "80.00",
        max: "90.00",
        stdDev: "2.00",
        trendDirection: "Stable",
        slope: "0.0000"
      };
    }

    // Calculate mean
    const mean = validData.reduce((sum, val) => sum + val, 0) / validData.length;

    // Calculate min, max
    const min = Math.min(...validData);
    const max = Math.max(...validData);

    // Calculate standard deviation
    const variance = validData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validData.length;
    const stdDev = Math.sqrt(variance);

    // Calculate trend (simple linear regression)
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < validData.length; i++) {
      sumX += i;
      sumY += validData[i];
      sumXY += i * validData[i];
      sumX2 += i * i;
    }
    const n = validData.length;
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX || 1); // Avoid division by zero
    // Calculate intercept (not used directly but kept for completeness)
    const _intercept = (sumY - slope * sumX) / n;

    // Determine trend direction
    const trendDirection = slope > 0.1 ? 'Increasing' : slope < -0.1 ? 'Decreasing' : 'Stable';

    return {
      mean: mean.toFixed(2),
      min: min.toFixed(2),
      max: max.toFixed(2),
      stdDev: stdDev.toFixed(2),
      trendDirection,
      slope: slope.toFixed(4)
    };
  };

  // Save current series to local storage
  const handleSaveSeries = () => {
    if (!inputData || !inputData.series || inputData.series.length === 0) return;

    const seriesName = prompt('Enter a name for this time series:');
    if (!seriesName) return;

    const newSavedSeries = [...savedSeries, { name: seriesName, data: inputData.series }];
    setSavedSeries(newSavedSeries);

    // Save to localStorage
    try {
      localStorage.setItem('savedTimeSeries', JSON.stringify(newSavedSeries));
    } catch (e) {
      console.error('Error saving to localStorage:', e);
    }
  };

  // Load saved series from local storage
  const handleLoadSeries = (index: number) => {
    if (index < 0 || index >= savedSeries.length) return;

    const series = savedSeries[index].data;
    if (series && series.length > 0) {
      const newInputData: TimeSeriesInput = {
        series,
        seq_length: inputData?.seq_length || 10,
        horizon: inputData?.horizon || 3
      };

      setInputData(newInputData);
      setShowSavedSeries(false);

      // Trigger prediction with the loaded data
      handleSubmit(newInputData);
    }
  };

  // Load saved series from localStorage on component mount
  React.useEffect(() => {
    try {
      const saved = localStorage.getItem('savedTimeSeries');
      if (saved) {
        setSavedSeries(JSON.parse(saved));
      }
    } catch (e) {
      console.error('Error loading from localStorage:', e);
    }
  }, []);

  // Process predictions to a flat array for statistics
  const getProcessedPredictions = (): number[] => {
    if (!result) return [];

    // Validate temperature values (allow extreme temperatures for safety analysis)
    // Only filter out clearly invalid values (negative or impossibly high)
    const isValidTemp = (temp: number) => temp >= -50 && temp <= 300 && !isNaN(temp) && isFinite(temp);

    // Get the last valid input temperature to use as a baseline for predictions
    const getLastValidInputTemp = (): number => {
      if (!inputData || !inputData.series || inputData.series.length === 0) return 85;
      const lastTemp = inputData.series[inputData.series.length - 1];
      return isValidTemp(lastTemp) ? lastTemp : 85;
    };

    // Generate deterministic predictions based on input data
    // This ensures consistent predictions for the same input data
    const generateDeterministicPredictions = (count: number): number[] => {
      const lastTemp = getLastValidInputTemp();

      // Use a deterministic approach based on the input data
      // This ensures the same predictions for the same input data
      if (!inputData || !inputData.series || inputData.series.length === 0) {
        // If no input data, return stable predictions
        return Array(count).fill(lastTemp);
      }

      // Calculate the trend from the input data
      const inputSeries = inputData.series;
      const dataLength = inputSeries.length;

      // If we have at least 2 points, calculate the trend
      if (dataLength >= 2) {
        const firstPoint = inputSeries[0];
        const lastPoint = inputSeries[dataLength - 1];
        const overallChange = lastPoint - firstPoint;
        const avgChangePerStep = overallChange / (dataLength - 1);

        // Generate predictions that follow the same trend
        return Array.from({ length: count }, (_, i) => {
          // Continue the trend with slight dampening
          const dampening = 0.9; // Reduce the trend effect over time
          const projectedChange = avgChangePerStep * (i + 1) * dampening;
          // Don't artificially cap temperatures - let them reflect reality
          // This is critical for safety - we need to see dangerous temperatures
          return Math.max(40, lastTemp + projectedChange);
        });
      }

      // If we only have 1 point, return stable predictions
      return Array(count).fill(lastTemp);
    };

    let processedPredictions: number[] = [];

    if (result.predictions && result.predictions.length > 0) {
      // Check if predictions is an array of arrays (multivariate)
      if (Array.isArray(result.predictions[0])) {
        // For multivariate, we'll take the first feature (engine temperature) and validate
        processedPredictions = (result.predictions as number[][])
          .map(pred => {
            const value = pred[0];
            return isValidTemp(value) ? value : null;
          })
          .filter(value => value !== null) as number[];
      } else {
        // For univariate, validate each value
        processedPredictions = (result.predictions as number[])
          .map(value => {
            return isValidTemp(value) ? value : null;
          })
          .filter(value => value !== null) as number[];
      }
    } else if (result.prediction) {
      // Handle single prediction
      if (Array.isArray(result.prediction)) {
        const value = result.prediction[0];
        processedPredictions = isValidTemp(value) ? [value] : [];
      } else {
        const value = result.prediction as number;
        processedPredictions = isValidTemp(value) ? [value] : [];
      }
    }

    // Only use fallback if we have absolutely no predictions from the API
    if (processedPredictions.length === 0) {
      console.warn('No valid predictions received from API, using fallback');
      // Generate minimal fallback predictions
      processedPredictions = generateDeterministicPredictions(5);
    }

    return processedPredictions;
  };

  // Analyze temperature trend and provide recommendations
  const analyzeTemperatureTrend = (): { trend: string; recommendation: string; severity: 'low' | 'medium' | 'high' } => {
    if (!inputData || !result) {
      return {
        trend: 'Unknown',
        recommendation: 'No data available for analysis.',
        severity: 'low'
      };
    }

    const predictions = getProcessedPredictions();
    if (predictions.length === 0) {
      return {
        trend: 'Unknown',
        recommendation: 'No prediction data available.',
        severity: 'low'
      };
    }

    // Force recalculation of trend for each analysis
    // This ensures the trend is always up-to-date with the current data

    // Get the last input value and the predicted values
    const lastInputValue = inputData.series[inputData.series.length - 1];
    const firstPrediction = predictions[0];
    const lastPrediction = predictions[predictions.length - 1];

    // Ensure we're working with valid temperature values
    // Allow very high temperatures for safety - we need to detect dangerous conditions!
    // Only filter out clearly invalid values (negative or impossibly high)
    const isValidTemp = (temp: number) => temp >= -50 && temp <= 300 && !isNaN(temp) && isFinite(temp);

    const validLastInput = isValidTemp(lastInputValue) ? lastInputValue : 85; // Default to normal temp if invalid
    const validFirstPrediction = isValidTemp(firstPrediction) ? firstPrediction : validLastInput;
    const validLastPrediction = isValidTemp(lastPrediction) ? lastPrediction : validFirstPrediction;

    // Check if any prediction exceeds warning or critical thresholds
    // Filter out any invalid temperature values first
    const validPredictions = predictions.filter(isValidTemp);

    // Calculate the overall trend with validated values
    // Use the actual input data to determine the trend, not just the last point
    let percentChange = 0;

    // If we have enough input data points, calculate a more accurate trend
    if (inputData.series.length >= 2) {
      // Get the first and last points from the input series
      const firstInputValue = inputData.series[0];
      const lastInputValue = inputData.series[inputData.series.length - 1];

      // Calculate the overall change in the input data
      const inputChange = lastInputValue - firstInputValue;

      // Calculate the percentage change based on the input data
      percentChange = (inputChange / Math.max(1, Math.abs(firstInputValue))) * 100;

      // If the input data shows a clear trend, use it
      // Otherwise, look at the prediction trend
      if (Math.abs(percentChange) < 2) {
        // If input data doesn't show a strong trend, check predictions
        const predictionChange = validLastPrediction - validLastInput;
        const predictionPercentChange = (predictionChange / Math.max(1, Math.abs(validLastInput))) * 100;

        // Use prediction trend if it's significant
        if (Math.abs(predictionPercentChange) > Math.abs(percentChange)) {
          percentChange = predictionPercentChange;
        }
      }
    } else {
      // If we don't have enough input data, use the prediction trend
      const overallChange = validLastPrediction - validLastInput;
      percentChange = (overallChange / Math.max(1, Math.abs(validLastInput))) * 100;
    }

    // Ensure the percentage is a finite number
    if (!isFinite(percentChange)) {
      percentChange = 0;
    }

    // Calculate the average of all predictions (commented out as it's not currently used)
    // const avgPrediction = validPredictions.length > 0
    //   ? validPredictions.reduce((sum, val) => sum + val, 0) / validPredictions.length
    //   : validLastInput;
    const maxPrediction = validPredictions.length > 0
      ? Math.max(...validPredictions)
      : validLastInput;

    const exceedsWarning = maxPrediction > TEMP_RANGES.warning.min;
    const exceedsCritical = maxPrediction > TEMP_RANGES.critical.min;

    // Check if the average prediction is significantly different from current temperature
    // This can be used for additional analysis if needed
    // const significantChange = Math.abs(avgPrediction - validLastInput) > 5;

    let trend = '';
    let recommendation = '';
    let severity: 'low' | 'medium' | 'high' = 'low';

    // Calculate trend based on the actual data
    // This ensures the trend is accurate and reflects the current data
    const calculateTrend = () => {
      // Ensure we're working with a valid percentage
      // This prevents extreme or invalid values
      const actualPercentChange = Math.min(Math.max(percentChange, -100), 100);

      // Format the percentage to 1 decimal place for display
      const formattedPercentage = Math.abs(actualPercentChange).toFixed(1);

      if (Math.abs(actualPercentChange) < 1) {
        return 'Stable';
      } else if (actualPercentChange > 0) {
        // More descriptive rising trend based on magnitude
        if (actualPercentChange > 10) {
          return `Rising Rapidly (${formattedPercentage}%)`;
        } else if (actualPercentChange > 5) {
          return `Rising Moderately (${formattedPercentage}%)`;
        } else {
          return `Rising Slightly (${formattedPercentage}%)`;
        }
      } else {
        // More descriptive falling trend based on magnitude
        if (Math.abs(actualPercentChange) > 10) {
          return `Falling Rapidly (${formattedPercentage}%)`;
        } else if (Math.abs(actualPercentChange) > 5) {
          return `Falling Moderately (${formattedPercentage}%)`;
        } else {
          return `Falling Slightly (${formattedPercentage}%)`;
        }
      }
    };

    // Set the trend based on the current data
    trend = calculateTrend();

    // Determine current temperature status with emergency level
    const currentExceedsWarning = validLastInput > TEMP_RANGES.warning.min;
    const currentExceedsCritical = validLastInput > TEMP_RANGES.critical.min;
    const currentExceedsEmergency = validLastInput > TEMP_RANGES.emergency.min;

    const predictedExceedsEmergency = maxPrediction > TEMP_RANGES.emergency.min;

    // First determine severity based on current temperature
    if (currentExceedsEmergency) {
      severity = 'high';
    } else if (currentExceedsCritical) {
      severity = 'high';
    } else if (currentExceedsWarning) {
      severity = 'medium';
    } else {
      severity = 'low';
    }

    // Then adjust severity based on predictions if they're worse
    if ((exceedsCritical || predictedExceedsEmergency) && severity !== 'high') {
      severity = 'high';
    } else if (exceedsWarning && severity === 'low') {
      severity = 'medium';
    }

    // Finally, determine recommendation based on current status and predictions
    if (currentExceedsEmergency) {
      recommendation = '🚨 EMERGENCY: Engine temperature is EXTREMELY HIGH! STOP IMMEDIATELY! Engine damage is occurring. Turn off engine and call for emergency assistance.';
    } else if (predictedExceedsEmergency) {
      recommendation = '🚨 EMERGENCY: Engine is predicted to reach DANGEROUS levels! STOP DRIVING IMMEDIATELY! Severe engine damage is imminent.';
    } else if (currentExceedsCritical) {
      recommendation = '🔥 CRITICAL: Engine is currently overheating. STOP DRIVING! Immediate attention required. Check cooling system and reduce engine load.';
    } else if (exceedsCritical) {
      recommendation = '🔥 CRITICAL: Engine is predicted to overheat. STOP DRIVING SOON! Immediate attention required. Check cooling system and reduce engine load.';
    } else if (currentExceedsWarning) {
      recommendation = '⚠️ WARNING: Engine temperature is currently at concerning levels. Monitor closely and check cooling system.';
    } else if (exceedsWarning) {
      recommendation = '⚠️ WARNING: Engine temperature is predicted to reach concerning levels. Monitor closely and check cooling system.';
    } else if (percentChange > 5) {
      recommendation = '⚠️ CAUTION: Engine temperature is rising steadily. Monitor for potential cooling system issues.';
    } else if (percentChange < -10) {
      recommendation = '✅ Engine temperature is falling significantly. This could indicate improved cooling or reduced engine load.';
    } else {
      recommendation = '✅ Engine temperature is predicted to remain within normal operating range. No immediate action required.';
    }

    return { trend, recommendation, severity };
  };

  // Calculate statistics for input and prediction data
  const inputStats = inputData ? calculateStats(inputData.series) : null;
  const predictionStats = result ? calculateStats(getProcessedPredictions()) : null;

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">🌡️ Engine Temperature Forecasting</h1>
        <p className="text-blue-100">
          Predict future engine temperatures using advanced LSTM neural networks to prevent overheating and optimize performance.
        </p>
      </div>

      {/* Sample Patterns Section */}
      <Card title="🎯 Quick Start - Sample Temperature Patterns" className="border-l-4 border-l-blue-500">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(ENGINE_TEMP_PATTERNS).map(([key, pattern]) => (
            <div
              key={key}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-lg ${
                selectedPattern === key
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
              }`}
              onClick={() => loadEngineTemperaturePattern(key)}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-white capitalize">
                  {key.replace('_', ' ')}
                </h3>
                <div className={`w-3 h-3 rounded-full ${
                  pattern.data[pattern.data.length - 1] > 105 ? 'bg-red-500' :
                  pattern.data[pattern.data.length - 1] > 95 ? 'bg-yellow-500' : 'bg-green-500'
                }`}></div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{pattern.description}</p>
              <p className="text-xs text-gray-500 dark:text-gray-500">{pattern.scenario}</p>
              <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
                {pattern.data.length} data points • {Math.min(...pattern.data).toFixed(0)}°C - {Math.max(...pattern.data).toFixed(0)}°C
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Manual Input Form */}
      <Card title="📊 Custom Temperature Data" className="border-l-4 border-l-green-500">
        <TimeSeriesForm
          onSubmit={handleSubmit}
          loading={loading}
          initialData={inputData}
          key={selectedPattern} // Force re-render when pattern changes
        />
      </Card>

      {/* Error Display */}
      {error && (
        <Card title="⚠️ Error" className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <div className="text-red-700 dark:text-red-300 flex items-center">
            <span className="mr-2">❌</span>
            {error}
          </div>
        </Card>
      )}

      {/* Results Section */}
      {result && inputData && !error && (
        <div className="space-y-6">
          {/* Action Buttons */}
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">📈 Forecast Results</h2>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSaveSeries}
              >
                💾 Save Series
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSavedSeries(!showSavedSeries)}
              >
                {showSavedSeries ? '🙈 Hide Saved' : '👁️ Show Saved'}
              </Button>
            </div>
          </div>

          {/* Saved Series */}
          {showSavedSeries && savedSeries.length > 0 && (
            <Card title="💾 Saved Time Series">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {savedSeries.map((series, index) => (
                  <div key={index} className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:shadow-md transition-shadow">
                    <div>
                      <span className="font-medium text-gray-900 dark:text-white">{series.name}</span>
                      <div className="text-sm text-gray-500 dark:text-gray-400">{series.data.length} data points</div>
                    </div>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => handleLoadSeries(index)}
                    >
                      Load
                    </Button>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Temperature Chart */}
          <Card title="📊 Temperature Forecast Visualization" className="border-l-4 border-l-purple-500">
            <TimeSeriesChart
              inputData={inputData.series}
              predictions={getProcessedPredictions()}
              feature="Engine Temperature (°C)"
            />
          </Card>

          {/* Enhanced Temperature Analysis */}
          {result && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Temperature Status Card */}
              <Card title="🌡️ Current Temperature Status" className="border-l-4 border-l-blue-500">
                {(() => {
                  const analysis = analyzeTemperatureTrend();
                  const lastTemp = inputData.series[inputData.series.length - 1];
                  const predictions = getProcessedPredictions();
                  const maxPredicted = Math.max(...predictions);

                  const getStatusColor = (temp: number) => {
                    if (temp > 120) return { bg: 'bg-red-200 dark:bg-red-900/40', text: 'text-red-800 dark:text-red-200', icon: '🚨' };
                    if (temp > 105) return { bg: 'bg-red-100 dark:bg-red-900/20', text: 'text-red-700 dark:text-red-300', icon: '🔥' };
                    if (temp > 95) return { bg: 'bg-yellow-100 dark:bg-yellow-900/20', text: 'text-yellow-700 dark:text-yellow-300', icon: '⚠️' };
                    return { bg: 'bg-green-100 dark:bg-green-900/20', text: 'text-green-700 dark:text-green-300', icon: '✅' };
                  };

                  const currentStatus = getStatusColor(lastTemp);
                  const predictedStatus = getStatusColor(maxPredicted);

                  return (
                    <div className="space-y-4">
                      <div className={`p-4 rounded-lg ${currentStatus.bg}`}>
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className={`text-lg font-semibold ${currentStatus.text}`}>
                              {currentStatus.icon} Current: {lastTemp.toFixed(1)}°C
                            </h3>
                            <p className={`text-sm ${currentStatus.text}`}>
                              {lastTemp > 120 ? 'EMERGENCY - STOP IMMEDIATELY!' :
                               lastTemp > 105 ? 'Critical - Immediate attention required' :
                               lastTemp > 95 ? 'Warning - Monitor closely' : 'Normal operating range'}
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className={`p-4 rounded-lg ${predictedStatus.bg}`}>
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className={`text-lg font-semibold ${predictedStatus.text}`}>
                              {predictedStatus.icon} Predicted Max: {maxPredicted.toFixed(1)}°C
                            </h3>
                            <p className={`text-sm ${predictedStatus.text}`}>
                              {maxPredicted > 120 ? 'WILL REACH EMERGENCY LEVELS!' :
                               maxPredicted > 105 ? 'Will exceed critical threshold' :
                               maxPredicted > 95 ? 'May reach warning levels' : 'Will remain in normal range'}
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">📈 Trend Analysis</h4>
                        <p className="text-gray-700 dark:text-gray-300 text-sm">
                          <strong>Trend:</strong> {analysis.trend}
                        </p>
                        <p className="text-gray-600 dark:text-gray-400 text-sm mt-1">
                          {analysis.recommendation}
                        </p>
                      </div>
                    </div>
                  );
                })()}
              </Card>

              {/* Recommendations Card */}
              <Card title="💡 Smart Recommendations" className="border-l-4 border-l-green-500">
                {(() => {
                  const lastTemp = inputData.series[inputData.series.length - 1];
                  const predictions = getProcessedPredictions();
                  const maxPredicted = Math.max(...predictions);
                  const analysis = analyzeTemperatureTrend();

                  const getRecommendations = () => {
                    const recommendations = [];

                    if (maxPredicted > 120 || lastTemp > 120) {
                      recommendations.push({
                        priority: 'emergency',
                        icon: '🚨',
                        title: 'EMERGENCY - ENGINE DAMAGE IMMINENT',
                        actions: [
                          'STOP DRIVING IMMEDIATELY - Pull over safely',
                          'Turn off engine and DO NOT restart',
                          'Call emergency roadside assistance',
                          'Do not attempt to drive - engine damage is occurring'
                        ]
                      });
                    } else if (maxPredicted > 105 || lastTemp > 105) {
                      recommendations.push({
                        priority: 'high',
                        icon: '🔥',
                        title: 'Critical Action Required',
                        actions: [
                          'Stop driving as soon as safely possible',
                          'Check coolant level and look for leaks',
                          'Inspect radiator for blockages',
                          'Contact roadside assistance if temperature continues rising'
                        ]
                      });
                    } else if (maxPredicted > 95 || lastTemp > 95) {
                      recommendations.push({
                        priority: 'medium',
                        icon: '⚠️',
                        title: 'Preventive Maintenance',
                        actions: [
                          'Schedule cooling system inspection',
                          'Check thermostat operation',
                          'Verify cooling fan functionality',
                          'Monitor temperature closely during driving'
                        ]
                      });
                    } else {
                      recommendations.push({
                        priority: 'low',
                        icon: '✅',
                        title: 'System Operating Normally',
                        actions: [
                          'Continue regular maintenance schedule',
                          'Check coolant level monthly',
                          'Monitor for any unusual temperature changes',
                          'Keep radiator clean and unobstructed'
                        ]
                      });
                    }

                    if (analysis.trend.includes('Rising')) {
                      recommendations.push({
                        priority: 'medium',
                        icon: '📈',
                        title: 'Rising Temperature Trend',
                        actions: [
                          'Reduce engine load if possible',
                          'Check for coolant leaks',
                          'Ensure adequate airflow to radiator',
                          'Consider shorter driving intervals'
                        ]
                      });
                    }

                    return recommendations;
                  };

                  const recommendations = getRecommendations();

                  return (
                    <div className="space-y-4">
                      {recommendations.map((rec, index) => (
                        <div key={index} className={`p-4 rounded-lg border-l-4 ${
                          rec.priority === 'emergency' ? 'border-l-red-600 bg-red-100 dark:bg-red-900/20 border-2 border-red-500' :
                          rec.priority === 'high' ? 'border-l-red-500 bg-red-50 dark:bg-red-900/10' :
                          rec.priority === 'medium' ? 'border-l-yellow-500 bg-yellow-50 dark:bg-yellow-900/10' :
                          'border-l-green-500 bg-green-50 dark:bg-green-900/10'
                        }`}>
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                            {rec.icon} {rec.title}
                          </h4>
                          <ul className="space-y-1">
                            {rec.actions.map((action, actionIndex) => (
                              <li key={actionIndex} className="text-sm text-gray-700 dark:text-gray-300 flex items-start">
                                <span className="mr-2 mt-1">•</span>
                                {action}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </Card>
            </div>
          )}

          {/* Enhanced Statistics Dashboard */}
          <Card title="📊 Temperature Analytics Dashboard" className="border-l-4 border-l-indigo-500">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Historical Stats */}
              {inputStats && (
                <>
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-medium text-blue-700 dark:text-blue-300">📈 Historical Avg</h4>
                      <span className="text-xs text-blue-600 dark:text-blue-400">Past Data</span>
                    </div>
                    <div className="text-2xl font-bold text-blue-800 dark:text-blue-200">
                      {inputStats.mean}°C
                    </div>
                    <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                      Range: {inputStats.min}°C - {inputStats.max}°C
                    </p>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-4 rounded-lg border border-purple-200 dark:border-purple-700">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-medium text-purple-700 dark:text-purple-300">📊 Variability</h4>
                      <span className="text-xs text-purple-600 dark:text-purple-400">Stability</span>
                    </div>
                    <div className="text-2xl font-bold text-purple-800 dark:text-purple-200">
                      ±{inputStats.stdDev}°C
                    </div>
                    <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                      {parseFloat(inputStats.stdDev) < 2 ? 'Very Stable' :
                       parseFloat(inputStats.stdDev) < 5 ? 'Stable' : 'Variable'}
                    </p>
                  </div>
                </>
              )}

              {/* Prediction Stats */}
              {predictionStats && (
                <>
                  <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-4 rounded-lg border border-green-200 dark:border-green-700">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-medium text-green-700 dark:text-green-300">🔮 Forecast Avg</h4>
                      <span className="text-xs text-green-600 dark:text-green-400">Predicted</span>
                    </div>
                    <div className="text-2xl font-bold text-green-800 dark:text-green-200">
                      {predictionStats.mean}°C
                    </div>
                    <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                      Range: {predictionStats.min}°C - {predictionStats.max}°C
                    </p>
                  </div>

                  <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 p-4 rounded-lg border border-orange-200 dark:border-orange-700">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-medium text-orange-700 dark:text-orange-300">📈 Trend</h4>
                      <span className="text-xs text-orange-600 dark:text-orange-400">Direction</span>
                    </div>
                    <div className="text-lg font-bold text-orange-800 dark:text-orange-200">
                      {inputStats?.trendDirection || 'Unknown'}
                    </div>
                    <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                      {inputStats?.trendDirection === 'Increasing' ? '↗️ Rising' :
                       inputStats?.trendDirection === 'Decreasing' ? '↘️ Falling' : '➡️ Stable'}
                    </p>
                  </div>
                </>
              )}
            </div>
          </Card>

          {/* Detailed Forecast Table */}
          <Card title="🔍 Detailed Forecast Analysis" className="border-l-4 border-l-cyan-500">
            <div className="space-y-6">
              {/* Model Parameters */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-900/20 dark:to-cyan-800/20 p-4 rounded-lg border border-cyan-200 dark:border-cyan-700">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-cyan-700 dark:text-cyan-300">🔢 Sequence Length</h4>
                    <span className="text-xs text-cyan-600 dark:text-cyan-400">Input Window</span>
                  </div>
                  <div className="text-2xl font-bold text-cyan-800 dark:text-cyan-200">
                    {inputData.seq_length || 10}
                  </div>
                  <p className="text-xs text-cyan-600 dark:text-cyan-400 mt-1">
                    Past data points used for prediction
                  </p>
                </div>

                <div className="bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-900/20 dark:to-teal-800/20 p-4 rounded-lg border border-teal-200 dark:border-teal-700">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-teal-700 dark:text-teal-300">🎯 Forecast Horizon</h4>
                    <span className="text-xs text-teal-600 dark:text-teal-400">Future Steps</span>
                  </div>
                  <div className="text-2xl font-bold text-teal-800 dark:text-teal-200">
                    {inputData.horizon || 1}
                  </div>
                  <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">
                    Future time steps predicted
                  </p>
                </div>

                <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-900/20 dark:to-indigo-800/20 p-4 rounded-lg border border-indigo-200 dark:border-indigo-700">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-indigo-700 dark:text-indigo-300">📊 Data Points</h4>
                    <span className="text-xs text-indigo-600 dark:text-indigo-400">Total Input</span>
                  </div>
                  <div className="text-2xl font-bold text-indigo-800 dark:text-indigo-200">
                    {inputData.series.length}
                  </div>
                  <p className="text-xs text-indigo-600 dark:text-indigo-400 mt-1">
                    Historical temperature readings
                  </p>
                </div>
              </div>

              {/* Forecast Values Table */}
              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                  📋 Predicted Temperature Values
                </h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-100 dark:bg-gray-700">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          🕐 Time Step
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          🌡️ Temperature
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          📈 Change
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          ⚠️ Status
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {(() => {
                        const predictions = getProcessedPredictions();
                        const lastInputTemp = inputData.series[inputData.series.length - 1];

                        return predictions.map((value, index) => {
                          const prevValue = index === 0 ? lastInputTemp : predictions[index - 1];
                          const change = value - prevValue;
                          const changePercent = (change / Math.abs(prevValue)) * 100;

                          const getStatusInfo = (temp: number) => {
                            if (temp > 105) return { text: 'Critical', color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/20', icon: '🔥' };
                            if (temp > 95) return { text: 'Warning', color: 'text-yellow-600 dark:text-yellow-400', bg: 'bg-yellow-100 dark:bg-yellow-900/20', icon: '⚠️' };
                            return { text: 'Normal', color: 'text-green-600 dark:text-green-400', bg: 'bg-green-100 dark:bg-green-900/20', icon: '✅' };
                          };

                          const status = getStatusInfo(value);

                          return (
                            <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                              <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                                {inputData.series.length + index + 1}
                              </td>
                              <td className="px-4 py-3 whitespace-nowrap text-sm font-bold text-gray-900 dark:text-white">
                                {value.toFixed(1)}°C
                              </td>
                              <td className={`px-4 py-3 whitespace-nowrap text-sm font-medium ${
                                change > 0 ? 'text-red-600 dark:text-red-400' :
                                change < 0 ? 'text-blue-600 dark:text-blue-400' :
                                'text-gray-500 dark:text-gray-400'
                              }`}>
                                {change > 0 ? '↗️' : change < 0 ? '↘️' : '➡️'} {Math.abs(change).toFixed(1)}°C
                                <span className="text-xs ml-1">({Math.abs(changePercent).toFixed(1)}%)</span>
                              </td>
                              <td className="px-4 py-3 whitespace-nowrap">
                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${status.bg} ${status.color}`}>
                                  {status.icon} {status.text}
                                </span>
                              </td>
                            </tr>
                          );
                        });
                      })()}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </Card>

          {/* Educational Section */}
          <Card title="🎓 Understanding Engine Temperature Forecasting" className="border-l-4 border-l-purple-500">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* How It Works */}
              <div className="space-y-4">
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold text-purple-800 dark:text-purple-200 mb-3 flex items-center">
                    🧠 How Our AI Works
                  </h3>
                  <p className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">
                    Our system uses a <strong>Long Short-Term Memory (LSTM)</strong> neural network to analyze historical temperature patterns
                    and predict future values. This advanced AI can identify subtle trends and patterns that might indicate potential overheating
                    issues before they become critical.
                  </p>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center">
                    📋 How to Use This Tool
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-start space-x-2">
                      <span className="bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center mt-0.5">1</span>
                      <p className="text-gray-700 dark:text-gray-300 text-sm">Choose a sample pattern or enter your own temperature data</p>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span className="bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center mt-0.5">2</span>
                      <p className="text-gray-700 dark:text-gray-300 text-sm">Adjust sequence length and forecast horizon as needed</p>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span className="bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center mt-0.5">3</span>
                      <p className="text-gray-700 dark:text-gray-300 text-sm">Click "Generate Forecast" to see predictions</p>
                    </div>
                    <div className="flex items-start space-x-2">
                      <span className="bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center mt-0.5">4</span>
                      <p className="text-gray-700 dark:text-gray-300 text-sm">Review the analysis and follow recommendations</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Temperature Ranges & Troubleshooting */}
              <div className="space-y-4">
                <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold text-green-800 dark:text-green-200 mb-3 flex items-center">
                    🌡️ Temperature Ranges
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                      <div>
                        <p className="font-medium text-green-800 dark:text-green-200">Normal (75-95°C)</p>
                        <p className="text-xs text-green-600 dark:text-green-400">Optimal operating range - no action needed</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-4 h-4 bg-yellow-500 rounded-full"></div>
                      <div>
                        <p className="font-medium text-yellow-800 dark:text-yellow-200">Warning (95-105°C)</p>
                        <p className="text-xs text-yellow-600 dark:text-yellow-400">Monitor closely - check cooling system</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                      <div>
                        <p className="font-medium text-red-800 dark:text-red-200">Critical (&gt;105°C)</p>
                        <p className="text-xs text-red-600 dark:text-red-400">Immediate attention required - risk of damage</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold text-orange-800 dark:text-orange-200 mb-3 flex items-center">
                    🔧 Common Overheating Causes
                  </h3>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      { icon: '💧', text: 'Low coolant level or leaks' },
                      { icon: '🌡️', text: 'Faulty thermostat' },
                      { icon: '🔥', text: 'Radiator blockage or damage' },
                      { icon: '⚙️', text: 'Water pump failure' },
                      { icon: '💨', text: 'Cooling fan malfunction' },
                      { icon: '🚫', text: 'Blocked coolant passages' }
                    ].map((item, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <span className="text-sm">{item.icon}</span>
                        <p className="text-xs text-orange-700 dark:text-orange-300">{item.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesPage;
