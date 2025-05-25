import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea
} from 'recharts';

interface TimeSeriesChartProps {
  inputData: number[];
  predictions: number[];
  feature?: string;
  showReferenceLines?: boolean;
}

// Temperature ranges for reference lines
const TEMP_RANGES = {
  normal: { min: 75, max: 95 },
  warning: { min: 95, max: 105 },
  critical: { min: 105, max: 120 },
  emergency: { min: 120, max: 300 }
};

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  inputData,
  predictions,
  feature = 'Engine Temperature (°C)',
  showReferenceLines = true
}) => {
  // Prepare chart data
  const chartData = React.useMemo(() => {
    const data = [];

    // Add input data points
    inputData.forEach((value, index) => {
      data.push({
        index: index + 1,
        actual: value,
        predicted: null,
        type: 'historical'
      });
    });

    // Add prediction data points
    predictions.forEach((value, index) => {
      data.push({
        index: inputData.length + index + 1,
        actual: null,
        predicted: value,
        type: 'forecast'
      });
    });

    return data;
  }, [inputData, predictions]);

  // Custom tooltip with temperature status
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;

      // Determine temperature status
      const getTemperatureStatus = (temp: number) => {
        if (temp >= TEMP_RANGES.emergency.min) return { status: '🚨 EMERGENCY', color: 'text-red-600 dark:text-red-400' };
        if (temp >= TEMP_RANGES.critical.min) return { status: '🔥 Critical', color: 'text-red-500 dark:text-red-400' };
        if (temp >= TEMP_RANGES.warning.min) return { status: '⚠️ Warning', color: 'text-yellow-600 dark:text-yellow-400' };
        return { status: '✅ Normal', color: 'text-green-600 dark:text-green-400' };
      };

      return (
        <div className="bg-white dark:bg-gray-800 p-4 border border-gray-200 dark:border-gray-600 rounded-lg shadow-xl">
          <p className="text-sm font-bold text-gray-900 dark:text-white mb-2">
            📍 Time Point: {label}
          </p>
          {data.actual !== null && (
            <div className="mb-2">
              <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                📊 Historical: {data.actual.toFixed(1)}°C
              </p>
              <p className={`text-xs font-medium ${getTemperatureStatus(data.actual).color}`}>
                {getTemperatureStatus(data.actual).status}
              </p>
            </div>
          )}
          {data.predicted !== null && (
            <div className="mb-2">
              <p className="text-sm font-medium text-orange-600 dark:text-orange-400">
                🔮 Forecast: {data.predicted.toFixed(1)}°C
              </p>
              <p className={`text-xs font-medium ${getTemperatureStatus(data.predicted).color}`}>
                {getTemperatureStatus(data.predicted).status}
              </p>
            </div>
          )}
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
            <p>Normal: 75-95°C | Warning: 95-105°C</p>
            <p>Critical: 105-120°C | Emergency: 120°C+</p>
          </div>
        </div>
      );
    }
    return null;
  };

  // Determine Y-axis domain with intelligent scaling
  const allValues = [...inputData, ...predictions].filter(v => v != null && isFinite(v));

  if (allValues.length === 0) {
    // Fallback domain if no valid data
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
        <p className="text-gray-500 dark:text-gray-400">No data available for visualization</p>
      </div>
    );
  }

  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);

  // Smart padding based on temperature ranges
  let yMin, yMax;

  if (maxValue <= 100) {
    // Normal temperature range
    yMin = Math.max(40, minValue - 10);
    yMax = Math.min(120, maxValue + 10);
  } else if (maxValue <= 150) {
    // Elevated temperature range
    yMin = Math.max(60, minValue - 15);
    yMax = Math.min(180, maxValue + 15);
  } else {
    // Extreme temperature range
    yMin = Math.max(80, minValue - 20);
    yMax = maxValue + 30;
  }

  const yDomain = [yMin, yMax];

  return (
    <div className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 20,
          }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            className="stroke-gray-200 dark:stroke-gray-700"
          />

          <XAxis
            dataKey="index"
            type="number"
            scale="linear"
            domain={['dataMin', 'dataMax']}
            className="text-gray-600 dark:text-gray-400"
            label={{
              value: 'Time Points',
              position: 'insideBottom',
              offset: -10,
              className: 'fill-gray-600 dark:fill-gray-400'
            }}
          />

          <YAxis
            domain={yDomain}
            className="text-gray-600 dark:text-gray-400"
            tickFormatter={(value) => `${Math.round(value)}°C`}
            width={60}
            label={{
              value: feature,
              angle: -90,
              position: 'insideLeft',
              style: { textAnchor: 'middle' },
              className: 'fill-gray-600 dark:fill-gray-400'
            }}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{
              paddingTop: '20px'
            }}
          />

          {/* Reference lines for temperature ranges */}
          {showReferenceLines && feature.toLowerCase().includes('temperature') && (
            <>
              {/* Only show reference areas that are visible in the current domain */}
              {yDomain[0] <= TEMP_RANGES.normal.max && yDomain[1] >= TEMP_RANGES.normal.min && (
                <ReferenceArea
                  y1={Math.max(TEMP_RANGES.normal.min, yDomain[0])}
                  y2={Math.min(TEMP_RANGES.normal.max, yDomain[1])}
                  fill="#22c55e"
                  fillOpacity={0.15}
                />
              )}

              {yDomain[0] <= TEMP_RANGES.warning.max && yDomain[1] >= TEMP_RANGES.warning.min && (
                <ReferenceArea
                  y1={Math.max(TEMP_RANGES.warning.min, yDomain[0])}
                  y2={Math.min(TEMP_RANGES.warning.max, yDomain[1])}
                  fill="#f59e0b"
                  fillOpacity={0.15}
                />
              )}

              {yDomain[0] <= TEMP_RANGES.critical.max && yDomain[1] >= TEMP_RANGES.critical.min && (
                <ReferenceArea
                  y1={Math.max(TEMP_RANGES.critical.min, yDomain[0])}
                  y2={Math.min(TEMP_RANGES.critical.max, yDomain[1])}
                  fill="#ef4444"
                  fillOpacity={0.15}
                />
              )}

              {yDomain[0] <= TEMP_RANGES.emergency.max && yDomain[1] >= TEMP_RANGES.emergency.min && (
                <ReferenceArea
                  y1={Math.max(TEMP_RANGES.emergency.min, yDomain[0])}
                  y2={Math.min(TEMP_RANGES.emergency.max, yDomain[1])}
                  fill="#dc2626"
                  fillOpacity={0.25}
                />
              )}

              {/* Reference lines - only show if they're within the visible domain */}
              {TEMP_RANGES.warning.min >= yDomain[0] && TEMP_RANGES.warning.min <= yDomain[1] && (
                <ReferenceLine
                  y={TEMP_RANGES.warning.min}
                  stroke="#f59e0b"
                  strokeDasharray="8 4"
                  strokeWidth={2}
                  label={{
                    value: "⚠️ Warning (95°C)",
                    position: "topLeft",
                    style: { fontSize: '12px', fontWeight: 'bold' }
                  }}
                />
              )}

              {TEMP_RANGES.critical.min >= yDomain[0] && TEMP_RANGES.critical.min <= yDomain[1] && (
                <ReferenceLine
                  y={TEMP_RANGES.critical.min}
                  stroke="#ef4444"
                  strokeDasharray="8 4"
                  strokeWidth={2}
                  label={{
                    value: "🔥 Critical (105°C)",
                    position: "topLeft",
                    style: { fontSize: '12px', fontWeight: 'bold' }
                  }}
                />
              )}

              {TEMP_RANGES.emergency.min >= yDomain[0] && TEMP_RANGES.emergency.min <= yDomain[1] && (
                <ReferenceLine
                  y={TEMP_RANGES.emergency.min}
                  stroke="#dc2626"
                  strokeDasharray="4 2"
                  strokeWidth={3}
                  label={{
                    value: "🚨 EMERGENCY (120°C)",
                    position: "topLeft",
                    style: { fontSize: '12px', fontWeight: 'bold', color: '#dc2626' }
                  }}
                />
              )}
            </>
          )}

          {/* Actual data line */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#2563eb"
            strokeWidth={3}
            dot={{ fill: '#2563eb', strokeWidth: 2, r: 5 }}
            activeDot={{ r: 7, stroke: '#2563eb', strokeWidth: 2, fill: '#ffffff' }}
            connectNulls={false}
            name="📊 Historical Data"
          />

          {/* Prediction line */}
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#ea580c"
            strokeWidth={3}
            strokeDasharray="8 4"
            dot={{ fill: '#ea580c', strokeWidth: 2, r: 5 }}
            activeDot={{ r: 7, stroke: '#ea580c', strokeWidth: 2, fill: '#ffffff' }}
            connectNulls={false}
            name="🔮 Forecast"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesChart;
