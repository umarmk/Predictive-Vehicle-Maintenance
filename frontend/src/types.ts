import { ReactNode } from 'react';

// Theme type
export type Theme = 'light' | 'dark';

// Navigation item type
export interface NavItem {
  name: string;
  path: string;
  icon: ReactNode;
}

// Vehicle data types
export interface VehicleData {
  id: string;
  timestamp: string;
  vehicleId: string;
  engineTemp: number;
  rpm: number;
  speed: number;
  fuelLevel: number;
  batteryVoltage: number;
  oilPressure: number;
  tirePressure: {
    frontLeft: number;
    frontRight: number;
    rearLeft: number;
    rearRight: number;
  };
  brakeFluid: number;
  coolantLevel: number;
  transmissionFluid: number;
  mileage: number;
  status: 'normal' | 'warning' | 'critical';
}

// Prediction result type
export interface PredictionResult {
  id: string;
  timestamp: string;
  vehicleId: string;
  component: string;
  probability: number;
  severity: 'low' | 'medium' | 'high';
  estimatedTimeToFailure: number; // in days
  recommendedAction: string;
}

// Time series data point
export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
}

// History item type
export interface HistoryItem {
  id: string;
  timestamp: string;
  vehicleId: string;
  type: 'classification' | 'timeseries';
  prediction?: string[];
  predictions?: number[];
  input?: Record<string, any>;
  series?: number[];
}

// Classification input and result types
export interface ClassificationInput {
  engineTemp: number;
  rpm: number;
  speed: number;
  fuelLevel: number;
  batteryVoltage: number;
  oilPressure: number;
  tirePressure: {
    frontLeft: number;
    frontRight: number;
    rearLeft: number;
    rearRight: number;
  };
  brakeFluid: number;
  coolantLevel: number;
  transmissionFluid: number;
  mileage: number;
  [key: string]: any;
}

export interface ClassificationResult {
  prediction?: number | string;
  predictions: number[] | string[];
  probabilities?: number[];
  success?: boolean;
  timestamp: string;
}

// Time series input and result types
export interface TimeSeriesInput {
  series: number[];
  seq_length?: number;
  horizon?: number;
  feature?: string;
  [key: string]: any;
}

export interface TimeSeriesResult {
  predictions?: number[] | number[][];
  prediction?: number | number[];
  success?: boolean;
  model?: string;
  request_id?: string;
  processing_time?: number;
  timestamp: string;
}

// Explainability types removed

// History response type
export interface HistoryResponse {
  history: HistoryItem[];
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
