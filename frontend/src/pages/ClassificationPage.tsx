import React, { useState } from 'react';
import ClassificationForm from '../components/forms/ClassificationForm';
import Card from '../components/ui/Card';
import type { ClassificationInput, ClassificationResult } from '../types';
import api from '../api/client';

const ClassificationPage: React.FC = () => {
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<ClassificationResult | null>(null);
  const [inputData, setInputData] = useState<ClassificationInput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async (data: ClassificationInput) => {
    try {
      setPredictionLoading(true);
      setError(null);

      console.log('Sending prediction request with data:', data);
      const result = await api.predict(data);
      console.log('Received prediction result:', result);

      // Check if the result has the expected structure
      if (!result) {
        console.error('Invalid prediction result - result is null or undefined:', result);
        setError('Received invalid prediction result from server. Check console for details.');
        return;
      }

      // Ensure predictions array exists
      if (!result.predictions) {
        console.log('Creating predictions array from prediction value');
        result.predictions = result.prediction !== undefined ? [result.prediction] : [];
      }

      // Ensure predictions is an array
      if (!Array.isArray(result.predictions)) {
        console.log('Converting predictions to array');
        result.predictions = [result.predictions];
      }

      console.log('Final processed result:', result);
      setPredictionResult(result);
      setInputData(data);
    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.error || 'Failed to make prediction. Please try again.');
    } finally {
      setPredictionLoading(false);
    }
  };

  // Map prediction code to human-readable failure type
  const getFailureType = (code: number | string | undefined): string => {
    if (code === undefined) return 'Unknown';

    // Convert string to number if needed
    const numericCode = typeof code === 'string' ? parseInt(code, 10) : code;

    // If conversion failed, return the original code
    if (isNaN(numericCode)) return `${code}`;

    // Updated mapping based on the expected prediction codes
    const failureTypes: Record<number, string> = {
      0: 'No Failure',
      1: 'Engine Failure',
      2: 'Brake System Issues',
      3: 'Battery Problems',
      4: 'Tire Pressure Warning',
      5: 'General Maintenance Required'
    };

    return failureTypes[numericCode] || `Unknown (${code})`;
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ClassificationForm onSubmit={handlePrediction} isLoading={predictionLoading} />

        {error && (
          <Card title="Error" className="bg-danger-50 dark:bg-danger-900 border border-danger-200 dark:border-danger-800">
            <div className="text-danger-700 dark:text-danger-300">
              {error}
            </div>
          </Card>
        )}

        {predictionResult && !error && (
          <Card title="Prediction Result">
            <div className="space-y-4">
              <div className="flex items-center justify-center">
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary-600 dark:text-primary-400">
                    {getFailureType(predictionResult.prediction)}
                  </div>
                  <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                    Prediction Code: {predictionResult.prediction}
                  </div>
                  {predictionResult.probabilities && (
                    <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                      Confidence: {Math.max(...predictionResult.probabilities).toFixed(2) * 100}%
                    </div>
                  )}
                </div>
              </div>

              <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Recommended Action</h4>
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                  {getFailureType(predictionResult.prediction).includes('No Failure') ? (
                    <p className="text-success-700 dark:text-success-300">
                      No maintenance action required at this time. Vehicle is operating normally.
                    </p>
                  ) : (
                    <p className="text-warning-700 dark:text-warning-300">
                      Schedule maintenance to address the {getFailureType(predictionResult.prediction)} issue.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Explainability section removed */}
    </div>
  );
};

export default ClassificationPage;
