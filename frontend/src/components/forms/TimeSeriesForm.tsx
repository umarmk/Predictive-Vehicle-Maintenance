import React, { useState } from 'react';
import Button from '../ui/Button';
import Input from '../ui/Input';
import type { TimeSeriesInput } from '../../types';

interface TimeSeriesFormProps {
  onSubmit: (data: TimeSeriesInput) => void;
  loading?: boolean;
  initialData?: TimeSeriesInput;
}

const TimeSeriesForm: React.FC<TimeSeriesFormProps> = ({
  onSubmit,
  loading = false,
  initialData
}) => {
  const [formData, setFormData] = useState<TimeSeriesInput>({
    series: initialData?.series || [],
    seq_length: initialData?.seq_length || 10,
    horizon: initialData?.horizon || 5,
    feature: initialData?.feature || 'Engine Temperature (°C)'
  });

  const [seriesInput, setSeriesInput] = useState<string>(
    initialData?.series?.join(', ') || ''
  );

  // Update form when initialData changes (e.g., when a pattern is selected)
  React.useEffect(() => {
    if (initialData) {
      setFormData({
        series: initialData.series || [],
        seq_length: initialData.seq_length || 10,
        horizon: initialData.horizon || 5,
        feature: initialData.feature || 'Engine Temperature (°C)'
      });
      setSeriesInput(initialData.series?.join(', ') || '');
    }
  }, [initialData]);

  const handleSeriesChange = (value: string) => {
    setSeriesInput(value);

    // Parse the comma-separated values
    const numbers = value
      .split(',')
      .map(s => s.trim())
      .filter(s => s !== '')
      .map(s => parseFloat(s))
      .filter(n => !isNaN(n));

    setFormData(prev => ({
      ...prev,
      series: numbers
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.series.length === 0) {
      alert('Please enter at least one temperature value');
      return;
    }

    if (formData.series.length < formData.seq_length) {
      alert(`Series length (${formData.series.length}) must be >= sequence length (${formData.seq_length})`);
      return;
    }

    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Temperature Series (comma-separated values in °C)
        </label>
        <textarea
          value={seriesInput}
          onChange={(e) => handleSeriesChange(e.target.value)}
          placeholder="85, 87, 89, 91, 93, 95, 97, 99, 101, 103"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          rows={3}
          required
        />
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Enter temperature values separated by commas. Example: 85, 87, 89, 91, 93
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input
          label="Sequence Length"
          type="number"
          value={formData.seq_length}
          onChange={(e) => setFormData(prev => ({
            ...prev,
            seq_length: parseInt(e.target.value) || 10
          }))}
          min={1}
          max={50}
          helperText="Number of past values to use for prediction"
        />

        <Input
          label="Forecast Horizon"
          type="number"
          value={formData.horizon}
          onChange={(e) => setFormData(prev => ({
            ...prev,
            horizon: parseInt(e.target.value) || 5
          }))}
          min={1}
          max={20}
          helperText="Number of future values to predict"
        />
      </div>

      <Input
        label="Feature Name"
        type="text"
        value={formData.feature}
        onChange={(e) => setFormData(prev => ({
          ...prev,
          feature: e.target.value
        }))}
        placeholder="Engine Temperature (°C)"
        helperText="Name of the feature being predicted"
      />

      <div className="flex justify-between items-center">
        <div className="text-sm text-gray-600 dark:text-gray-400">
          Series length: {formData.series.length} values
        </div>

        <Button
          type="submit"
          variant="primary"
          isLoading={loading}
          disabled={formData.series.length === 0 || formData.series.length < formData.seq_length}
        >
          Generate Forecast
        </Button>
      </div>
    </form>
  );
};

export default TimeSeriesForm;
