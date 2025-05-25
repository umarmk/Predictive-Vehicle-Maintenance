import React, { useEffect, useState } from 'react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import type { HistoryItem } from '../types';
import api from '../api/client';

const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'classification' | 'timeseries'>('all');
  const [limit, setLimit] = useState(20);

  // Additional filters
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [resultFilter, setResultFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const type = filter === 'all' ? undefined : filter;
      const response = await api.getHistory(limit, type as any);
      setHistory(response.history);
      setError(null);
    } catch (err: any) {
      console.error('Error fetching history:', err);
      setError(err.response?.data?.error || 'Failed to load prediction history. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Apply all filters to the history data
  const applyFilters = () => {
    let filtered = [...history];

    // Apply date range filter
    if (startDate) {
      const startDateTime = new Date(startDate).getTime();
      filtered = filtered.filter(item => new Date(item.timestamp).getTime() >= startDateTime);
    }

    if (endDate) {
      const endDateTime = new Date(endDate).getTime() + (24 * 60 * 60 * 1000); // Include the end date (add 1 day)
      filtered = filtered.filter(item => new Date(item.timestamp).getTime() <= endDateTime);
    }

    // Apply result filter for classification predictions
    if (filter === 'classification' && resultFilter !== 'all') {
      const resultCode = parseInt(resultFilter);
      filtered = filtered.filter(item =>
        item.type === 'classification' &&
        item.prediction &&
        item.prediction[0] === resultCode
      );
    }

    // Apply search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(item => {
        // Search in timestamp
        if (item.timestamp.toLowerCase().includes(term)) return true;

        // Search in result (for classification)
        if (item.type === 'classification' && item.prediction) {
          const failureType = getFailureType(item.prediction[0]);
          if (failureType.toLowerCase().includes(term)) return true;
        }

        return false;
      });
    }

    setFilteredHistory(filtered);
  };

  // Fetch history when filter or limit changes
  useEffect(() => {
    fetchHistory();
  }, [filter, limit]);

  // Apply filters when history or any filter changes
  useEffect(() => {
    applyFilters();
  }, [history, startDate, endDate, resultFilter, searchTerm]);

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

  return (
    <div className="space-y-6">
      <Card title="Prediction History">
        <div className="mb-6 space-y-4">
          {/* Basic filters row */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex items-center space-x-4">
              <label htmlFor="filter" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Type:
              </label>
              <select
                id="filter"
                className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={filter}
                onChange={(e) => setFilter(e.target.value as any)}
              >
                <option value="all">All Predictions</option>
                <option value="classification">Classification Only</option>
                <option value="timeseries">Time Series Only</option>
              </select>
            </div>

            <div className="flex items-center space-x-4">
              <label htmlFor="limit" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Show:
              </label>
              <select
                id="limit"
                className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value))}
              >
                <option value="10">10 items</option>
                <option value="20">20 items</option>
                <option value="50">50 items</option>
                <option value="100">100 items</option>
              </select>

              <Button
                variant="outline"
                onClick={fetchHistory}
                icon={
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                  </svg>
                }
              >
                Refresh
              </Button>
            </div>
          </div>

          {/* Advanced filters row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Date range filters */}
            <div className="flex flex-col space-y-2">
              <label htmlFor="startDate" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Start Date
              </label>
              <input
                type="date"
                id="startDate"
                className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>

            <div className="flex flex-col space-y-2">
              <label htmlFor="endDate" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                End Date
              </label>
              <input
                type="date"
                id="endDate"
                className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>

            {/* Result filter (only shown for classification) */}
            {filter === 'classification' && (
              <div className="flex flex-col space-y-2">
                <label htmlFor="resultFilter" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Result Type
                </label>
                <select
                  id="resultFilter"
                  className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                  value={resultFilter}
                  onChange={(e) => setResultFilter(e.target.value)}
                >
                  <option value="all">All Results</option>
                  <option value="0">No Failure</option>
                  <option value="1">Engine Failure</option>
                  <option value="2">Brake System Issues</option>
                  <option value="3">Battery Problems</option>
                  <option value="4">Tire Pressure Warning</option>
                  <option value="5">General Maintenance Required</option>
                </select>
              </div>
            )}

            {/* Search input */}
            <div className="flex flex-col space-y-2 md:col-span-3">
              <label htmlFor="search" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Search
              </label>
              <input
                type="text"
                id="search"
                placeholder="Search in predictions..."
                className="rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          {/* Export button */}
          <div className="flex justify-end">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                // Create CSV content
                const headers = ['Time', 'Type', 'Input', 'Result'];
                const rows = filteredHistory.map(item => [
                  formatTimestamp(item.timestamp),
                  item.type === 'classification' ? 'Classification' : 'Time Series',
                  item.type === 'classification' ? 'Vehicle Sensor Data' : `${item.series?.length} data points`,
                  item.type === 'classification'
                    ? getFailureType(item.prediction?.[0] || 0)
                    : `${item.predictions?.length} forecast values`
                ]);

                const csvContent = [
                  headers.join(','),
                  ...rows.map(row => row.join(','))
                ].join('\n');

                // Create download link
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.setAttribute('href', url);
                link.setAttribute('download', `prediction-history-${new Date().toISOString().split('T')[0]}.csv`);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }}
            >
              Export to CSV
            </Button>
          </div>
        </div>

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
        ) : history.length === 0 ? (
          <div className="py-8 text-center text-gray-600 dark:text-gray-400">
            No predictions found. Start by making a prediction.
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="py-8 text-center text-gray-600 dark:text-gray-400">
            No predictions match your filters. Try adjusting your search criteria.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Time
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Type
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Input
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Result
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {filteredHistory.map((item, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
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
                        ? 'Vehicle Sensor Data'
                        : `${item.series?.length} data points`}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {item.type === 'classification'
                        ? `${getFailureType(item.prediction?.[0] || 0)}`
                        : `${item.predictions?.length} forecast values`}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      <Button
                        variant="text"
                        size="sm"
                        onClick={() => setSelectedItem(item)}
                      >
                        Details
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Details Modal */}
      {selectedItem && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Prediction Details
                </h3>
                <button
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  onClick={() => setSelectedItem(null)}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">Time</h4>
                    <p className="text-gray-900 dark:text-white">{formatTimestamp(selectedItem.timestamp)}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">Type</h4>
                    <p className="text-gray-900 dark:text-white">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        selectedItem.type === 'classification'
                          ? 'bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200'
                          : 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
                      }`}>
                        {selectedItem.type === 'classification' ? 'Classification' : 'Time Series'}
                      </span>
                    </p>
                  </div>
                </div>

                {selectedItem.type === 'classification' && selectedItem.input && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Input Parameters</h4>
                    <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(selectedItem.input).map(([key, value]) => (
                          <div key={key}>
                            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400">{key.replace(/_/g, ' ')}</h5>
                            <p className="text-gray-900 dark:text-white">{value}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {selectedItem.type === 'timeseries' && selectedItem.series && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Input Series</h4>
                    <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                      <div className="max-h-40 overflow-y-auto">
                        <p className="text-gray-900 dark:text-white font-mono text-sm">
                          {selectedItem.series.join(', ')}
                        </p>
                      </div>
                      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                        {selectedItem.series.length} data points
                      </p>
                    </div>
                  </div>
                )}

                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Prediction Result</h4>
                  <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                    {selectedItem.type === 'classification' && selectedItem.prediction && (
                      <div>
                        <h5 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {getFailureType(selectedItem.prediction[0])}
                        </h5>
                        <p className="text-gray-500 dark:text-gray-400 mt-1">
                          {selectedItem.prediction[0] === 0
                            ? 'No issues detected. Vehicle is operating normally.'
                            : 'Potential issue detected. Maintenance recommended.'}
                        </p>
                      </div>
                    )}

                    {selectedItem.type === 'timeseries' && selectedItem.predictions && (
                      <div>
                        <h5 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Forecast Values</h5>
                        <div className="max-h-40 overflow-y-auto">
                          <p className="text-gray-900 dark:text-white font-mono text-sm">
                            {selectedItem.predictions.map(p => p.toFixed(4)).join(', ')}
                          </p>
                        </div>
                        <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                          {selectedItem.predictions.length} forecast points
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end">
                <Button
                  variant="outline"
                  onClick={() => setSelectedItem(null)}
                >
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoryPage;
