import React, { useState } from 'react';
import './App.css';
import { predictSingle, predictBatch, getHistory, explainPrediction } from './api';

function App() {
  const [input, setInput] = useState({});
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setInput({ ...input, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handlePredict = async () => {
    setLoading(true);
    setPrediction(null);
    setExplanation(null);
    if (file) {
      const result = await predictBatch(file);
      setPrediction(result);
    } else {
      const result = await predictSingle(input);
      setPrediction(result);
    }
    setLoading(false);
  };

  const handleExplain = async (method = 'shap') => {
    setLoading(true);
    if (input) {
      const result = await explainPrediction(input, method);
      setExplanation(result);
    }
    setLoading(false);
  };

  const fetchHistory = async () => {
    const hist = await getHistory();
    setHistory(hist);
  };

  return (
    <div className="App">
      <h1>Predictive Vehicle Maintenance Dashboard</h1>
      <div className="input-section">
        <h2>Upload Sensor Data (CSV for batch or fill below for single)</h2>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <div>
          <input name="sensor1" placeholder="Sensor 1" onChange={handleChange} />
          <input name="sensor2" placeholder="Sensor 2" onChange={handleChange} />
          <input name="mileage" placeholder="Mileage" onChange={handleChange} />
          <input name="temperature" placeholder="Temperature" onChange={handleChange} />
          {/* Add more fields as needed */}
        </div>
        <button onClick={handlePredict} disabled={loading}>
          Predict
        </button>
        <button onClick={() => handleExplain('shap')} disabled={loading}>
          Explain (SHAP)
        </button>
        <button onClick={() => handleExplain('lime')} disabled={loading}>
          Explain (LIME)
        </button>
        <button onClick={fetchHistory}>Show Prediction History</button>
      </div>
      <div className="output-section">
        {loading && <p>Loading...</p>}
        {prediction && (
          <div>
            <h3>Prediction Result</h3>
            <pre>{JSON.stringify(prediction, null, 2)}</pre>
          </div>
        )}
        {explanation && (
          <div>
            <h3>Explanation</h3>
            <pre>{JSON.stringify(explanation, null, 2)}</pre>
          </div>
        )}
        {history.length > 0 && (
          <div>
            <h3>Prediction History</h3>
            <pre style={{maxHeight: '200px', overflow: 'auto'}}>{JSON.stringify(history, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
