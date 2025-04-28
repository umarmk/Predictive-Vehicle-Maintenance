import React, { useState } from 'react'
import { predictVehicle } from './api'

const fields = [
  { name: 'Engine_Temperature_(°C)', label: 'Engine Temperature (°C)' },
  { name: 'Brake_Pad_Thickness_(mm)', label: 'Brake Pad Thickness (mm)' },
  { name: 'Tire_Pressure_(PSI)', label: 'Tire Pressure (PSI)' },
  { name: 'Anomaly_Indication', label: 'Anomaly Indication' },
  { name: 'is_engine_failure', label: 'Engine Failure Flag' },
  { name: 'is_brake_failure', label: 'Brake Failure Flag' },
  { name: 'is_battery_failure', label: 'Battery Failure Flag' },
  { name: 'is_low_tire_pressure', label: 'Low Tire Pressure Flag' },
  { name: 'is_maintenance_required', label: 'Maintenance Required Flag' },
]

function App() {
  const initial = {}
  fields.forEach(f => { initial[f.name] = '' })
  const [formData, setFormData] = useState(initial)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = e => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = async e => {
    e.preventDefault()
    setError(null)
    setPrediction(null)
    try {
      const payload = {}
      fields.forEach(f => {
        payload[f.name] = parseFloat(formData[f.name])
      })
      const res = await predictVehicle(payload)
      setPrediction(res.predictions[0])
    } catch (err) {
      setError(err.response?.data?.error || err.message)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-lg mx-auto bg-white p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-4">Vehicle Maintenance Predictor</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {fields.map(f => (
            <div key={f.name}>
              <label className="block font-medium mb-1">{f.label}</label>
              <input
                type="number"
                step="any"
                name={f.name}
                value={formData[f.name]}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded p-2"
                required
              />
            </div>
          ))}
          <button
            type="submit"
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Predict
          </button>
        </form>
        {prediction !== null && (
          <div className="mt-4 text-green-700">Prediction: <span className="font-bold">{prediction}</span></div>
        )}
        {error && (
          <div className="mt-4 text-red-600">Error: {error}</div>
        )}
      </div>
    </div>
  )
}

export default App
