import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar,
} from 'recharts'
import LiveMonitor from './LiveMonitor'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

const DEFAULTS = {
  num_sensors: 4096,
  num_timesteps: 2048,
  window: 32,
  threshold: 3.0,
  seed: 42,
  sample_sensor_id: 0,
}

function buildSignalData(values, anomalyIndices) {
  const anomalySet = new Set(anomalyIndices)
  return values.map((v, t) => ({
    t,
    value: v,
    anomaly: anomalySet.has(t) ? v : null,
  }))
}

function App() {
  const [tab, setTab] = useState('live')
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm((f) => ({ ...f, [name]: Number(value) }))
  }

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch(`${API_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed: ${resp.status}`)
      }
      setResult(await resp.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const signalData = result
    ? buildSignalData(result.sample_sensor.values, result.sample_sensor.anomaly_indices)
    : []

  const timingData = result
    ? [
        { name: 'CPU (NumPy)', ms: result.cpu_ms },
        ...(result.gpu_ms != null ? [{ name: 'GPU (CUDA)', ms: result.gpu_ms }] : []),
      ]
    : []

  return (
    <div className="dashboard">
      <h1>GPU-Accelerated Sensor Anomaly Detector</h1>
      <p className="subtitle">
        Rolling z-score anomaly detection — live on this machine's real hardware, or as a synthetic CUDA vs NumPy benchmark.
      </p>

      <div className="tabs">
        <button className={tab === 'live' ? 'tab tab--active' : 'tab'} onClick={() => setTab('live')}>
          Live monitor (real data)
        </button>
        <button className={tab === 'benchmark' ? 'tab tab--active' : 'tab'} onClick={() => setTab('benchmark')}>
          Synthetic benchmark
        </button>
      </div>

      {tab === 'live' && <LiveMonitor />}

      {tab === 'benchmark' && (
        <>
      <section className="controls">
        <label>
          Sensors
          <input type="number" name="num_sensors" value={form.num_sensors} onChange={handleChange} min={1} />
        </label>
        <label>
          Timesteps
          <input type="number" name="num_timesteps" value={form.num_timesteps} onChange={handleChange} min={1} />
        </label>
        <label>
          Window
          <input type="number" name="window" value={form.window} onChange={handleChange} min={1} />
        </label>
        <label>
          Threshold (z)
          <input type="number" name="threshold" value={form.threshold} onChange={handleChange} step={0.1} />
        </label>
        <label>
          Sample sensor id
          <input type="number" name="sample_sensor_id" value={form.sample_sensor_id} onChange={handleChange} min={0} />
        </label>
        <button onClick={handleRun} disabled={loading}>
          {loading ? 'Running...' : 'Run detection'}
        </button>
      </section>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <section className="card">
            <h2>CPU vs GPU timing</h2>
            {!result.gpu_available && (
              <p className="note">GPU extension not available on this backend — showing CPU timing only.</p>
            )}
            {result.speedup_x != null && (
              <p className="speedup">Speedup: <strong>{result.speedup_x.toFixed(1)}x</strong></p>
            )}
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={timingData} layout="vertical">
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={100} />
                <Tooltip formatter={(v) => `${v.toFixed(2)} ms`} />
                <Bar dataKey="ms" fill="#4f8cff" />
              </BarChart>
            </ResponsiveContainer>
          </section>

          <section className="card">
            <h2>Sensor {result.sample_sensor.sensor_id} signal</h2>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={signalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#4f8cff" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="anomaly" stroke="#ff4f4f" dot={{ r: 3 }} strokeWidth={0} connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
            <p className="note">
              {result.sample_sensor.anomaly_indices.length} anomalies flagged on this sensor
              {result.sample_sensor.anomaly_indices.length > 0 &&
                ` at t = ${result.sample_sensor.anomaly_indices.slice(0, 10).join(', ')}${
                  result.sample_sensor.anomaly_indices.length > 10 ? ', ...' : ''
                }`}
            </p>
          </section>

          <section className="card">
            <h2>Flagged sensors ({result.flagged_sensors.length})</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Sensor ID</th>
                    <th>Anomaly count</th>
                  </tr>
                </thead>
                <tbody>
                  {result.flagged_sensors.slice(0, 200).map((s) => (
                    <tr key={s.sensor_id}>
                      <td>{s.sensor_id}</td>
                      <td>{s.anomaly_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
        </>
      )}
    </div>
  )
}

export default App
