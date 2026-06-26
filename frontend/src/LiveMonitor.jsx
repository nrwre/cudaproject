import { useEffect, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
const POLL_INTERVAL_MS = 2000

function average(series) {
  return series.reduce((sum, v) => sum + v, 0) / series.length
}

// Combine the 16 (or however many) per-core CPU series into one average-usage
// series, so the dashboard shows one readable chart instead of 16 tiny ones.
function deriveDisplayMetrics(rawMetrics) {
  const byName = Object.fromEntries(rawMetrics.map((m) => [m.name, m]))
  const coreMetrics = rawMetrics.filter((m) => m.name.startsWith('cpu_core_'))

  const display = []

  if (coreMetrics.length > 0) {
    const len = coreMetrics[0].values.length
    const avgValues = []
    const anomalyIndices = new Set()
    for (let t = 0; t < len; t++) {
      avgValues.push(average(coreMetrics.map((m) => m.values[t])))
    }
    coreMetrics.forEach((m) => m.anomaly_indices.forEach((i) => anomalyIndices.add(i)))
    display.push({
      name: 'cpu_avg_percent',
      label: `CPU usage (avg of ${coreMetrics.length} cores)`,
      unit: '%',
      values: avgValues,
      anomaly_indices: Array.from(anomalyIndices),
    })
  }

  const rest = [
    ['memory_percent', 'Memory usage', '%'],
    ['disk_read_mbps', 'Disk read', 'MB/s'],
    ['disk_write_mbps', 'Disk write', 'MB/s'],
    ['net_sent_mbps', 'Network sent', 'MB/s'],
    ['net_recv_mbps', 'Network received', 'MB/s'],
    ['gpu_temp_c', 'GPU temperature', '°C'],
    ['gpu_util_percent', 'GPU utilization', '%'],
  ]
  for (const [name, label, unit] of rest) {
    if (byName[name]) {
      display.push({ ...byName[name], label, unit })
    }
  }

  return display
}

function MetricChart({ metric }) {
  const anomalySet = new Set(metric.anomaly_indices)
  const data = metric.values.map((v, t) => ({
    t,
    value: v,
    anomaly: anomalySet.has(t) ? v : null,
  }))
  const latest = metric.values[metric.values.length - 1]
  const isAnomalous = anomalySet.has(metric.values.length - 1)

  return (
    <div className={`metric-card${isAnomalous ? ' metric-card--alert' : ''}`}>
      <div className="metric-card__header">
        <span>{metric.label}</span>
        <span className="metric-card__value">
          {latest != null ? latest.toFixed(1) : '—'} {metric.unit}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={80}>
        <LineChart data={data}>
          <XAxis dataKey="t" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <Tooltip formatter={(v) => `${v.toFixed(2)} ${metric.unit}`} labelFormatter={() => ''} />
          <Line type="monotone" dataKey="value" stroke="#4f8cff" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="anomaly" stroke="#ff4f4f" dot={{ r: 2 }} strokeWidth={0} connectNulls={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function LiveMonitor() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const resp = await fetch(`${API_BASE}/live`)
        if (!resp.ok) throw new Error(`Request failed: ${resp.status}`)
        setData(await resp.json())
        setError(null)
      } catch (e) {
        setError(e.message)
      }
    }
    poll()
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS)
    return () => clearInterval(timerRef.current)
  }, [])

  if (error) return <div className="error">Live monitor error: {error}</div>
  if (!data) return <p className="note">Connecting to live monitor...</p>
  if (data.note) return <p className="note">{data.note}</p>

  const displayMetrics = deriveDisplayMetrics(data.metrics)
  const anomalousNow = displayMetrics.filter((m) => m.anomaly_indices.includes(m.values.length - 1))

  return (
    <div>
      <p className="note">
        Real metrics from this machine, sampled every second. Rolling z-score (window={data.window}, threshold=
        {data.threshold}) applied live — not synthetic data.
      </p>
      {anomalousNow.length > 0 && (
        <p className="error">
          Anomalous right now: {anomalousNow.map((m) => m.label).join(', ')}
        </p>
      )}
      <div className="metric-grid">
        {displayMetrics.map((m) => (
          <MetricChart key={m.name} metric={m} />
        ))}
      </div>
    </div>
  )
}
