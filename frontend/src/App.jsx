import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, Target, Activity, Calendar } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

// Stat Card Component
const StatCard = ({ title, value, change, icon: Icon, color, prefix = '' }) => {
  const colorClasses = {
    green: 'border-green-500 bg-green-50',
    red: 'border-red-500 bg-red-50',
    blue: 'border-blue-500 bg-blue-50',
    yellow: 'border-yellow-500 bg-yellow-50',
  }
  
  return (
    <div className={`stat-card ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{prefix}{value}</p>
          {change && (
            <p className={`text-sm mt-1 ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {change >= 0 ? '↑' : '↓'} {Math.abs(change)}%
            </p>
          )}
        </div>
        <Icon className={`w-10 h-10 text-${color}-500 opacity-80`} />
      </div>
    </div>
  )
}

// Main App Component
function App() {
  const [summary, setSummary] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [monthly, setMonthly] = useState([])
  const [loading, setLoading] = useState(true)
  const [ticker, setTicker] = useState('SPY')

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      // Load CSV data from public folder or API
      const [summaryRes, predictionsRes, monthlyRes] = await Promise.all([
        fetch('/data/powerbi_summary.csv').then(r => r.text()),
        fetch('/data/powerbi_predictions.csv').then(r => r.text()),
        fetch('/data/powerbi_monthly.csv').then(r => r.text()),
      ])
      
      setSummary(parseCSV(summaryRes)[0])
      setPredictions(parseCSV(predictionsRes))
      setMonthly(parseCSV(monthlyRes))
    } catch (error) {
      console.error('Error loading data:', error)
      // Use sample data for demo
      loadSampleData()
    }
    setLoading(false)
  }

  const loadSampleData = () => {
    setSummary({
      Ticker: 'SPY',
      Total_Days: 2754,
      Total_Trades: 2712,
      Accuracy: 0.5614,
      Initial_Capital: 10000,
      Final_Strategy_Value: 65949.07,
      Strategy_Return_Pct: 559.49,
      Market_Return_Pct: 300.19,
      Outperformance_Pct: 259.30,
      Profitable: 'Yes',
      Beats_Market: 'Yes',
    })
    
    // Generate sample predictions
    const samplePredictions = []
    let strategyValue = 10000
    let marketValue = 10000
    for (let i = 0; i < 100; i++) {
      const date = new Date(2025, 0, 1)
      date.setDate(date.getDate() + i)
      const dailyReturn = (Math.random() - 0.48) * 0.02
      strategyValue *= (1 + dailyReturn * 0.8)
      marketValue *= (1 + dailyReturn)
      samplePredictions.push({
        Date: date.toISOString().split('T')[0],
        Strategy_Portfolio: strategyValue,
        Market_Portfolio: marketValue,
        Prediction: Math.random() > 0.4 ? 1 : 0,
        Correct: Math.random() > 0.44 ? 1 : 0,
      })
    }
    setPredictions(samplePredictions)
    
    // Generate sample monthly data
    const months = ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06']
    const sampleMonthly = months.map(m => ({
      Month: m,
      Strategy_Return_Pct: (Math.random() - 0.3) * 10,
      Market_Return_Pct: (Math.random() - 0.4) * 8,
      Monthly_Accuracy: 0.5 + Math.random() * 0.2,
    }))
    setMonthly(sampleMonthly)
  }

  const parseCSV = (csv) => {
    const lines = csv.trim().split('\n')
    const headers = lines[0].split(',')
    return lines.slice(1).map(line => {
      const values = line.split(',')
      const obj = {}
      headers.forEach((h, i) => {
        const val = values[i]
        obj[h] = isNaN(val) ? val : parseFloat(val)
      })
      return obj
    })
  }

  const COLORS = ['#10B981', '#EF4444']

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-900">FinSight Dashboard</h1>
          </div>
          <div className="flex items-center gap-4">
            <select 
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="border rounded-lg px-3 py-2"
            >
              <option value="SPY">SPY</option>
              <option value="AAPL">AAPL</option>
              <option value="MSFT">MSFT</option>
              <option value="GOOGL">GOOGL</option>
            </select>
            <button onClick={loadData} className="btn-primary">
              Refresh Data
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Strategy Return"
            value={`${summary?.Strategy_Return_Pct?.toFixed(1)}%`}
            color="green"
            icon={TrendingUp}
          />
          <StatCard
            title="Final Portfolio"
            value={summary?.Final_Strategy_Value?.toLocaleString()}
            prefix="$"
            color="blue"
            icon={DollarSign}
          />
          <StatCard
            title="Accuracy"
            value={`${(summary?.Accuracy * 100)?.toFixed(1)}%`}
            color="yellow"
            icon={Target}
          />
          <StatCard
            title="Outperformance"
            value={`${summary?.Outperformance_Pct?.toFixed(1)}%`}
            color={summary?.Outperformance_Pct > 0 ? 'green' : 'red'}
            icon={summary?.Outperformance_Pct > 0 ? TrendingUp : TrendingDown}
          />
        </div>

        {/* Portfolio Performance Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 card">
            <h2 className="text-lg font-semibold mb-4">Portfolio Performance</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={predictions.filter((_, i) => i % 10 === 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Date" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="Strategy_Portfolio" stroke="#3B82F6" name="Strategy" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Market_Portfolio" stroke="#9CA3AF" name="Market" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Profitability Indicator */}
          <div className="card flex flex-col items-center justify-center">
            <h2 className="text-lg font-semibold mb-4">Profitability Status</h2>
            <div className={`w-32 h-32 rounded-full flex items-center justify-center ${
              summary?.Profitable === 'Yes' ? 'bg-green-100' : 'bg-red-100'
            }`}>
              <div className={`w-24 h-24 rounded-full flex items-center justify-center text-white font-bold text-lg ${
                summary?.Profitable === 'Yes' ? 'bg-green-500' : 'bg-red-500'
              }`}>
                {summary?.Profitable === 'Yes' ? 'PROFIT' : 'LOSS'}
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-600">
              {summary?.Beats_Market === 'Yes' ? '✓ Beats Market' : '✗ Underperforms Market'}
            </p>
          </div>
        </div>

        {/* Monthly Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Monthly Returns</h2>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={monthly}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Strategy_Return_Pct" fill="#3B82F6" name="Strategy %" />
                <Bar dataKey="Market_Return_Pct" fill="#9CA3AF" name="Market %" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Monthly Accuracy</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={monthly}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Month" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v*100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => `${(v*100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="Monthly_Accuracy" stroke="#F59E0B" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Summary Table */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Performance Summary</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b">
                  <th className="py-3 px-4">Metric</th>
                  <th className="py-3 px-4 text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">Initial Capital</td>
                  <td className="py-3 px-4 text-right">${summary?.Initial_Capital?.toLocaleString()}</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">Final Strategy Value</td>
                  <td className="py-3 px-4 text-right font-semibold text-green-600">${summary?.Final_Strategy_Value?.toLocaleString()}</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">Total Trading Days</td>
                  <td className="py-3 px-4 text-right">{summary?.Total_Days}</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">Total Trades</td>
                  <td className="py-3 px-4 text-right">{summary?.Total_Trades}</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">Strategy Return</td>
                  <td className="py-3 px-4 text-right text-green-600">+{summary?.Strategy_Return_Pct}%</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">Market Return (Buy & Hold)</td>
                  <td className="py-3 px-4 text-right">+{summary?.Market_Return_Pct}%</td>
                </tr>
                <tr className="hover:bg-gray-50">
                  <td className="py-3 px-4">Outperformance vs Market</td>
                  <td className="py-3 px-4 text-right font-semibold text-blue-600">+{summary?.Outperformance_Pct}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-8 py-4">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-500 text-sm">
          FinSight Dashboard © 2026 | Last Updated: {new Date().toLocaleString()}
        </div>
      </footer>
    </div>
  )
}

export default App
