import { useEffect, useRef, useState, useCallback } from 'react'
import { Plot } from '../../lib/plot'
import { LossChart } from '../../lib/losschart'
import { createOptimizer, type OptimizerName } from '../../lib/optimizers'
import { mean, std } from '../../lib/mathutils'
import { getData } from './dataStore'
import {
  Slider,
  Select,
  ButtonRow,
  Stat,
  SectionHeading,
} from '../../components/Controls'

// ─── Polynomial feature builder ───────────────────────────────────────────────

function polyFeatures(x: number, degree: number): number[] {
  const f: number[] = [1]
  for (let d = 1; d <= degree; d++) f.push(Math.pow(x, d))
  return f
}

function buildDesignMatrix(xs: number[], degree: number): number[][] {
  return xs.map((x) => polyFeatures(x, degree))
}

// ─── Compute MSE and R² ───────────────────────────────────────────────────────

function computeMetrics(
  weights: number[],
  Xs: number[][],
  ys: number[],
): { mse: number; r2: number } {
  const n = ys.length
  let ssRes = 0
  const yMean = mean(ys)
  let ssTot = 0
  for (let i = 0; i < n; i++) {
    let pred = 0
    for (let j = 0; j < weights.length; j++) pred += weights[j] * Xs[i][j]
    ssRes += (pred - ys[i]) ** 2
    ssTot += (ys[i] - yMean) ** 2
  }
  return { mse: ssRes / n, r2: 1 - ssRes / (ssTot + 1e-12) }
}

// ─── Gradient computation (full batch MSE) ────────────────────────────────────

function computeGradients(
  weights: number[],
  Xs: number[][],
  ys: number[],
): number[] {
  const n = ys.length
  const grads = new Array(weights.length).fill(0)
  for (let i = 0; i < n; i++) {
    let pred = 0
    for (let j = 0; j < weights.length; j++) pred += weights[j] * Xs[i][j]
    const err = pred - ys[i]
    for (let j = 0; j < weights.length; j++) {
      grads[j] += (2 / n) * err * Xs[i][j]
    }
  }
  return grads
}

// ─── Scaler ──────────────────────────────────────────────────────────────────

interface FitScaler {
  means: number[]
  stds: number[]
}

function fitScaler(X: number[][]): FitScaler {
  if (X.length === 0 || X[0].length === 0) return { means: [], stds: [] }
  const nF = X[0].length
  const means2: number[] = []
  const stds2: number[] = []
  for (let j = 0; j < nF; j++) {
    const col = X.map((r) => r[j])
    // Skip column 0 (bias, always 1)
    if (j === 0) { means2.push(0); stds2.push(1); continue }
    const m = mean(col)
    const s = std(col)
    means2.push(m)
    stds2.push(s < 1e-12 ? 1 : s)
  }
  return { means: means2, stds: stds2 }
}

function applyScaler(X: number[][], sc: FitScaler): number[][] {
  return X.map((row) =>
    row.map((v, j) => (v - sc.means[j]) / sc.stds[j]),
  )
}

// ─── Transform weights from standardized space to original space ──────────────

function unstandardizeWeights(wScaled: number[], sc: FitScaler): number[] {
  // In standardized space: ŷ = Σ wScaled[j] * (x^j - means[j]) / stds[j]
  // = Σ wScaled[j]/stds[j] * x^j  - Σ wScaled[j]*means[j]/stds[j]
  // So w_orig[j] = wScaled[j] / stds[j], and bias adjustment to w_orig[0]
  const w: number[] = new Array(wScaled.length).fill(0)
  let biasAdj = 0
  for (let j = 0; j < wScaled.length; j++) {
    w[j] = wScaled[j] / sc.stds[j]
    biasAdj -= wScaled[j] * sc.means[j] / sc.stds[j]
  }
  w[0] += biasAdj
  return w
}

// ─── Component ────────────────────────────────────────────────────────────────

const LR_OPTIONS = [
  { value: '0.001', label: '0.001' },
  { value: '0.003', label: '0.003' },
  { value: '0.01',  label: '0.01' },
  { value: '0.03',  label: '0.03' },
  { value: '0.1',   label: '0.1' },
  { value: '0.3',   label: '0.3' },
]

const OPT_OPTIONS: Array<{ value: OptimizerName; label: string }> = [
  { value: 'SGD',      label: 'SGD' },
  { value: 'Momentum', label: 'Momentum' },
  { value: 'Adam',     label: 'Adam' },
]

const STEPS_PER_FRAME = 5

export function Visualization() {
  // ─── State ──────────────────────────────────────────────────────────────
  const [modelDegree, setModelDegree] = useState(2)
  const [lrKey, setLrKey] = useState('0.03')
  const [optName, setOptName] = useState<OptimizerName>('Adam')
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [mse, setMse] = useState<number>(0)
  const [r2, setR2] = useState<number>(0)

  // ─── Refs ────────────────────────────────────────────────────────────────
  const plotCanvasRef = useRef<HTMLCanvasElement>(null)
  const lossCanvasRef = useRef<HTMLCanvasElement>(null)
  const plotRef = useRef<Plot | null>(null)
  const lossChartRef = useRef<LossChart | null>(null)
  const rafRef = useRef<number | null>(null)

  // Training state (mutable, not react state — avoids stale closure)
  const weightsRef = useRef<number[]>([])
  const scalerRef = useRef<FitScaler>({ means: [], stds: [] })
  const XsScaledRef = useRef<number[][]>([])
  const ysRef = useRef<number[]>([])
  const epochRef = useRef(0)
  const optimizerRef = useRef(createOptimizer(optName, parseFloat(lrKey)))
  const isTrainingRef = useRef(false)

  // ─── Initialize / reset training ─────────────────────────────────────────
  const initTraining = useCallback(() => {
    const { x, y } = getData()
    const X = buildDesignMatrix(x, modelDegree)
    const scaler = fitScaler(X)
    const Xsc = applyScaler(X, scaler)

    // Initialize weights to zero
    const w = new Array(modelDegree + 1).fill(0)

    scalerRef.current = scaler
    XsScaledRef.current = Xsc
    ysRef.current = y
    weightsRef.current = w
    epochRef.current = 0

    optimizerRef.current = createOptimizer(optName, parseFloat(lrKey))
    optimizerRef.current.reset()

    lossChartRef.current?.reset()

    const metrics = computeMetrics(w, Xsc, y)
    setEpoch(0)
    setMse(metrics.mse)
    setR2(metrics.r2)
  }, [modelDegree, optName, lrKey])

  // ─── Canvas setup ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (plotCanvasRef.current) {
      plotRef.current = new Plot(plotCanvasRef.current)
    }
    if (lossCanvasRef.current) {
      lossChartRef.current = new LossChart(lossCanvasRef.current, {
        color: '#818cf8',
        label: 'MSE',
      })
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  // ─── Draw ─────────────────────────────────────────────────────────────────
  const drawFrame = useCallback(() => {
    const plot = plotRef.current
    const lossChart = lossChartRef.current
    if (!plot) return

    plot.resize()
    const { x, y } = getData()

    // Compute fitted curve y-samples across the x-range (using unstandardized weights)
    const wOrig = unstandardizeWeights(weightsRef.current, scalerRef.current)
    const xMin = Math.min(...x)
    const xMax = Math.max(...x)
    const curveYs: number[] = []
    for (let i = 0; i <= 100; i++) {
      const xi = xMin + (i / 100) * (xMax - xMin)
      let val = 0
      for (let j = 0; j < wOrig.length; j++) val += wOrig[j] * Math.pow(xi, j)
      if (isFinite(val)) curveYs.push(val)
    }

    // Compute union y-range (data ∪ curve), apply padding, set bounds once
    const allY = curveYs.length > 0 ? [...y, ...curveYs] : [...y]
    const yMin = Math.min(...allY)
    const yMax = Math.max(...allY)
    const xPad = (xMax - xMin) * 0.15 || 1
    const yPad = (yMax - yMin) * 0.15 || 1
    plot.setBounds({
      xMin: xMin - xPad,
      xMax: xMax + xPad,
      yMin: yMin - yPad,
      yMax: yMax + yPad,
    })

    // Single draw pass
    plot.clear()
    plot.drawAxesGrid()
    plot.scatter(
      x.map((xi, i) => ({ x: xi, y: y[i] })),
      { color: '#818cf8', r: 4, alpha: 0.7 },
    )
    plot.line(
      (xi: number) => {
        let val = 0
        for (let j = 0; j < wOrig.length; j++) val += wOrig[j] * Math.pow(xi, j)
        return val
      },
      { color: '#06b6d4', width: 2.5 },
    )

    lossChart?.resize()
    lossChart?.draw()
  }, [])

  // Re-init when settings change
  useEffect(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      isTrainingRef.current = false
      setIsTraining(false)
    }
    initTraining()
    drawFrame()
  }, [modelDegree, optName, lrKey, initTraining, drawFrame])

  // ─── Training loop ────────────────────────────────────────────────────────
  const trainLoop = useCallback(() => {
    if (!isTrainingRef.current) return

    for (let s = 0; s < STEPS_PER_FRAME; s++) {
      const grads = computeGradients(
        weightsRef.current,
        XsScaledRef.current,
        ysRef.current,
      )
      weightsRef.current = optimizerRef.current.step(weightsRef.current, grads)
      epochRef.current++
    }

    const metrics = computeMetrics(
      weightsRef.current,
      XsScaledRef.current,
      ysRef.current,
    )
    lossChartRef.current?.push(metrics.mse)
    setEpoch(epochRef.current)
    setMse(metrics.mse)
    setR2(metrics.r2)

    drawFrame()

    if (isTrainingRef.current) {
      rafRef.current = requestAnimationFrame(trainLoop)
    }
  }, [drawFrame])

  // ─── Controls ────────────────────────────────────────────────────────────
  const handleTrain = () => {
    isTrainingRef.current = true
    setIsTraining(true)
    rafRef.current = requestAnimationFrame(trainLoop)
  }

  const handlePause = () => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
  }

  const handleStep = () => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)

    const grads = computeGradients(
      weightsRef.current,
      XsScaledRef.current,
      ysRef.current,
    )
    weightsRef.current = optimizerRef.current.step(weightsRef.current, grads)
    epochRef.current++

    const metrics = computeMetrics(
      weightsRef.current,
      XsScaledRef.current,
      ysRef.current,
    )
    lossChartRef.current?.push(metrics.mse)
    setEpoch(epochRef.current)
    setMse(metrics.mse)
    setR2(metrics.r2)
    drawFrame()
  }

  const handleReset = () => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    initTraining()
    drawFrame()
  }

  // ─── Coefficient bar chart ────────────────────────────────────────────────
  const wOrig = unstandardizeWeights(weightsRef.current, scalerRef.current)
  const maxAbsW = Math.max(...wOrig.map(Math.abs), 0.01)

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Controls row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 shrink-0">
        <Slider
          label="Model degree"
          value={modelDegree}
          min={1}
          max={9}
          step={1}
          onChange={setModelDegree}
          disabled={isTraining}
        />
        <Select
          label="Learning rate"
          value={lrKey}
          options={LR_OPTIONS}
          onChange={setLrKey}
          disabled={isTraining}
        />
        <Select
          label="Optimizer"
          value={optName}
          options={OPT_OPTIONS}
          onChange={(v) => setOptName(v as OptimizerName)}
          disabled={isTraining}
        />
        <div className="flex flex-col justify-end">
          <ButtonRow
            buttons={[
              isTraining
                ? { label: 'Pause', onClick: handlePause, variant: 'danger' }
                : { label: 'Train', onClick: handleTrain, variant: 'primary' },
              { label: 'Step', onClick: handleStep, disabled: isTraining },
              { label: 'Reset', onClick: handleReset, disabled: isTraining },
            ]}
          />
        </div>
      </div>

      {/* Metrics row */}
      <div className="flex flex-wrap gap-3 shrink-0">
        <Stat label="Epoch" value={epoch} color="text-slate-300" />
        <Stat label="MSE" value={mse} color="text-amber-300" />
        <Stat label="R²" value={r2} color="text-green-300" />
      </div>

      {/* Canvas + loss chart */}
      <div className="flex flex-1 gap-4 min-h-0">
        {/* Main plot */}
        <div className="flex-1 min-h-0 rounded-md overflow-hidden border border-slate-700/60 bg-slate-950">
          <canvas
            ref={plotCanvasRef}
            className="w-full h-full block"
            style={{ minHeight: 260 }}
          />
        </div>

        {/* Right column: loss + coefficients */}
        <div className="flex flex-col gap-3 w-52 shrink-0">
          <SectionHeading>Loss curve</SectionHeading>
          <div className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 h-36">
            <canvas
              ref={lossCanvasRef}
              className="w-full h-full block"
            />
          </div>

          <SectionHeading>Coefficients (orig. space)</SectionHeading>
          <div className="space-y-1.5 overflow-y-auto max-h-52">
            {wOrig.map((w, j) => (
              <div key={j} className="flex items-center gap-2">
                <span className="metric text-[10px] text-slate-500 w-5 text-right">
                  w{j}
                </span>
                <div className="flex-1 bg-slate-800 rounded-sm h-3 overflow-hidden">
                  <div
                    className="h-full rounded-sm transition-all duration-150"
                    style={{
                      width: `${(Math.abs(w) / maxAbsW) * 100}%`,
                      background: w >= 0 ? '#818cf8' : '#f87171',
                    }}
                  />
                </div>
                <span className="metric text-[10px] text-slate-400 w-14 text-right truncate">
                  {isFinite(w) ? w.toFixed(3) : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
