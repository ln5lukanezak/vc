import { useEffect, useRef, useState, useCallback } from 'react'
import { SVM } from './svm'
import {
  getDataConfig,
  getTrainConfig,
  getData,
  setDataConfig,
  regenerate,
  subscribe,
  type DatasetType,
} from './dataStore'
import {
  Slider,
  Select,
  ButtonRow,
  Stat,
  SectionHeading,
} from '../../components/Controls'
import type { Point2D } from '../../lib/datagen'
import type { KernelType } from './svm'

// ─── Constants ────────────────────────────────────────────────────────────────

const GRID_SIZE = 60           // decision-surface resolution
const PASSES_PER_FRAME = 2     // SMO passes per RAF tick
const MAX_AUTO_PASSES = 200    // stop animation after this many passes

// Class colours: indigo (class 0) / emerald (class 1)
const CLASS_RGB: [[number,number,number],[number,number,number]] = [
  [99, 102, 241],    // indigo-500
  [16, 185, 129],    // emerald-500
]
const CLASS_COLORS_RGBA = ['rgba(129,140,248,0.92)', 'rgba(52,211,153,0.92)']

const DATASET_OPTIONS: Array<{ value: DatasetType; label: string }> = [
  { value: 'blobs-sep',  label: 'Separable blobs' },
  { value: 'blobs-over', label: 'Overlapping blobs' },
  { value: 'circles',   label: 'Concentric circles' },
  { value: 'moons',     label: 'Two moons' },
]

const KERNEL_OPTIONS: Array<{ value: KernelType; label: string }> = [
  { value: 'linear', label: 'Linear' },
  { value: 'poly',   label: 'Polynomial' },
  { value: 'rbf',    label: 'RBF (Gaussian)' },
]

// ─── Grid helpers ─────────────────────────────────────────────────────────────

function makeGridCoords(lo: number, hi: number, size: number): number[] {
  return Array.from({ length: size }, (_, i) => lo + (i / (size - 1)) * (hi - lo))
}

function getDataBounds(data: Point2D[]) {
  if (data.length === 0) return { xMin: -1.5, xMax: 1.5, yMin: -1.5, yMax: 1.5 }
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
  for (const p of data) {
    if (p.x < xMin) xMin = p.x
    if (p.x > xMax) xMax = p.x
    if (p.y < yMin) yMin = p.y
    if (p.y > yMax) yMax = p.y
  }
  const xPad = (xMax - xMin) * 0.15 || 0.4
  const yPad = (yMax - yMin) * 0.15 || 0.4
  return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad }
}

// ─── Canvas renderer ──────────────────────────────────────────────────────────

/**
 * Draw decision-value surface + margin bands + data points + support vectors.
 *
 * Colormap:
 *   f < −1 → indigo (class 0, outside margin)
 *   f ∈ (−1, 0) → indigo tinted lighter (class 0 side, inside margin)
 *   f ∈ (0, +1) → emerald tinted lighter (class 1 side, inside margin)
 *   f > +1 → emerald (class 1, outside margin)
 *
 * Near f = 0 (decision boundary) and f = ±1 (margin contours) a faint light
 * strip is rendered for visual contrast.
 */
function drawDecisionSurface(
  canvas: HTMLCanvasElement,
  grid: number[][],
  data: Point2D[],
  svIndices: Set<number>,
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const rows = grid.length
  const cols = rows > 0 ? grid[0].length : 0
  if (rows === 0 || cols === 0) return

  const cellW = w / cols
  const cellH = h / rows

  // Background
  ctx.fillStyle = '#0f172a'
  ctx.fillRect(0, 0, w, h)

  const bg: [number,number,number] = [15, 23, 42]   // slate-950

  // ── Decision surface ──────────────────────────────────────────────────

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const f = grid[r][c]
      const fC = Math.max(-3, Math.min(3, f))

      let R: number, G: number, B: number

      // Determine class side and margin factor
      const classIdx = fC >= 0 ? 1 : 0
      const [cr, cg, cb] = CLASS_RGB[classIdx]
      const absF = Math.abs(fC)
      // blend factor: low near 0, full at ±3
      const tFull = Math.min(1, absF / 3) * 0.72 + 0.05
      // inside margin → reduce intensity
      const t = absF < 1 ? tFull * 0.45 : tFull

      R = Math.round(bg[0] + (cr - bg[0]) * t)
      G = Math.round(bg[1] + (cg - bg[1]) * t)
      B = Math.round(bg[2] + (cb - bg[2]) * t)

      ctx.fillStyle = `rgb(${R},${G},${B})`
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
    }
  }

  // ── Contour overlays at f = 0 (boundary) and f = ±1 (margin bands) ───

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const f = grid[r][c]
      // margin contours at ±1
      const dist1 = Math.abs(Math.abs(f) - 1)
      if (dist1 < 0.18) {
        const a = ((0.18 - dist1) / 0.18) * 0.28
        ctx.fillStyle = `rgba(226,232,240,${a.toFixed(3)})`
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
      }
      // decision boundary at f = 0
      if (Math.abs(f) < 0.07) {
        ctx.fillStyle = 'rgba(226,232,240,0.6)'
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
      }
    }
  }

  // ── Data points ────────────────────────────────────────────────────────

  const { xMin, xMax, yMin, yMax } = bounds
  const toX = (x: number) => ((x - xMin) / (xMax - xMin)) * w
  const toY = (y: number) => h - ((y - yMin) / (yMax - yMin)) * h

  for (let idx = 0; idx < data.length; idx++) {
    const pt = data[idx]
    const cx = toX(pt.x)
    const cy = toY(pt.y)
    if (cx < -8 || cx > w + 8 || cy < -8 || cy > h + 8) continue

    const isSV = svIndices.has(idx)
    const r = isSV ? 6 : 4

    // Outer ring for support vectors
    if (isSV) {
      ctx.beginPath()
      ctx.arc(cx, cy, r + 4.5, 0, Math.PI * 2)
      ctx.strokeStyle = CLASS_COLORS_RGBA[pt.label]
      ctx.lineWidth = 2
      ctx.stroke()
    }

    ctx.beginPath()
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.fillStyle = CLASS_COLORS_RGBA[pt.label]
    ctx.fill()
    ctx.strokeStyle = 'rgba(15,23,42,0.85)'
    ctx.lineWidth = 1
    ctx.stroke()
  }
}

// ─── Main Visualization component ─────────────────────────────────────────────

export function Visualization() {
  const initData = getDataConfig()
  const initTrain = getTrainConfig()

  // UI state
  const [dataset, setDataset] = useState<DatasetType>(initData.dataset)
  const [noise, setNoise] = useState(initData.noise)
  const [n, setN] = useState(initData.n)
  const [kernel, setKernel] = useState<KernelType>(initTrain.kernel)
  const [C, setC] = useState(initTrain.C)
  const [gamma, setGamma] = useState(initTrain.gamma)
  const [degree, setDegree] = useState(initTrain.degree)
  const [isTraining, setIsTraining] = useState(false)

  // Readouts
  const [passes, setPasses] = useState(0)
  const [alphaChanges, setAlphaChanges] = useState(0)
  const [svCount, setSvCount] = useState(0)
  const [converged, setConverged] = useState(false)

  // Refs (avoid re-render overhead)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const isTrainingRef = useRef(false)
  const modelRef = useRef<SVM | null>(null)
  const dataRef = useRef<Point2D[]>(getData())
  const boundsRef = useRef(getDataBounds(getData()))
  const svIndicesRef = useRef<Set<number>>(new Set())

  // ── Build model ────────────────────────────────────────────────────────

  const buildModel = useCallback((
    kType: KernelType,
    cVal: number,
    gVal: number,
    deg: number,
  ) => {
    const data = dataRef.current
    const xs: Array<[number, number]> = data.map(p => [p.x, p.y])
    const labels = data.map(p => p.label)

    const svm = new SVM({ kernel: kType, C: cVal, gamma: gVal, degree: deg })
    const hasClass0 = labels.some(l => l === 0)
    const hasClass1 = labels.some(l => l === 1)
    if (hasClass0 && hasClass1 && xs.length >= 4) {
      svm.setData(xs, labels)
    }
    modelRef.current = svm
    svIndicesRef.current = new Set()
    setPasses(0)
    setAlphaChanges(0)
    setSvCount(0)
    setConverged(false)
  }, [])

  // ── Draw frame ─────────────────────────────────────────────────────────

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current
    const model = modelRef.current
    if (!canvas || !model || model.getN() === 0) return

    const bounds = boundsRef.current
    const xCoords = makeGridCoords(bounds.xMin, bounds.xMax, GRID_SIZE)
    const yCoords = makeGridCoords(bounds.yMin, bounds.yMax, GRID_SIZE)
    const grid = model.decisionGrid(xCoords, yCoords)
    drawDecisionSurface(canvas, grid, dataRef.current, svIndicesRef.current, bounds)
  }, [])

  // ── Train loop ─────────────────────────────────────────────────────────

  const trainLoop = useCallback(() => {
    if (!isTrainingRef.current) return
    const model = modelRef.current
    if (!model || model.getN() === 0) {
      isTrainingRef.current = false
      setIsTraining(false)
      return
    }

    model.runPasses(PASSES_PER_FRAME)
    const state = model.getState()
    svIndicesRef.current = new Set(model.supportVectorIndices())

    setPasses(state.passes)
    setAlphaChanges(state.alphaChanges)
    setSvCount(state.svCount)
    setConverged(state.converged)

    drawFrame()

    if (state.converged || state.passes >= MAX_AUTO_PASSES) {
      isTrainingRef.current = false
      setIsTraining(false)
      return
    }

    if (isTrainingRef.current) {
      rafRef.current = requestAnimationFrame(trainLoop)
    }
  }, [drawFrame])

  // ── Stop helper ────────────────────────────────────────────────────────

  const stopTraining = useCallback(() => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
  }, [])

  // ── Rebuild + redraw ───────────────────────────────────────────────────

  const rebuildAndDraw = useCallback((
    kType: KernelType,
    cVal: number,
    gVal: number,
    deg: number,
  ) => {
    stopTraining()
    buildModel(kType, cVal, gVal, deg)
    drawFrame()
  }, [stopTraining, buildModel, drawFrame])

  // ── Mount ──────────────────────────────────────────────────────────────

  useEffect(() => {
    dataRef.current = getData()
    boundsRef.current = getDataBounds(getData())
    buildModel(kernel, C, gamma, degree)
    drawFrame()

    const unsub = subscribe(() => {
      dataRef.current = getData()
      boundsRef.current = getDataBounds(getData())
      stopTraining()
      buildModel(kernel, C, gamma, degree)
      drawFrame()
    })

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      unsub()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Button handlers ────────────────────────────────────────────────────

  const handleTrain = () => {
    isTrainingRef.current = true
    setIsTraining(true)
    rafRef.current = requestAnimationFrame(trainLoop)
  }

  const handlePause = () => stopTraining()

  const handleStep = () => {
    stopTraining()
    const model = modelRef.current
    if (!model || model.getN() === 0) return
    model.runPasses(1)
    const state = model.getState()
    svIndicesRef.current = new Set(model.supportVectorIndices())
    setPasses(state.passes)
    setAlphaChanges(state.alphaChanges)
    setSvCount(state.svCount)
    setConverged(state.converged)
    drawFrame()
  }

  const handleReset = () => {
    stopTraining()
    buildModel(kernel, C, gamma, degree)
    drawFrame()
  }

  const handleRegenerate = () => {
    stopTraining()
    setDataConfig({ dataset, noise, n, seed: Math.floor(Math.random() * 100000) })
    regenerate()
    dataRef.current = getData()
    boundsRef.current = getDataBounds(getData())
    buildModel(kernel, C, gamma, degree)
    drawFrame()
  }

  // ── Dataset change helper ──────────────────────────────────────────────

  const handleDatasetChange = (d: DatasetType) => {
    setDataset(d)
    setDataConfig({ dataset: d })
    regenerate()
    dataRef.current = getData()
    boundsRef.current = getDataBounds(getData())
    rebuildAndDraw(kernel, C, gamma, degree)
  }

  const handleNoiseChange = (v: number) => {
    setNoise(v)
    setDataConfig({ noise: v })
    regenerate()
    dataRef.current = getData()
    boundsRef.current = getDataBounds(getData())
    rebuildAndDraw(kernel, C, gamma, degree)
  }

  const handleNChange = (v: number) => {
    setN(v)
    setDataConfig({ n: v })
    regenerate()
    dataRef.current = getData()
    boundsRef.current = getDataBounds(getData())
    rebuildAndDraw(kernel, C, gamma, degree)
  }

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 h-full">

      {/* Controls row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 shrink-0">

        <Select
          label="Dataset"
          value={dataset}
          options={DATASET_OPTIONS}
          onChange={v => handleDatasetChange(v as DatasetType)}
          disabled={isTraining}
        />

        <Select
          label="Kernel"
          value={kernel}
          options={KERNEL_OPTIONS}
          onChange={v => {
            const k = v as KernelType
            setKernel(k)
            rebuildAndDraw(k, C, gamma, degree)
          }}
          disabled={isTraining}
        />

        <Slider
          label="C (soft margin)"
          value={C}
          min={0.05}
          max={20}
          step={0.05}
          onChange={v => {
            setC(v)
            rebuildAndDraw(kernel, v, gamma, degree)
          }}
          format={v => v.toFixed(2)}
          disabled={isTraining}
        />

        {kernel === 'rbf' && (
          <Slider
            label="γ (RBF)"
            value={gamma}
            min={0.05}
            max={5}
            step={0.05}
            onChange={v => {
              setGamma(v)
              rebuildAndDraw(kernel, C, v, degree)
            }}
            format={v => v.toFixed(2)}
            disabled={isTraining}
          />
        )}

        {kernel === 'poly' && (
          <Slider
            label="Degree (d)"
            value={degree}
            min={2}
            max={7}
            step={1}
            onChange={v => {
              setDegree(v)
              rebuildAndDraw(kernel, C, gamma, v)
            }}
            disabled={isTraining}
          />
        )}

        <Slider
          label="Noise (σ)"
          value={noise}
          min={0.02}
          max={0.4}
          step={0.02}
          onChange={handleNoiseChange}
          format={v => v.toFixed(2)}
          disabled={isTraining}
        />

        <Slider
          label="Samples (n)"
          value={n}
          min={20}
          max={120}
          step={10}
          onChange={handleNChange}
          disabled={isTraining}
        />

        {/* Action buttons */}
        <div className="flex flex-col justify-end gap-2">
          <ButtonRow
            buttons={[
              isTraining
                ? { label: 'Pause',  onClick: handlePause,  variant: 'danger' }
                : { label: 'Train',  onClick: handleTrain,  variant: 'primary' },
              { label: 'Step',   onClick: handleStep,   disabled: isTraining },
              { label: 'Reset',  onClick: handleReset,  disabled: isTraining },
            ]}
          />
          <ButtonRow
            buttons={[
              { label: 'Regen data', onClick: handleRegenerate, disabled: isTraining, variant: 'secondary' },
            ]}
          />
        </div>
      </div>

      {/* Stats row */}
      <div className="flex flex-wrap gap-3 shrink-0">
        <Stat label="Passes"    value={passes}         color="text-slate-300" />
        <Stat label="α changed" value={alphaChanges}   color={alphaChanges === 0 && passes > 0 ? 'text-emerald-300' : 'text-amber-300'} />
        <Stat label="SVs"       value={svCount}        color="text-indigo-300" />
        <Stat
          label="Status"
          value={converged ? 'Converged' : passes === 0 ? 'Untrained' : 'Training…'}
          color={converged ? 'text-emerald-300' : 'text-amber-300'}
        />
      </div>

      {/* Main canvas */}
      <div className="flex flex-1 gap-4 min-h-0">
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <SectionHeading>Decision Surface · boundary f=0 · margins f=±1</SectionHeading>
          <div
            className="flex-1 min-h-0 rounded-md overflow-hidden border border-slate-700/60 bg-slate-950"
            style={{ minHeight: 260 }}
          >
            <canvas ref={canvasRef} className="w-full h-full block" />
          </div>
        </div>

        {/* Legend + info */}
        <div className="flex flex-col gap-3 w-48 shrink-0">
          <SectionHeading>Legend</SectionHeading>
          <div className="space-y-2 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ background: 'rgb(99,102,241)' }} />
              Class 0 (f &lt; 0)
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full flex-shrink-0" style={{ background: 'rgb(16,185,129)' }} />
              Class 1 (f &gt; 0)
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="inline-block w-8 h-px flex-shrink-0 bg-slate-200" />
              Boundary (f=0)
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-8 h-px flex-shrink-0 bg-slate-400 opacity-50" />
              Margins (f=±1)
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span
                className="flex-shrink-0 w-4 h-4 rounded-full border-2"
                style={{ borderColor: 'rgba(129,140,248,0.9)' }}
              />
              Support vector (ringed)
            </div>
          </div>

          <div className="mt-2 text-xs text-slate-500 space-y-1.5">
            <p>
              The lighter band between f = −1 and f = +1 is the{' '}
              <em>margin zone</em>. Support vectors sit on or inside it.
            </p>
            <p>
              Ringed points are support vectors — the only training points that
              define the boundary.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
