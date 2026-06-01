import { useEffect, useRef, useState, useCallback } from 'react'
import { SoftmaxClassifier } from './logistic'
import {
  getDataConfig,
  setDataConfig,
  getTrainConfig,
  getData,
  regenerate,
  subscribe,
} from './dataStore'
import { createOptimizer, type OptimizerName } from '../../lib/optimizers'
import { LossChart } from '../../lib/losschart'
import {
  Slider,
  Select,
  ButtonRow,
  Stat,
  SectionHeading,
} from '../../components/Controls'
import type { Point2D } from '../../lib/datagen'

// ─── Constants ────────────────────────────────────────────────────────────────

const GRID_SIZE = 60          // probability surface resolution
const EPOCHS_PER_FRAME = 5    // epochs per RAF tick (logistic is cheap)
const CM_THROTTLE = 8         // update confusion matrix every N epochs

// Class colours (indigo / emerald / orange / pink)
const CLASS_COLORS_RGBA = [
  'rgba(129,140,248,0.9)',
  'rgba(52,211,153,0.9)',
  'rgba(251,146,60,0.9)',
  'rgba(244,114,182,0.9)',
]

// RGB triples for the probability surface fill
const CLASS_RGB: [number, number, number][] = [
  [99,  102, 241],   // indigo-500
  [16,  185, 129],   // emerald-500
  [249, 115,  22],   // orange-500
  [236,  72, 153],   // pink-500
]

const OPT_OPTIONS: Array<{ value: OptimizerName; label: string }> = [
  { value: 'SGD',      label: 'SGD' },
  { value: 'Momentum', label: 'Momentum' },
  { value: 'RMSProp',  label: 'RMSProp' },
  { value: 'Adam',     label: 'Adam' },
]

const LR_OPTIONS = ['0.001','0.005','0.01','0.05','0.1','0.3','1.0'].map(v => ({ value: v, label: v }))

// ─── Grid helpers ─────────────────────────────────────────────────────────────

function makeGridCoords(min: number, max: number, size: number): number[] {
  return Array.from({ length: size }, (_, i) => min + (i / (size - 1)) * (max - min))
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
  const xPad = (xMax - xMin) * 0.12 || 0.3
  const yPad = (yMax - yMin) * 0.12 || 0.3
  return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad }
}

// ─── Probability surface renderer ─────────────────────────────────────────────

/**
 * Draws the probability surface directly on a canvas:
 *  - K=2: diverging two-colour shading (class 0 → indigo, class 1 → emerald)
 *  - K>2: each cell coloured by the argmax class, blended toward a neutral dark
 *         background by (1 - max_prob) to show uncertainty
 */
function drawSurface(
  canvas: HTMLCanvasElement,
  prob0Grid: number[][],     // [row][col] = prob(class 0)
  classGrid: number[][],     // [row][col] = argmax class
  data: Point2D[],
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  numCls: number,
  model: SoftmaxClassifier,
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

  const rows = prob0Grid.length
  const cols = rows > 0 ? prob0Grid[0].length : 0
  if (rows === 0 || cols === 0) return

  const cellW = w / cols
  const cellH = h / rows

  // Draw grid cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let R: number, G: number, B: number

      if (numCls === 2) {
        // Diverging: class 0 → indigo, class 1 → emerald
        const p0 = prob0Grid[r][c]
        if (p0 > 0.5) {
          const t = (p0 - 0.5) * 2  // 0..1
          const bg: [number, number, number] = [15, 23, 42]
          R = Math.round(bg[0] + t * (CLASS_RGB[0][0] - bg[0]))
          G = Math.round(bg[1] + t * (CLASS_RGB[0][1] - bg[1]))
          B = Math.round(bg[2] + t * (CLASS_RGB[0][2] - bg[2]))
        } else {
          const t = (0.5 - p0) * 2
          const bg: [number, number, number] = [15, 23, 42]
          R = Math.round(bg[0] + t * (CLASS_RGB[1][0] - bg[0]))
          G = Math.round(bg[1] + t * (CLASS_RGB[1][1] - bg[1]))
          B = Math.round(bg[2] + t * (CLASS_RGB[1][2] - bg[2]))
        }
      } else {
        // Multi-class: colour by argmax, fade by confidence
        const k = classGrid[r][c]
        const xs = makeGridCoords(bounds.xMin, bounds.xMax, cols)
        const ys = makeGridCoords(bounds.yMin, bounds.yMax, rows)
        const probs = model.predict(xs[c], ys[r])
        const maxP = probs[k]
        const t = Math.max(0, (maxP - 1 / numCls) / (1 - 1 / numCls))
        const [cr, cg, cb] = CLASS_RGB[k] ?? CLASS_RGB[0]
        const bg = 15
        R = Math.round(bg + t * (cr - bg))
        G = Math.round(bg + t * (cg - bg))
        B = Math.round(bg + t * (cb - bg))
      }

      ctx.fillStyle = `rgb(${R},${G},${B})`
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
    }
  }

  // Overlay data points
  const { xMin, xMax, yMin, yMax } = bounds
  const toX = (x: number) => ((x - xMin) / (xMax - xMin)) * w
  const toY = (y: number) => h - ((y - yMin) / (yMax - yMin)) * h

  for (const pt of data) {
    const cx = toX(pt.x)
    const cy = toY(pt.y)
    if (cx < -6 || cx > w + 6 || cy < -6 || cy > h + 6) continue

    ctx.beginPath()
    ctx.arc(cx, cy, 4, 0, Math.PI * 2)
    ctx.fillStyle = CLASS_COLORS_RGBA[pt.label] ?? '#94a3b8'
    ctx.fill()
    ctx.strokeStyle = 'rgba(15,23,42,0.85)'
    ctx.lineWidth = 0.9
    ctx.stroke()
  }
}

// ─── Confusion Matrix component ───────────────────────────────────────────────

function ConfusionMatrix({ cm, classes }: { cm: number[][]; classes: number }) {
  if (cm.length === 0) return null
  const total = cm.flat().reduce((s, v) => s + v, 0) || 1
  const classLabels = Array.from({ length: classes }, (_, k) => `C${k}`)

  return (
    <div>
      <div className="text-[10px] text-slate-500 mb-1.5 uppercase tracking-wider">Confusion Matrix</div>
      <div className="overflow-x-auto">
        <table className="text-[10px] border-collapse">
          <thead>
            <tr>
              <td className="pr-1 pb-1 text-slate-600 text-right">true↓ pred→</td>
              {classLabels.map(l => (
                <td key={l} className="w-8 text-center pb-1 text-slate-500">{l}</td>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, r) => (
              <tr key={r}>
                <td className="pr-1 text-slate-500 text-right">{classLabels[r]}</td>
                {row.map((v, c) => {
                  const isCorrect = r === c
                  const frac = v / total
                  const bg = isCorrect
                    ? `rgba(99,102,241,${(frac * 0.8 + 0.05).toFixed(2)})`
                    : v > 0 ? `rgba(239,68,68,${(frac * 0.7 + 0.04).toFixed(2)})` : 'transparent'
                  return (
                    <td
                      key={c}
                      className="w-8 h-6 text-center font-mono"
                      style={{ background: bg, color: v > 0 ? '#e2e8f0' : '#475569' }}
                    >
                      {v}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Main Visualization component ─────────────────────────────────────────────

export function Visualization() {
  const initTrain = getTrainConfig()
  const initData = getDataConfig()

  // UI state
  const [classes, setClasses] = useState(initData.classes)
  const [separation, setSeparation] = useState(initData.separation)
  const [noise, setNoise] = useState(initData.noise)
  const [n, setN] = useState(initData.n)
  const [optName, setOptName] = useState<OptimizerName>(initTrain.optimizer)
  const [lrKey, setLrKey] = useState(String(initTrain.lr))
  const [l2, setL2] = useState(initTrain.l2)
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [loss, setLoss] = useState(0)
  const [acc, setAcc] = useState(0)
  const [confMatrix, setConfMatrix] = useState<number[][]>([])

  // Refs (no re-render on change)
  const surfaceCanvasRef = useRef<HTMLCanvasElement>(null)
  const lossCanvasRef = useRef<HTMLCanvasElement>(null)
  const accCanvasRef = useRef<HTMLCanvasElement>(null)
  const lossChartRef = useRef<LossChart | null>(null)
  const accChartRef = useRef<LossChart | null>(null)
  const rafRef = useRef<number | null>(null)
  const isTrainingRef = useRef(false)
  const modelRef = useRef<SoftmaxClassifier | null>(null)
  const epochRef = useRef(0)
  const cmCountRef = useRef(0)
  const dataRef = useRef<Point2D[]>(getData())
  const boundsRef = useRef(getDataBounds(getData()))
  const classesRef = useRef(initData.classes)

  // ── Build model ─────────────────────────────────────────────────────────────

  const buildModel = useCallback((numCls: number, optN: OptimizerName, lr: number, lmbda: number) => {
    const opt = createOptimizer(optN, lr)
    modelRef.current = new SoftmaxClassifier(numCls, lmbda, opt, 42)
    epochRef.current = 0
    cmCountRef.current = 0
    lossChartRef.current?.reset()
    accChartRef.current?.reset()
    setEpoch(0)
    setLoss(0)
    setAcc(0)
    setConfMatrix([])
  }, [])

  // ── Draw surface ────────────────────────────────────────────────────────────

  const drawSurfaceFrame = useCallback(() => {
    const canvas = surfaceCanvasRef.current
    const model = modelRef.current
    if (!canvas || !model) return

    const data = dataRef.current
    const bounds = boundsRef.current
    const numCls = classesRef.current
    const xs = makeGridCoords(bounds.xMin, bounds.xMax, GRID_SIZE)
    const ys = makeGridCoords(bounds.yMin, bounds.yMax, GRID_SIZE)
    const { grid, classGrid } = model.predictGrid(xs, ys)

    drawSurface(canvas, grid, classGrid, data, bounds, numCls, model)
    lossChartRef.current?.resize()
    lossChartRef.current?.draw()
    accChartRef.current?.resize()
    accChartRef.current?.draw()
  }, [])

  // ── Training loop ────────────────────────────────────────────────────────────

  const trainLoop = useCallback(() => {
    if (!isTrainingRef.current) return
    const model = modelRef.current
    if (!model) return

    for (let i = 0; i < EPOCHS_PER_FRAME; i++) {
      const result = model.trainEpoch(dataRef.current, 32)
      epochRef.current++
      lossChartRef.current?.push(result.loss)
      accChartRef.current?.push(result.acc)
      cmCountRef.current++

      if (i === EPOCHS_PER_FRAME - 1) {
        setEpoch(epochRef.current)
        setLoss(result.loss)
        setAcc(result.acc)
      }
    }

    drawSurfaceFrame()

    if (cmCountRef.current >= CM_THROTTLE) {
      cmCountRef.current = 0
      setConfMatrix(model.confusionMatrix(dataRef.current))
    }

    if (isTrainingRef.current) {
      rafRef.current = requestAnimationFrame(trainLoop)
    }
  }, [drawSurfaceFrame])

  // ── Stop + reset helper ──────────────────────────────────────────────────────

  const stopTraining = useCallback(() => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
  }, [])

  // ── On mount ─────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (lossCanvasRef.current) {
      lossChartRef.current = new LossChart(lossCanvasRef.current, { color: '#f87171', label: 'Loss' })
    }
    if (accCanvasRef.current) {
      accChartRef.current = new LossChart(accCanvasRef.current, { color: '#34d399', label: 'Accuracy' })
    }

    const dataCfg = getDataConfig()
    const trainCfg = getTrainConfig()
    classesRef.current = dataCfg.classes
    buildModel(dataCfg.classes, trainCfg.optimizer, trainCfg.lr, trainCfg.l2)
    drawSurfaceFrame()

    const unsub = subscribe(() => {
      dataRef.current = getData()
      boundsRef.current = getDataBounds(getData())
      const newCfg = getDataConfig()
      classesRef.current = newCfg.classes
      stopTraining()
      buildModel(newCfg.classes, optName, parseFloat(lrKey), l2)
      drawSurfaceFrame()
    })

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      unsub()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Hyperparameter change handler ────────────────────────────────────────────

  const handleParamChange = useCallback((
    numCls: number,
    optN: OptimizerName,
    lr: number,
    lmbda: number,
  ) => {
    stopTraining()
    classesRef.current = numCls
    buildModel(numCls, optN, lr, lmbda)
    drawSurfaceFrame()
  }, [stopTraining, buildModel, drawSurfaceFrame])

  // ── Data controls (update dataStore + rebuild) ────────────────────────────────

  const handleDataChange = useCallback((
    newClasses: number,
    newSep: number,
    newNoise: number,
    newN: number,
  ) => {
    setDataConfig({ classes: newClasses, separation: newSep, noise: newNoise, n: newN })
    regenerate()
    dataRef.current = getData()
    boundsRef.current = getDataBounds(getData())
    classesRef.current = newClasses
    stopTraining()
    buildModel(newClasses, optName, parseFloat(lrKey), l2)
    drawSurfaceFrame()
  }, [stopTraining, buildModel, drawSurfaceFrame, optName, lrKey, l2])

  // ── Button handlers ───────────────────────────────────────────────────────────

  const handleTrain = () => {
    isTrainingRef.current = true
    setIsTraining(true)
    rafRef.current = requestAnimationFrame(trainLoop)
  }

  const handlePause = () => stopTraining()

  const handleStep = () => {
    stopTraining()
    const model = modelRef.current
    if (!model) return
    const result = model.trainEpoch(dataRef.current, 32)
    epochRef.current++
    lossChartRef.current?.push(result.loss)
    accChartRef.current?.push(result.acc)
    setEpoch(epochRef.current)
    setLoss(result.loss)
    setAcc(result.acc)
    setConfMatrix(model.confusionMatrix(dataRef.current))
    drawSurfaceFrame()
  }

  const handleReset = () => {
    stopTraining()
    buildModel(classes, optName, parseFloat(lrKey), l2)
    drawSurfaceFrame()
  }

  const handleRegenerate = () => {
    handleDataChange(classes, separation, noise, n)
  }

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 h-full">

      {/* Row 1: Controls */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 shrink-0">

        {/* Data controls */}
        <Slider
          label="Classes (K)"
          value={classes}
          min={2} max={4} step={1}
          onChange={v => {
            setClasses(v)
            handleDataChange(v, separation, noise, n)
          }}
          disabled={isTraining}
        />
        <Slider
          label="Separation"
          value={separation}
          min={0.3} max={3.0} step={0.1}
          onChange={v => {
            setSeparation(v)
            handleDataChange(classes, v, noise, n)
          }}
          format={v => v.toFixed(1)}
          disabled={isTraining}
        />
        <Slider
          label="Noise (σ)"
          value={noise}
          min={0.05} max={0.6} step={0.05}
          onChange={v => {
            setNoise(v)
            handleDataChange(classes, separation, v, n)
          }}
          format={v => v.toFixed(2)}
          disabled={isTraining}
        />
        <Slider
          label="Samples (n)"
          value={n}
          min={50} max={500} step={50}
          onChange={v => {
            setN(v)
            handleDataChange(classes, separation, noise, v)
          }}
          disabled={isTraining}
        />

        {/* Train controls */}
        <Select
          label="Optimizer"
          value={optName}
          options={OPT_OPTIONS}
          onChange={v => {
            const o = v as OptimizerName
            setOptName(o)
            handleParamChange(classes, o, parseFloat(lrKey), l2)
          }}
          disabled={isTraining}
        />
        <Select
          label="Learning rate"
          value={lrKey}
          options={LR_OPTIONS}
          onChange={v => {
            setLrKey(v)
            handleParamChange(classes, optName, parseFloat(v), l2)
          }}
          disabled={isTraining}
        />
        <Slider
          label="L2 λ"
          value={l2}
          min={0} max={0.5} step={0.01}
          onChange={v => {
            setL2(v)
            handleParamChange(classes, optName, parseFloat(lrKey), v)
          }}
          format={v => v.toFixed(2)}
          disabled={isTraining}
        />

        {/* Action buttons */}
        <div className="flex flex-col justify-end gap-2">
          <ButtonRow
            buttons={[
              isTraining
                ? { label: 'Pause',      onClick: handlePause,      variant: 'danger' }
                : { label: 'Train',      onClick: handleTrain,      variant: 'primary' },
              { label: 'Step',           onClick: handleStep,       disabled: isTraining },
              { label: 'Reset',          onClick: handleReset,      disabled: isTraining },
            ]}
          />
          <ButtonRow
            buttons={[
              { label: 'Regen data', onClick: handleRegenerate, disabled: isTraining, variant: 'secondary' },
            ]}
          />
        </div>
      </div>

      {/* Row 2: Metrics */}
      <div className="flex flex-wrap gap-3 shrink-0">
        <Stat label="Epoch"    value={epoch}                             color="text-slate-300" />
        <Stat label="Loss"     value={loss}                              color="text-red-300" />
        <Stat label="Accuracy" value={`${(acc * 100).toFixed(1)}%`}     color="text-emerald-300" />
        <Stat label="Params"   value={modelRef.current?.paramCount ?? 0} color="text-slate-400" />
      </div>

      {/* Row 3: Main visuals */}
      <div className="flex flex-1 gap-4 min-h-0">

        {/* Probability surface */}
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <SectionHeading>Probability Surface</SectionHeading>
          <div
            className="flex-1 min-h-0 rounded-md overflow-hidden border border-slate-700/60 bg-slate-950"
            style={{ minHeight: 240 }}
          >
            <canvas ref={surfaceCanvasRef} className="w-full h-full block" />
          </div>
        </div>

        {/* Right panel */}
        <div className="flex flex-col gap-3 w-56 shrink-0 overflow-y-auto">

          <SectionHeading>Loss</SectionHeading>
          <div className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 h-28 shrink-0">
            <canvas ref={lossCanvasRef} className="w-full h-full block" />
          </div>

          <SectionHeading>Accuracy</SectionHeading>
          <div className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 h-28 shrink-0">
            <canvas ref={accCanvasRef} className="w-full h-full block" />
          </div>

          {confMatrix.length > 0 && (
            <>
              <SectionHeading>Confusion</SectionHeading>
              <ConfusionMatrix cm={confMatrix} classes={classes} />
            </>
          )}

          {/* Class legend */}
          <SectionHeading>Legend</SectionHeading>
          <div className="flex flex-col gap-1.5">
            {Array.from({ length: classes }, (_, k) => (
              <div key={k} className="flex items-center gap-2 text-xs text-slate-400">
                <span
                  className="inline-block w-3 h-3 rounded-full flex-shrink-0"
                  style={{ background: CLASS_COLORS_RGBA[k] }}
                />
                Class {k}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
