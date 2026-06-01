import { useEffect, useRef, useState, useCallback } from 'react'
import { MLP, type ActivationName, type LayerConfig } from './mlp'
import {
  getDataConfig,
  setDataConfig,
  getTrainConfig,
  getData,
  setTrainConfig,
  regenerate,
  subscribe,
  numClasses,
  type DatasetName,
} from './dataStore'
import { createOptimizer, type OptimizerName } from '../../lib/optimizers'
import { Heatmap } from '../../lib/heatmap'
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

const GRID_SIZE = 50          // decision boundary grid resolution
const EPOCHS_PER_FRAME = 3    // epochs per RAF tick
const NEURON_GRID = 20        // neuron activation map resolution (smaller = faster)
const NEURON_THROTTLE = 5     // update neuron maps every N epochs

const OPT_OPTIONS: Array<{ value: OptimizerName; label: string }> = [
  { value: 'SGD',      label: 'SGD' },
  { value: 'Momentum', label: 'Momentum' },
  { value: 'RMSProp',  label: 'RMSProp' },
  { value: 'Adam',     label: 'Adam' },
]

const ACT_OPTIONS: Array<{ value: ActivationName; label: string }> = [
  { value: 'relu',      label: 'ReLU' },
  { value: 'leakyrelu', label: 'LeakyReLU' },
  { value: 'tanh',      label: 'Tanh' },
  { value: 'sigmoid',   label: 'Sigmoid' },
]

const LR_OPTIONS = ['0.0001','0.0003','0.001','0.003','0.01','0.03','0.1'].map(v => ({ value: v, label: v }))

const CLASS_COLORS_RGBA = [
  'rgba(129,140,248,0.9)',  // indigo-400
  'rgba(52,211,153,0.9)',   // emerald-400
  'rgba(251,146,60,0.9)',   // orange-400
]

// ─── Grid helpers ─────────────────────────────────────────────────────────────

function makeGridCoords(min: number, max: number, size: number): number[] {
  return Array.from({ length: size }, (_, i) => min + (i / (size - 1)) * (max - min))
}

function getDataBounds(data: Point2D[]): { xMin: number; xMax: number; yMin: number; yMax: number } {
  if (data.length === 0) return { xMin: -1.5, xMax: 1.5, yMin: -1.5, yMax: 1.5 }
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
  for (const p of data) {
    if (p.x < xMin) xMin = p.x
    if (p.x > xMax) xMax = p.x
    if (p.y < yMin) yMin = p.y
    if (p.y > yMax) yMax = p.y
  }
  const xPad = (xMax - xMin) * 0.1 || 0.2
  const yPad = (yMax - yMin) * 0.1 || 0.2
  return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad }
}

// ─── Decision boundary canvas (custom render with data points overlaid) ───────

function drawDecisionBoundary(
  canvas: HTMLCanvasElement,
  grid: number[][],           // [rows][cols] = prob(class 0)
  data: Point2D[],
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  numCls: number,
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

  // Draw heatmap cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p0 = grid[r][c]  // prob(class 0)
      let R: number, G: number, B: number
      if (numCls === 2) {
        // diverging: class0=indigo, class1=emerald
        if (p0 > 0.5) {
          const t = (p0 - 0.5) * 2  // 0..1
          R = Math.round(15 + t * (99 - 15))   // slate-950 → indigo-ish
          G = Math.round(23 + t * (102 - 23))
          B = Math.round(42 + t * (241 - 42))
        } else {
          const t = (0.5 - p0) * 2  // 0..1
          R = Math.round(15 + t * (6 - 15))
          G = Math.round(23 + t * (78 - 23))
          B = Math.round(42 + t * (59 - 42))
        }
      } else {
        // 3-class: blend between indigo/emerald/orange
        // p0 is prob class 0, but for 3-class we'd need more info
        // approximate: map p0 to hue
        const t = p0
        R = Math.round(15 + t * 84)
        G = Math.round(23 + t * 79)
        B = Math.round(42 + t * 199)
      }
      ctx.fillStyle = `rgb(${R},${G},${B})`
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
    }
  }

  // Draw data points
  const { xMin, xMax, yMin, yMax } = bounds
  const toX = (x: number) => ((x - xMin) / (xMax - xMin)) * w
  const toY = (y: number) => h - ((y - yMin) / (yMax - yMin)) * h

  for (const pt of data) {
    const cx = toX(pt.x)
    const cy = toY(pt.y)
    if (cx < -5 || cx > w + 5 || cy < -5 || cy > h + 5) continue

    ctx.beginPath()
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2)
    ctx.fillStyle = CLASS_COLORS_RGBA[pt.label] ?? '#94a3b8'
    ctx.fill()
    ctx.strokeStyle = 'rgba(15,23,42,0.8)'
    ctx.lineWidth = 0.8
    ctx.stroke()
  }
}

// ─── Architecture editor component ───────────────────────────────────────────

function ArchEditor({
  layers,
  onChange,
  disabled,
}: {
  layers: LayerConfig[]
  onChange: (layers: LayerConfig[]) => void
  disabled: boolean
}) {
  const addLayer = () => {
    if (layers.length >= 4) return
    onChange([...layers, { neurons: 8 }])
  }
  const removeLayer = (idx: number) => {
    if (layers.length <= 1) return
    onChange(layers.filter((_, i) => i !== idx))
  }
  const setNeurons = (idx: number, n: number) => {
    const next = layers.map((l, i) => i === idx ? { neurons: n } : l)
    onChange(next)
  }

  return (
    <div className="space-y-2">
      <div className="text-xs text-slate-400 mb-1">Hidden layers ({layers.length}/4)</div>
      {layers.map((l, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 w-14 shrink-0">Layer {i + 1}</span>
          <input
            type="range"
            min={1} max={16} step={1}
            value={l.neurons}
            disabled={disabled}
            onChange={e => setNeurons(i, Number(e.target.value))}
            className="slider-track flex-1 h-1.5 appearance-none rounded-full bg-slate-700 accent-indigo-500 cursor-pointer disabled:opacity-40"
          />
          <span className="metric text-[10px] text-indigo-300 w-5 text-right">{l.neurons}</span>
          <button
            onClick={() => removeLayer(i)}
            disabled={disabled || layers.length <= 1}
            className="text-slate-500 hover:text-red-400 disabled:opacity-30 text-xs px-1"
            title="Remove layer"
          >✕</button>
        </div>
      ))}
      <button
        onClick={addLayer}
        disabled={disabled || layers.length >= 4}
        className="text-xs text-indigo-400 hover:text-indigo-300 disabled:opacity-30 disabled:cursor-not-allowed"
      >
        + Add layer
      </button>
    </div>
  )
}

// ─── Neuron activation maps ───────────────────────────────────────────────────

function NeuronMaps({
  grids,
  neuronInfo,
  hiddenSizes,
}: {
  grids: number[][][]
  neuronInfo: Array<{ layerIdx: number; neuronIdx: number }>
  hiddenSizes: number[]
}) {
  const canvasRefs = useRef<Map<number, HTMLCanvasElement>>(new Map())
  const heatmapRefs = useRef<Map<number, Heatmap>>(new Map())

  useEffect(() => {
    // Initialize or update heatmaps
    grids.forEach((grid, idx) => {
      const canvas = canvasRefs.current.get(idx)
      if (!canvas) return

      let hm = heatmapRefs.current.get(idx)
      if (!hm) {
        hm = new Heatmap(canvas, { colormap: 'grayscale', padding: 1 })
        heatmapRefs.current.set(idx, hm)
      }
      hm.render(grid, { colormap: 'grayscale', normMin: -1, normMax: 1 })
    })
  }, [grids])

  if (grids.length === 0) return null

  // Group by layer
  const byLayer: number[][] = hiddenSizes.map(() => [])
  neuronInfo.forEach((info, idx) => {
    if (byLayer[info.layerIdx]) byLayer[info.layerIdx].push(idx)
  })

  return (
    <div className="space-y-3">
      {byLayer.map((neuronIdxs, layerIdx) => (
        <div key={layerIdx}>
          <div className="text-[10px] text-slate-500 mb-1">Layer {layerIdx + 1} activations</div>
          <div className="flex flex-wrap gap-1.5">
            {neuronIdxs.map(idx => (
              <div key={idx} className="flex flex-col items-center gap-0.5">
                <canvas
                  ref={el => {
                    if (el) canvasRefs.current.set(idx, el)
                  }}
                  style={{ width: 40, height: 40 }}
                  className="rounded border border-slate-700/50"
                />
                <span className="text-[8px] text-slate-600">
                  n{neuronInfo[idx].neuronIdx + 1}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Main Visualization component ─────────────────────────────────────────────

export function Visualization() {
  const initTrain = getTrainConfig()

  // ── UI State ──────────────────────────────────────────────────────────────
  const [hiddenLayers, setHiddenLayers] = useState<LayerConfig[]>(initTrain.hiddenLayers)
  const [activation, setActivation] = useState<ActivationName>(initTrain.activation)
  const [optName, setOptName] = useState<OptimizerName>(initTrain.optimizer)
  const [lrKey, setLrKey] = useState(String(initTrain.lr))
  const [l2, setL2] = useState(initTrain.l2)
  const [dropout, setDropout] = useState(initTrain.dropout)
  const [batchSize, setBatchSize] = useState(initTrain.batchSize)
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [loss, setLoss] = useState(0)
  const [acc, setAcc] = useState(0)
  const [neuronGrids, setNeuronGrids] = useState<number[][][]>([])
  const [neuronInfo, setNeuronInfo] = useState<Array<{ layerIdx: number; neuronIdx: number }>>([])
  const [hiddenSizes, setHiddenSizes] = useState<number[]>([])

  // ── Refs (mutable training state, no re-render) ───────────────────────────
  const boundaryCanvasRef = useRef<HTMLCanvasElement>(null)
  const lossCanvasRef = useRef<HTMLCanvasElement>(null)
  const lossChartRef = useRef<LossChart | null>(null)
  const accChartRef = useRef<LossChart | null>(null)
  const accCanvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const isTrainingRef = useRef(false)
  const mlpRef = useRef<MLP | null>(null)
  const epochRef = useRef(0)
  const neuronEpochCountRef = useRef(0)
  const dataRef = useRef<Point2D[]>(getData())
  const boundsRef = useRef(getDataBounds(getData()))

  // ── Build / rebuild the MLP ───────────────────────────────────────────────
  const buildMLP = useCallback((cfg: {
    hiddenLayers: LayerConfig[]
    activation: ActivationName
    optName: OptimizerName
    lr: number
    l2: number
    dropout: number
  }) => {
    const dataCfg = getDataConfig()
    const nCls = numClasses(dataCfg.dataset)
    const opt = createOptimizer(cfg.optName, cfg.lr)
    mlpRef.current = new MLP(
      {
        hiddenLayers: cfg.hiddenLayers,
        activation: cfg.activation,
        numClasses: nCls,
        l2: cfg.l2,
        dropout: cfg.dropout,
      },
      opt,
      42,
    )
    epochRef.current = 0
    neuronEpochCountRef.current = 0
    lossChartRef.current?.reset()
    accChartRef.current?.reset()
    setEpoch(0)
    setLoss(0)
    setAcc(0)
    setNeuronGrids([])
    setNeuronInfo([])
    setHiddenSizes(cfg.hiddenLayers.map(l => l.neurons))
  }, [])

  // ── Initialize canvases and MLP on mount ──────────────────────────────────
  useEffect(() => {
    if (lossCanvasRef.current) {
      lossChartRef.current = new LossChart(lossCanvasRef.current, {
        color: '#f87171',
        label: 'Loss',
      })
    }
    if (accCanvasRef.current) {
      accChartRef.current = new LossChart(accCanvasRef.current, {
        color: '#34d399',
        label: 'Accuracy',
      })
    }

    buildMLP({ hiddenLayers, activation, optName, lr: parseFloat(lrKey), l2, dropout })

    // Subscribe to dataset changes
    const unsub = subscribe(() => {
      dataRef.current = getData()
      boundsRef.current = getDataBounds(getData())
      // Reset the MLP when data changes
      buildMLP({ hiddenLayers, activation, optName, lr: parseFloat(lrKey), l2, dropout })
      drawBoundary()
    })

    drawBoundary()

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      unsub()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Draw decision boundary ────────────────────────────────────────────────
  const drawBoundary = useCallback(() => {
    const canvas = boundaryCanvasRef.current
    const mlp = mlpRef.current
    if (!canvas || !mlp) return

    const data = dataRef.current
    const bounds = boundsRef.current
    const xs = makeGridCoords(bounds.xMin, bounds.xMax, GRID_SIZE)
    const ys = makeGridCoords(bounds.yMin, bounds.yMax, GRID_SIZE)
    const grid = mlp.predictGrid(xs, ys)
    const nCls = numClasses(getDataConfig().dataset)

    drawDecisionBoundary(canvas, grid, data, bounds, nCls)
    lossChartRef.current?.resize()
    lossChartRef.current?.draw()
    accChartRef.current?.resize()
    accChartRef.current?.draw()
  }, [])

  // Redraw on viewport resize (canvases re-fit to their container on each draw)
  useEffect(() => {
    let raf = 0
    const onResize = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => drawBoundary())
    }
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      cancelAnimationFrame(raf)
    }
  }, [drawBoundary])

  // ── Update neuron maps (throttled) ────────────────────────────────────────
  const updateNeuronMaps = useCallback(() => {
    const mlp = mlpRef.current
    if (!mlp) return
    const bounds = boundsRef.current
    const xs = makeGridCoords(bounds.xMin, bounds.xMax, NEURON_GRID)
    const ys = makeGridCoords(bounds.yMin, bounds.yMax, NEURON_GRID)
    const { grids, neuronInfo: info } = mlp.neuronActivations(xs, ys)
    setNeuronGrids(grids)
    setNeuronInfo(info)
    setHiddenSizes(mlp.hiddenSizes)
  }, [])

  // ── Training loop ─────────────────────────────────────────────────────────
  const trainLoop = useCallback(() => {
    if (!isTrainingRef.current) return
    const mlp = mlpRef.current
    if (!mlp) return

    for (let i = 0; i < EPOCHS_PER_FRAME; i++) {
      const result = mlp.trainEpoch(dataRef.current, batchSize)
      epochRef.current++
      lossChartRef.current?.push(result.loss)
      accChartRef.current?.push(result.acc)
      neuronEpochCountRef.current++

      if (i === EPOCHS_PER_FRAME - 1) {
        setEpoch(epochRef.current)
        setLoss(result.loss)
        setAcc(result.acc)
      }
    }

    drawBoundary()

    // Throttle neuron map updates
    if (neuronEpochCountRef.current >= NEURON_THROTTLE) {
      neuronEpochCountRef.current = 0
      updateNeuronMaps()
    }

    if (isTrainingRef.current) {
      rafRef.current = requestAnimationFrame(trainLoop)
    }
  }, [batchSize, drawBoundary, updateNeuronMaps])

  // ── Re-init when architecture/hyperparams change ──────────────────────────
  const handleConfigChange = useCallback((
    newLayers: LayerConfig[],
    newAct: ActivationName,
    newOpt: OptimizerName,
    newLr: number,
    newL2: number,
    newDropout: number,
  ) => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    isTrainingRef.current = false
    setIsTraining(false)

    buildMLP({
      hiddenLayers: newLayers,
      activation: newAct,
      optName: newOpt,
      lr: newLr,
      l2: newL2,
      dropout: newDropout,
    })
    drawBoundary()
  }, [buildMLP, drawBoundary])

  // ─── Button handlers ──────────────────────────────────────────────────────
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

    const mlp = mlpRef.current
    if (!mlp) return
    const result = mlp.trainEpoch(dataRef.current, batchSize)
    epochRef.current++
    lossChartRef.current?.push(result.loss)
    accChartRef.current?.push(result.acc)
    setEpoch(epochRef.current)
    setLoss(result.loss)
    setAcc(result.acc)
    drawBoundary()
    updateNeuronMaps()
  }

  const handleReset = () => {
    isTrainingRef.current = false
    setIsTraining(false)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    buildMLP({ hiddenLayers, activation, optName, lr: parseFloat(lrKey), l2, dropout })
    drawBoundary()
  }

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-4 min-h-full lg:h-full">

      {/* ── Row 1: Controls ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 shrink-0">
        <Select
          label="Dataset"
          value={getDataConfig().dataset}
          options={[
            { value: 'moons',   label: 'Two Moons' },
            { value: 'circles', label: 'Circles' },
            { value: 'xor',     label: 'XOR' },
            { value: 'spiral',  label: 'Spiral 2-class' },
            { value: 'spiral3', label: 'Spiral 3-class' },
          ]}
          onChange={v => {
            setDataConfig({ dataset: v as DatasetName })
            regenerate()
            dataRef.current = getData()
            boundsRef.current = getDataBounds(getData())
            handleConfigChange(hiddenLayers, activation, optName, parseFloat(lrKey), l2, dropout)
          }}
          disabled={isTraining}
        />
        <Select
          label="Activation"
          value={activation}
          options={ACT_OPTIONS}
          onChange={v => {
            const a = v as ActivationName
            setActivation(a)
            handleConfigChange(hiddenLayers, a, optName, parseFloat(lrKey), l2, dropout)
          }}
          disabled={isTraining}
        />
        <Select
          label="Optimizer"
          value={optName}
          options={OPT_OPTIONS}
          onChange={v => {
            const o = v as OptimizerName
            setOptName(o)
            handleConfigChange(hiddenLayers, activation, o, parseFloat(lrKey), l2, dropout)
          }}
          disabled={isTraining}
        />
        <Select
          label="Learning rate"
          value={lrKey}
          options={LR_OPTIONS}
          onChange={v => {
            setLrKey(v)
            handleConfigChange(hiddenLayers, activation, optName, parseFloat(v), l2, dropout)
          }}
          disabled={isTraining}
        />
        <Slider
          label="L2 λ"
          value={l2}
          min={0} max={0.1} step={0.001}
          onChange={v => {
            setL2(v)
            handleConfigChange(hiddenLayers, activation, optName, parseFloat(lrKey), v, dropout)
          }}
          format={v => v.toFixed(3)}
          disabled={isTraining}
        />
        <Slider
          label="Dropout"
          value={dropout}
          min={0} max={0.8} step={0.05}
          onChange={v => {
            setDropout(v)
            handleConfigChange(hiddenLayers, activation, optName, parseFloat(lrKey), l2, v)
          }}
          format={v => v.toFixed(2)}
          disabled={isTraining}
        />
        <Slider
          label="Batch size"
          value={batchSize}
          min={8} max={256} step={8}
          onChange={setBatchSize}
          disabled={isTraining}
        />
        <div className="flex flex-col justify-end">
          <ButtonRow
            buttons={[
              isTraining
                ? { label: 'Pause', onClick: handlePause, variant: 'danger' }
                : { label: 'Train', onClick: handleTrain, variant: 'primary' },
              { label: 'Step',  onClick: handleStep,  disabled: isTraining },
              { label: 'Reset', onClick: handleReset, disabled: isTraining },
            ]}
          />
        </div>
      </div>

      {/* ── Row 2: Metrics ── */}
      <div className="flex flex-wrap gap-3 shrink-0">
        <Stat label="Epoch"    value={epoch} color="text-slate-300" />
        <Stat label="Loss"     value={loss}  color="text-red-300" />
        <Stat label="Accuracy" value={`${(acc * 100).toFixed(1)}%`} color="text-emerald-300" />
        <Stat label="Params"   value={mlpRef.current?.paramCount ?? 0} color="text-slate-400" />
      </div>

      {/* ── Row 3: Main visuals ── */}
      <div className="flex flex-col lg:flex-row flex-1 gap-4 min-h-0">

        {/* Decision boundary */}
        <div className="flex flex-col gap-2 h-80 lg:h-auto lg:flex-1 min-h-0">
          <SectionHeading>Decision Boundary</SectionHeading>
          <div className="flex-1 min-h-0 rounded-md overflow-hidden border border-slate-700/60 bg-slate-950"
               style={{ minHeight: 240 }}>
            <canvas ref={boundaryCanvasRef} className="w-full h-full block" />
          </div>
        </div>

        {/* Right panel: architecture + charts + neuron maps */}
        <div className="flex flex-col gap-3 w-full lg:w-56 shrink-0 lg:overflow-y-auto">

          <SectionHeading>Architecture</SectionHeading>
          <ArchEditor
            layers={hiddenLayers}
            disabled={isTraining}
            onChange={newLayers => {
              setHiddenLayers(newLayers)
              setTrainConfig({ hiddenLayers: newLayers })
              handleConfigChange(newLayers, activation, optName, parseFloat(lrKey), l2, dropout)
            }}
          />

          <SectionHeading>Loss</SectionHeading>
          <div className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 h-28 shrink-0">
            <canvas ref={lossCanvasRef} className="w-full h-full block" />
          </div>

          <SectionHeading>Accuracy</SectionHeading>
          <div className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 h-28 shrink-0">
            <canvas ref={accCanvasRef} className="w-full h-full block" />
          </div>

          {neuronGrids.length > 0 && (
            <>
              <SectionHeading>Neuron Activations</SectionHeading>
              <NeuronMaps
                grids={neuronGrids}
                neuronInfo={neuronInfo}
                hiddenSizes={hiddenSizes}
              />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
