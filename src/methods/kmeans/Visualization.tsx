import { useEffect, useRef, useState, useCallback } from 'react'
import { KMeans, fitInertia, type InitMethod } from './kmeans'
import { getConfig, setConfig, regenerate, subscribe, getData } from './dataStore'
import {
  Slider,
  Select,
  ButtonRow,
  Stat,
  SectionHeading,
} from '../../components/Controls'
import { LossChart } from '../../lib/losschart'
import type { KMeansPoint } from './kmeans'

// ─── Constants ────────────────────────────────────────────────────────────────

/** Voronoi background grid resolution (points per axis). Keep cheap. */
const VORONOI_RES = 80
/** Milliseconds between animation steps */
const STEP_INTERVAL_MS = 220
/** Max auto iterations before stopping */
const MAX_ITERS = 200
/** Max k for elbow plot */
const ELBOW_MAX_K = 8

// Per-cluster colours (indigo, emerald, amber, rose, cyan, violet, teal, orange)
const CLUSTER_COLORS: string[] = [
  '#818cf8', // indigo-400
  '#34d399', // emerald-400
  '#fbbf24', // amber-400
  '#f87171', // rose-400
  '#22d3ee', // cyan-400
  '#a78bfa', // violet-400
  '#2dd4bf', // teal-400
  '#fb923c', // orange-400
]

// Same colours as RGBA with varying alpha for Voronoi background tint
const CLUSTER_BG_RGBA: Array<[number, number, number]> = [
  [129, 140, 248],
  [52,  211, 153],
  [251, 191, 36],
  [248, 113, 113],
  [34,  211, 238],
  [167, 139, 250],
  [45,  212, 191],
  [251, 146, 60],
]

const INIT_OPTIONS: Array<{ value: InitMethod; label: string }> = [
  { value: 'kmeanspp', label: 'K-Means++' },
  { value: 'random',   label: 'Random' },
]

// ─── Voronoi background renderer ─────────────────────────────────────────────

function drawVoronoi(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  centroids: KMeansPoint[],
): void {
  if (centroids.length === 0) return
  const res = VORONOI_RES
  const cellW = w / res
  const cellH = h / res

  for (let r = 0; r < res; r++) {
    const py = (r + 0.5) / res
    for (let c = 0; c < res; c++) {
      const px = (c + 0.5) / res
      // Find nearest centroid (pixel coords normalised to [0,1])
      let best = 0
      let bestD = Infinity
      for (let k = 0; k < centroids.length; k++) {
        const dx = px - centroids[k].x
        const dy = py - centroids[k].y
        const d = dx * dx + dy * dy
        if (d < bestD) { bestD = d; best = k }
      }
      const [R, G, B] = CLUSTER_BG_RGBA[best % CLUSTER_BG_RGBA.length]
      ctx.fillStyle = `rgba(${R},${G},${B},0.13)`
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
    }
  }
}

// ─── Main canvas renderer ─────────────────────────────────────────────────────

interface FrameData {
  pts: KMeansPoint[]
  assignments: number[]
  centroids: KMeansPoint[]
  prevCentroids: KMeansPoint[]
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number }
}

function renderFrame(canvas: HTMLCanvasElement, fd: FrameData): void {
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

  const { pts, assignments, centroids, prevCentroids, bounds } = fd
  const { xMin, xMax, yMin, yMax } = bounds

  // Map data coords → pixel coords on [0,w]×[0,h]
  const toX = (x: number) => ((x - xMin) / (xMax - xMin)) * w
  const toY = (y: number) => h - ((y - yMin) / (yMax - yMin)) * h

  // Normalise centroids for Voronoi (pixel space 0..1)
  const normCentroids = centroids.map(c => ({
    x: (c.x - xMin) / (xMax - xMin),
    y: 1 - (c.y - yMin) / (yMax - yMin),
  }))

  // Background
  ctx.fillStyle = '#0f172a'
  ctx.fillRect(0, 0, w, h)

  // Voronoi background tinting
  drawVoronoi(ctx, w, h, normCentroids)

  // Centroid trails (prev → current)
  for (let k = 0; k < centroids.length; k++) {
    const px1 = toX(prevCentroids[k].x)
    const py1 = toY(prevCentroids[k].y)
    const px2 = toX(centroids[k].x)
    const py2 = toY(centroids[k].y)
    const dx = px2 - px1
    const dy = py2 - py1
    if (dx * dx + dy * dy < 1) continue  // skip trivial movement

    const color = CLUSTER_COLORS[k % CLUSTER_COLORS.length]
    ctx.save()
    ctx.strokeStyle = color
    ctx.globalAlpha = 0.45
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(px1, py1)
    ctx.lineTo(px2, py2)
    ctx.stroke()
    // Arrow head
    ctx.setLineDash([])
    ctx.globalAlpha = 0.65
    const angle = Math.atan2(dy, dx)
    const al = 7
    ctx.beginPath()
    ctx.moveTo(px2, py2)
    ctx.lineTo(px2 - al * Math.cos(angle - 0.4), py2 - al * Math.sin(angle - 0.4))
    ctx.lineTo(px2 - al * Math.cos(angle + 0.4), py2 - al * Math.sin(angle + 0.4))
    ctx.closePath()
    ctx.fillStyle = color
    ctx.fill()
    ctx.restore()
  }

  // Data points coloured by assignment
  for (let i = 0; i < pts.length; i++) {
    const cx = toX(pts[i].x)
    const cy = toY(pts[i].y)
    if (cx < -6 || cx > w + 6 || cy < -6 || cy > h + 6) continue
    const color = CLUSTER_COLORS[assignments[i] % CLUSTER_COLORS.length]
    ctx.beginPath()
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.globalAlpha = 0.82
    ctx.fill()
    ctx.globalAlpha = 1
    ctx.strokeStyle = 'rgba(15,23,42,0.6)'
    ctx.lineWidth = 0.8
    ctx.stroke()
  }

  // Centroids as larger star/diamond markers
  for (let k = 0; k < centroids.length; k++) {
    const cx = toX(centroids[k].x)
    const cy = toY(centroids[k].y)
    const color = CLUSTER_COLORS[k % CLUSTER_COLORS.length]

    // Glow ring
    ctx.beginPath()
    ctx.arc(cx, cy, 11, 0, Math.PI * 2)
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.4
    ctx.stroke()
    ctx.globalAlpha = 1

    // Filled centroid
    ctx.beginPath()
    ctx.arc(cx, cy, 7, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.strokeStyle = '#0f172a'
    ctx.lineWidth = 2
    ctx.stroke()

    // White cross in centre
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.9
    ctx.beginPath()
    ctx.moveTo(cx - 3.5, cy); ctx.lineTo(cx + 3.5, cy)
    ctx.moveTo(cx, cy - 3.5); ctx.lineTo(cx, cy + 3.5)
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  // Cluster index label next to each centroid
  ctx.font = '11px JetBrains Mono, monospace'
  ctx.textBaseline = 'middle'
  for (let k = 0; k < centroids.length; k++) {
    const cx = toX(centroids[k].x)
    const cy = toY(centroids[k].y)
    ctx.fillStyle = CLUSTER_COLORS[k % CLUSTER_COLORS.length]
    ctx.textAlign = 'left'
    ctx.fillText(`μ${k}`, cx + 13, cy)
  }
}

// ─── Elbow chart renderer ─────────────────────────────────────────────────────

function renderElbow(
  canvas: HTMLCanvasElement,
  inertias: number[],
  highlightK: number,
): void {
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

  const pad = { top: 14, right: 12, bottom: 28, left: 50 }
  const plotW = w - pad.left - pad.right
  const plotH = h - pad.top - pad.bottom

  ctx.fillStyle = '#0f172a'
  ctx.fillRect(0, 0, w, h)

  if (inertias.length === 0) {
    ctx.fillStyle = '#475569'
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('Press "Compute Elbow" to generate', w / 2, h / 2)
    return
  }

  const yMax = Math.max(...inertias) * 1.05
  const yMin = 0
  const n = inertias.length

  const toX = (ki: number) => pad.left + (ki / (n - 1)) * plotW
  const toY = (v: number) => pad.top + (1 - (v - yMin) / (yMax - yMin + 1e-12)) * plotH

  // Grid
  ctx.strokeStyle = 'rgba(148,163,184,0.08)'
  ctx.lineWidth = 1
  for (let i = 0; i <= 3; i++) {
    const y = pad.top + (i / 3) * plotH
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke()
  }

  // Y labels
  ctx.fillStyle = '#64748b'
  ctx.font = '9px JetBrains Mono, monospace'
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  for (let i = 0; i <= 3; i++) {
    const v = yMax - (i / 3) * yMax
    const y = pad.top + (i / 3) * plotH
    const label = v >= 10000 ? v.toExponential(1) : v.toPrecision(3)
    ctx.fillText(label, pad.left - 4, y)
  }

  // Line
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const x = toX(i)
    const y = toY(inertias[i])
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.strokeStyle = '#818cf8'
  ctx.lineWidth = 2
  ctx.lineJoin = 'round'
  ctx.stroke()

  // Dots + x labels
  for (let i = 0; i < n; i++) {
    const k = i + 1
    const x = toX(i)
    const y = toY(inertias[i])

    ctx.beginPath()
    ctx.arc(x, y, k === highlightK ? 5.5 : 3.5, 0, Math.PI * 2)
    ctx.fillStyle = k === highlightK ? '#f59e0b' : '#818cf8'
    ctx.fill()

    ctx.fillStyle = '#64748b'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText(String(k), x, pad.top + plotH + 4)
  }

  // X axis label
  ctx.fillStyle = '#475569'
  ctx.font = '9px JetBrains Mono, monospace'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  ctx.fillText('k', pad.left + plotW / 2, pad.top + plotH + 18)

  // Border
  ctx.strokeStyle = 'rgba(148,163,184,0.2)'
  ctx.lineWidth = 1
  ctx.strokeRect(pad.left, pad.top, plotW, plotH)
}

// ─── Main Visualization component ─────────────────────────────────────────────

export function Visualization() {
  const initCfg = getConfig()

  // Controls state
  const [k, setK] = useState(3)
  const [initMethod, setInitMethod] = useState<InitMethod>('kmeanspp')
  const [maxIter, setMaxIter] = useState(MAX_ITERS)
  const [isRunning, setIsRunning] = useState(false)

  // Readout state
  const [iteration, setIteration] = useState(0)
  const [inertia, setInertia] = useState(0)
  const [converged, setConverged] = useState(false)
  const [elbowData, setElbowData] = useState<number[]>([])
  const [elbowComputing, setElbowComputing] = useState(false)

  // Dataset state (to sync cluster/sep/noise/n sliders in this tab too)
  const [clusters, setClusters] = useState(initCfg.clusters)
  const [separation, setSeparation] = useState(initCfg.separation)
  const [noise, setNoise] = useState(initCfg.noise)
  const [n, setN] = useState(initCfg.n)

  // Refs
  const mainCanvasRef = useRef<HTMLCanvasElement>(null)
  const lossCanvasRef = useRef<HTMLCanvasElement>(null)
  const elbowCanvasRef = useRef<HTMLCanvasElement>(null)
  const modelRef = useRef<KMeans | null>(null)
  const dataRef = useRef<KMeansPoint[]>(getData())
  const isRunningRef = useRef(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lossChartRef = useRef<LossChart | null>(null)

  // ── Bounds helper ──────────────────────────────────────────────────────

  function getBounds(pts: KMeansPoint[]) {
    if (pts.length === 0) return { xMin: -2, xMax: 2, yMin: -2, yMax: 2 }
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
    for (const p of pts) {
      if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x
      if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y
    }
    const xPad = (xMax - xMin) * 0.12 || 0.5
    const yPad = (yMax - yMin) * 0.12 || 0.5
    return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad }
  }

  // ── Draw current model state ───────────────────────────────────────────

  const drawFrame = useCallback(() => {
    const canvas = mainCanvasRef.current
    const model = modelRef.current
    if (!canvas || !model) return
    const pts = dataRef.current
    renderFrame(canvas, {
      pts,
      assignments: model.assignments(),
      centroids: model.centroids(),
      prevCentroids: model.prevCentroids(),
      bounds: getBounds(pts),
    })
  }, [])

  // ── Init / rebuild model ───────────────────────────────────────────────

  const buildModel = useCallback((
    kVal: number,
    init: InitMethod,
    seed: number,
  ) => {
    const pts = dataRef.current
    const actualK = Math.min(kVal, pts.length || 1)
    const km = new KMeans(actualK, pts, init, seed)
    modelRef.current = km
    setIteration(0)
    setInertia(km.inertia())
    setConverged(false)
    if (lossChartRef.current) lossChartRef.current.reset()
    lossChartRef.current?.push(km.inertia())
    lossChartRef.current?.draw()
  }, [])

  // ── Stop animation ─────────────────────────────────────────────────────

  const stopAnimation = useCallback(() => {
    isRunningRef.current = false
    setIsRunning(false)
    if (timerRef.current !== null) { clearTimeout(timerRef.current); timerRef.current = null }
  }, [])

  // ── Animation tick ─────────────────────────────────────────────────────

  const doStep = useCallback(() => {
    const model = modelRef.current
    if (!model) return
    const { changed, inertia: w } = model.step()
    const iter = model.iteration()
    setIteration(iter)
    setInertia(w)
    lossChartRef.current?.push(w)
    lossChartRef.current?.draw()
    drawFrame()
    if (!changed || iter >= maxIter) {
      setConverged(!changed)
      stopAnimation()
    }
  }, [drawFrame, stopAnimation, maxIter])

  const scheduleNext = useCallback(() => {
    if (!isRunningRef.current) return
    timerRef.current = setTimeout(() => {
      if (!isRunningRef.current) return
      doStep()
      scheduleNext()
    }, STEP_INTERVAL_MS)
  }, [doStep])

  // ── Mount ──────────────────────────────────────────────────────────────

  useEffect(() => {
    // Init loss chart
    if (lossCanvasRef.current && !lossChartRef.current) {
      lossChartRef.current = new LossChart(lossCanvasRef.current, {
        color: '#818cf8',
        label: 'Inertia (WCSS)',
      })
    }

    dataRef.current = getData()
    buildModel(k, initMethod, 42)
    drawFrame()

    const unsub = subscribe(() => {
      dataRef.current = getData()
      stopAnimation()
      buildModel(k, initMethod, 42)
      drawFrame()
      setElbowData([])
    })

    return () => {
      stopAnimation()
      unsub()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-draw elbow whenever elbowData changes
  useEffect(() => {
    if (elbowCanvasRef.current) {
      renderElbow(elbowCanvasRef.current, elbowData, k)
    }
  }, [elbowData, k])

  // ── Button handlers ────────────────────────────────────────────────────

  const handleRun = () => {
    if (isRunningRef.current) return
    isRunningRef.current = true
    setIsRunning(true)
    scheduleNext()
  }

  const handlePause = () => stopAnimation()

  const handleStep = () => {
    stopAnimation()
    doStep()
  }

  const handleReset = () => {
    stopAnimation()
    buildModel(k, initMethod, Math.floor(Math.random() * 100000))
    drawFrame()
    setElbowData([])
  }

  const handleRegenerate = () => {
    stopAnimation()
    const newSeed = Math.floor(Math.random() * 100000)
    setConfig({ clusters, separation, noise, n, seed: newSeed })
    regenerate()
    dataRef.current = getData()
    buildModel(k, initMethod, 42)
    drawFrame()
    setElbowData([])
  }

  const handleKChange = (v: number) => {
    setK(v)
    stopAnimation()
    buildModel(v, initMethod, 42)
    drawFrame()
    setElbowData([])
  }

  const handleInitChange = (v: string) => {
    const im = v as InitMethod
    setInitMethod(im)
    stopAnimation()
    buildModel(k, im, 42)
    drawFrame()
  }

  const handleComputeElbow = () => {
    if (elbowComputing) return
    setElbowComputing(true)
    // Run synchronously — cheap for n ≤ 300 and k ≤ 8
    const pts = dataRef.current
    const inertias = []
    for (let ki = 1; ki <= ELBOW_MAX_K; ki++) {
      inertias.push(fitInertia(pts, ki, initMethod, 42, 100))
    }
    setElbowData(inertias)
    setElbowComputing(false)
  }

  const handleDataChange = (key: 'clusters' | 'separation' | 'noise' | 'n', val: number) => {
    stopAnimation()
    const updates: Record<string, number> = { [key]: val }
    if (key === 'clusters') { setClusters(val); updates.clusters = val }
    if (key === 'separation') { setSeparation(val); updates.separation = val }
    if (key === 'noise') { setNoise(val); updates.noise = val }
    if (key === 'n') { setN(val); updates.n = val }
    setConfig(updates)
    regenerate()
    dataRef.current = getData()
    buildModel(k, initMethod, 42)
    drawFrame()
    setElbowData([])
  }

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-4 h-full">

      {/* Top controls */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 shrink-0">
        <Slider
          label="k (clusters)"
          value={k}
          min={2}
          max={8}
          step={1}
          onChange={handleKChange}
          disabled={isRunning}
        />
        <Select
          label="Init method"
          value={initMethod}
          options={INIT_OPTIONS}
          onChange={handleInitChange}
          disabled={isRunning}
        />
        <Slider
          label="Max iterations"
          value={maxIter}
          min={10}
          max={200}
          step={10}
          onChange={v => setMaxIter(v)}
          disabled={isRunning}
        />
        <Slider
          label="True clusters"
          value={clusters}
          min={2}
          max={6}
          step={1}
          onChange={v => handleDataChange('clusters', v)}
          disabled={isRunning}
        />
        <Slider
          label="Separation"
          value={separation}
          min={0.4}
          max={3.0}
          step={0.1}
          onChange={v => handleDataChange('separation', v)}
          format={v => v.toFixed(1)}
          disabled={isRunning}
        />
        <Slider
          label="Noise (σ)"
          value={noise}
          min={0.05}
          max={0.5}
          step={0.05}
          onChange={v => handleDataChange('noise', v)}
          format={v => v.toFixed(2)}
          disabled={isRunning}
        />
        <Slider
          label="Samples (n)"
          value={n}
          min={30}
          max={300}
          step={10}
          onChange={v => handleDataChange('n', v)}
          disabled={isRunning}
        />

        {/* Action buttons */}
        <div className="flex flex-col justify-end gap-2">
          <ButtonRow
            buttons={[
              isRunning
                ? { label: 'Pause',  onClick: handlePause,  variant: 'danger' }
                : { label: 'Run',    onClick: handleRun,    variant: 'primary' },
              { label: 'Step',   onClick: handleStep,   disabled: isRunning },
              { label: 'Reset',  onClick: handleReset,  disabled: isRunning },
            ]}
          />
          <ButtonRow
            buttons={[
              { label: 'Regen data', onClick: handleRegenerate, disabled: isRunning, variant: 'secondary' },
            ]}
          />
        </div>
      </div>

      {/* Stats row */}
      <div className="flex flex-wrap gap-3 shrink-0">
        <Stat label="Iteration" value={iteration}      color="text-slate-300" />
        <Stat label="Inertia"   value={inertia}        color="text-indigo-300" />
        <Stat
          label="Status"
          value={converged ? 'Converged' : iteration === 0 ? 'Initialized' : 'Running…'}
          color={converged ? 'text-emerald-300' : 'text-amber-300'}
        />
        <Stat label="k"         value={k}              color="text-cyan-300" />
        <Stat label="Init"      value={initMethod === 'kmeanspp' ? 'K-Means++' : 'Random'} color="text-slate-400" />
      </div>

      {/* Main layout: canvas + side panels */}
      <div className="flex flex-1 gap-4 min-h-0">

        {/* Main scatter / Voronoi canvas */}
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <SectionHeading>Clustering · Voronoi regions · centroid trails</SectionHeading>
          <div
            className="flex-1 min-h-0 rounded-md overflow-hidden border border-slate-700/60 bg-slate-950"
            style={{ minHeight: 260 }}
          >
            <canvas ref={mainCanvasRef} className="w-full h-full block" />
          </div>
        </div>

        {/* Right panel: inertia curve + elbow */}
        <div className="flex flex-col gap-3 w-56 shrink-0">

          <SectionHeading>Inertia per iteration</SectionHeading>
          <div
            className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 shrink-0"
            style={{ height: 130 }}
          >
            <canvas ref={lossCanvasRef} className="w-full h-full block" />
          </div>

          <SectionHeading>Elbow method (k = 1…{ELBOW_MAX_K})</SectionHeading>
          <div
            className="rounded-md overflow-hidden border border-slate-700/60 bg-slate-950 shrink-0"
            style={{ height: 130 }}
          >
            <canvas ref={elbowCanvasRef} className="w-full h-full block" />
          </div>
          <button
            onClick={handleComputeElbow}
            disabled={elbowComputing || isRunning}
            className="px-3 py-1.5 text-xs rounded-md font-medium transition-colors duration-150
                       bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600
                       disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {elbowComputing ? 'Computing…' : 'Compute Elbow'}
          </button>

          {/* Legend */}
          <SectionHeading>Legend</SectionHeading>
          <div className="space-y-1.5 text-xs text-slate-400">
            {Array.from({ length: k }, (_, ki) => (
              <div key={ki} className="flex items-center gap-2">
                <span
                  className="inline-block w-3 h-3 rounded-full flex-shrink-0"
                  style={{ background: CLUSTER_COLORS[ki % CLUSTER_COLORS.length] }}
                />
                Cluster {ki}
              </div>
            ))}
            <div className="flex items-center gap-2 mt-2">
              <span className="inline-block w-3 h-3 rounded-full border-2 border-white flex-shrink-0" />
              Centroid (μ_k)
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-6 h-px border-t-2 border-dashed border-indigo-400 flex-shrink-0" />
              Centroid trail
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
