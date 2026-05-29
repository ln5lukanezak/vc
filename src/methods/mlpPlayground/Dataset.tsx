import { useState, useEffect, useRef } from 'react'
import { Slider, Select, Button, SectionHeading, InfoBox } from '../../components/Controls'
import {
  getDataConfig,
  setDataConfig,
  regenerate,
  subscribe,
  getData,
  type DatasetName,
} from './dataStore'
import type { DatasetProps } from '../registry'
import type { Point2D } from '../../lib/datagen'

const DATASET_OPTIONS: Array<{ value: DatasetName; label: string }> = [
  { value: 'moons',   label: 'Two Moons' },
  { value: 'circles', label: 'Concentric Circles' },
  { value: 'xor',     label: 'XOR (4 quadrants)' },
  { value: 'spiral',  label: 'Spiral (2-class)' },
  { value: 'spiral3', label: 'Spiral (3-class)' },
]

const CLASS_COLORS = ['#818cf8', '#34d399', '#fb923c']

function DataPreview({ data }: { data: Point2D[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    ctx.scale(dpr, dpr)

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, w, h)

    // Find data range
    const xs = data.map(p => p.x)
    const ys = data.map(p => p.y)
    const xMin = Math.min(...xs)
    const xMax = Math.max(...xs)
    const yMin = Math.min(...ys)
    const yMax = Math.max(...ys)
    const pad = 20
    const xRange = xMax - xMin || 1
    const yRange = yMax - yMin || 1

    const toCanvasX = (x: number) => pad + ((x - xMin) / xRange) * (w - pad * 2)
    const toCanvasY = (y: number) => h - pad - ((y - yMin) / yRange) * (h - pad * 2)

    for (const pt of data) {
      const cx = toCanvasX(pt.x)
      const cy = toCanvasY(pt.y)
      ctx.beginPath()
      ctx.arc(cx, cy, 3, 0, Math.PI * 2)
      ctx.fillStyle = CLASS_COLORS[pt.label] ?? '#94a3b8'
      ctx.globalAlpha = 0.85
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }, [data])

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-md border border-slate-700 bg-slate-950"
      style={{ height: 200 }}
    />
  )
}

export function Dataset({ onDataChange }: DatasetProps) {
  const [cfg, setCfg] = useState(getDataConfig)
  const [data, setData] = useState<Point2D[]>(getData)

  useEffect(() => {
    return subscribe(() => {
      setCfg(getDataConfig())
      setData(getData())
    })
  }, [])

  function update<K extends keyof typeof cfg>(key: K, value: (typeof cfg)[K]) {
    const next = { ...cfg, [key]: value }
    setCfg(next)
    setDataConfig({ [key]: value })
    regenerate()
    onDataChange?.()
  }

  function handleRegenerate() {
    const newSeed = Math.floor(Math.random() * 100000)
    const next = { ...cfg, seed: newSeed }
    setCfg(next)
    setDataConfig(next)
    regenerate()
    onDataChange?.()
  }

  const datasetDescriptions: Record<DatasetName, string> = {
    moons: 'Two interleaved half-circles. A classic benchmark for non-linear classifiers — no straight line can separate the classes, but a shallow MLP can easily learn the boundary.',
    circles: 'Inner circle vs. outer ring. Radially symmetric: logistic regression fails completely, but even a 1-hidden-layer net with enough neurons can learn a circular boundary.',
    xor: 'Four quadrants in a checkerboard pattern. The simplest non-linearly separable problem — historically used to show that single-layer perceptrons are insufficient.',
    spiral: 'Two interleaved spiral arms. Requires depth to learn the winding structure. A single hidden layer typically cannot fit this; ≥2 layers with enough neurons succeeds.',
    spiral3: '3-class spiral (three arms). Even harder than 2-class spiral. Tests the network\'s ability to separate multiple interleaved non-linear classes.',
  }

  return (
    <div className="max-w-md space-y-6">
      <SectionHeading>Dataset</SectionHeading>

      <Select
        label="Pattern"
        value={cfg.dataset}
        options={DATASET_OPTIONS}
        onChange={v => update('dataset', v as DatasetName)}
      />

      <div className="space-y-4">
        <Slider
          label="Samples (n)"
          value={cfg.n}
          min={50}
          max={500}
          step={50}
          onChange={v => update('n', v)}
        />
        <Slider
          label="Noise (σ)"
          value={cfg.noise}
          min={0}
          max={0.5}
          step={0.01}
          onChange={v => update('noise', v)}
          format={v => v.toFixed(2)}
        />
      </div>

      <Button label="New Random Seed" onClick={handleRegenerate} variant="primary" />

      <DataPreview data={data} />

      <InfoBox>
        <p className="font-medium text-indigo-300 mb-1">{DATASET_OPTIONS.find(o => o.value === cfg.dataset)?.label}</p>
        <p className="text-slate-400">{datasetDescriptions[cfg.dataset]}</p>
      </InfoBox>

      <div className="text-xs text-slate-500 space-y-1">
        <div>Seed: <span className="metric text-slate-400">{cfg.seed}</span></div>
        <div>Points: <span className="metric text-slate-400">{data.length}</span></div>
        <div>Classes: <span className="metric text-slate-400">{cfg.dataset === 'spiral3' ? 3 : 2}</span></div>
      </div>
    </div>
  )
}
