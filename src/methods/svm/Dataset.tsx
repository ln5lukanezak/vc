import { useState, useEffect, useRef } from 'react'
import { Slider, Select, Button, SectionHeading, InfoBox } from '../../components/Controls'
import {
  getDataConfig,
  setDataConfig,
  regenerate,
  subscribe,
  getData,
  type DatasetType,
} from './dataStore'
import type { DatasetProps } from '../registry'
import type { Point2D } from '../../lib/datagen'

const DATASET_OPTIONS: Array<{ value: DatasetType; label: string }> = [
  { value: 'blobs-sep',  label: 'Separable blobs' },
  { value: 'blobs-over', label: 'Overlapping blobs' },
  { value: 'circles',   label: 'Concentric circles' },
  { value: 'moons',     label: 'Two moons' },
]

const DATASET_INFO: Record<DatasetType, string> = {
  'blobs-sep':  'Two well-separated Gaussian blobs. Any kernel (including linear) can classify this perfectly.',
  'blobs-over': 'Two overlapping blobs with tight separation. A soft margin (low C) lets points cross the boundary gracefully.',
  'circles':    'Inner ring vs outer ring. A linear kernel will fail — switch to RBF or poly to see the circular boundary.',
  'moons':      'Two interleaved half-circles. Requires a non-linear kernel (RBF recommended) to separate correctly.',
}

const CLASS_COLORS = ['#818cf8', '#34d399']

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

    const xs = data.map(p => p.x)
    const ys = data.map(p => p.y)
    const xMin = Math.min(...xs)
    const xMax = Math.max(...xs)
    const yMin = Math.min(...ys)
    const yMax = Math.max(...ys)
    const xRange = xMax - xMin || 1
    const yRange = yMax - yMin || 1
    const pad = 16

    const toX = (x: number) => pad + ((x - xMin) / xRange) * (w - pad * 2)
    const toY = (y: number) => h - pad - ((y - yMin) / yRange) * (h - pad * 2)

    for (const pt of data) {
      const cx = toX(pt.x)
      const cy = toY(pt.y)
      ctx.beginPath()
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2)
      ctx.fillStyle = CLASS_COLORS[pt.label] ?? '#94a3b8'
      ctx.globalAlpha = 0.85
      ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = 'rgba(15,23,42,0.5)'
      ctx.lineWidth = 0.7
      ctx.stroke()
    }
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

  return (
    <div className="max-w-md space-y-6">
      <SectionHeading>Dataset</SectionHeading>

      <div className="space-y-4">
        <Select
          label="Dataset type"
          value={cfg.dataset}
          options={DATASET_OPTIONS}
          onChange={v => update('dataset', v as DatasetType)}
        />
        <Slider
          label="Noise (σ)"
          value={cfg.noise}
          min={0.02}
          max={0.4}
          step={0.02}
          onChange={v => update('noise', v)}
          format={v => v.toFixed(2)}
        />
        <Slider
          label="Samples (n)"
          value={cfg.n}
          min={20}
          max={120}
          step={10}
          onChange={v => update('n', v)}
        />
      </div>

      <Button label="Regenerate (new seed)" onClick={handleRegenerate} variant="primary" />

      <DataPreview data={data} />

      <div className="flex gap-3 flex-wrap">
        {CLASS_COLORS.map((c, k) => (
          <div key={k} className="flex items-center gap-1.5 text-xs text-slate-400">
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: c }} />
            Class {k}
          </div>
        ))}
      </div>

      <InfoBox>
        <p className="font-medium text-indigo-300 mb-1">{DATASET_OPTIONS.find(o => o.value === cfg.dataset)?.label}</p>
        <p className="text-slate-400">{DATASET_INFO[cfg.dataset]}</p>
      </InfoBox>

      <SectionHeading>Kernel–Dataset Presets</SectionHeading>
      <div className="space-y-2 text-xs text-slate-400">
        <div className="bg-slate-800 rounded p-2 border border-slate-700">
          <span className="text-slate-300 font-medium">Separable blobs + linear:</span>{' '}
          clean hard-margin or soft-margin boundary, very few SVs.
        </div>
        <div className="bg-slate-800 rounded p-2 border border-slate-700">
          <span className="text-slate-300 font-medium">Overlapping blobs + low C:</span>{' '}
          wider margin with more support vectors; some misclassifications accepted.
        </div>
        <div className="bg-slate-800 rounded p-2 border border-slate-700">
          <span className="text-slate-300 font-medium">Circles + RBF (γ ≈ 1):</span>{' '}
          circular boundary perfectly separates rings; linear will fail.
        </div>
        <div className="bg-slate-800 rounded p-2 border border-slate-700">
          <span className="text-slate-300 font-medium">Moons + RBF (γ ≈ 0.5):</span>{' '}
          curved boundary follows the two half-circles smoothly.
        </div>
      </div>

      <div className="text-xs text-slate-500 space-y-1">
        <div>Seed: <span className="metric text-slate-400">{cfg.seed}</span></div>
        <div>Points: <span className="metric text-slate-400">{data.length}</span></div>
      </div>
    </div>
  )
}
