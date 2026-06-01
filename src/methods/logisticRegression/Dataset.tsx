import { useState, useEffect, useRef } from 'react'
import { Slider, Button, SectionHeading, InfoBox } from '../../components/Controls'
import {
  getDataConfig,
  setDataConfig,
  regenerate,
  subscribe,
  getData,
} from './dataStore'
import type { DatasetProps } from '../registry'
import type { Point2D } from '../../lib/datagen'

const CLASS_COLORS = ['#818cf8', '#34d399', '#fb923c', '#f472b6']

function DataPreview({ data, classes }: { data: Point2D[]; classes: number }) {
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
    const pad = 20
    const xRange = xMax - xMin || 1
    const yRange = yMax - yMin || 1

    const toCanvasX = (x: number) => pad + ((x - xMin) / xRange) * (w - pad * 2)
    const toCanvasY = (y: number) => h - pad - ((y - yMin) / yRange) * (h - pad * 2)

    for (const pt of data) {
      const cx = toCanvasX(pt.x)
      const cy = toCanvasY(pt.y)
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
    <div>
      <canvas
        ref={canvasRef}
        className="w-full rounded-md border border-slate-700 bg-slate-950"
        style={{ height: 200 }}
      />
      <div className="flex gap-3 mt-2 flex-wrap">
        {Array.from({ length: classes }, (_, k) => (
          <div key={k} className="flex items-center gap-1.5 text-xs text-slate-400">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ background: CLASS_COLORS[k] }}
            />
            Class {k}
          </div>
        ))}
      </div>
    </div>
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
      <SectionHeading>Blob Generator</SectionHeading>

      <div className="space-y-4">
        <Slider
          label="Classes (K)"
          value={cfg.classes}
          min={2}
          max={4}
          step={1}
          onChange={v => update('classes', v)}
        />
        <Slider
          label="Separation"
          value={cfg.separation}
          min={0.3}
          max={3.0}
          step={0.1}
          onChange={v => update('separation', v)}
          format={v => v.toFixed(1)}
        />
        <Slider
          label="Noise (σ)"
          value={cfg.noise}
          min={0.05}
          max={0.6}
          step={0.05}
          onChange={v => update('noise', v)}
          format={v => v.toFixed(2)}
        />
        <Slider
          label="Samples (n)"
          value={cfg.n}
          min={50}
          max={500}
          step={50}
          onChange={v => update('n', v)}
        />
      </div>

      <Button label="Regenerate (new seed)" onClick={handleRegenerate} variant="primary" />

      <DataPreview data={data} classes={cfg.classes} />

      <InfoBox>
        <p className="font-medium text-indigo-300 mb-1">How the blobs are generated</p>
        <p className="text-slate-400">
          K cluster centres are placed evenly around a circle of radius{' '}
          <strong className="text-slate-300">separation</strong>. Each point is drawn by
          sampling a Gaussian with standard deviation{' '}
          <strong className="text-slate-300">noise</strong> around its cluster centre.
          The clusters are isotropic (same spread in all directions). A high separation
          with low noise gives well-separated, easily classifiable blobs; a low
          separation with high noise creates heavy overlap that challenges any linear
          classifier.
        </p>
      </InfoBox>

      <div className="text-xs text-slate-500 space-y-1">
        <div>Seed: <span className="metric text-slate-400">{cfg.seed}</span></div>
        <div>Points: <span className="metric text-slate-400">{data.length}</span></div>
        <div>Classes: <span className="metric text-slate-400">{cfg.classes}</span></div>
      </div>
    </div>
  )
}
