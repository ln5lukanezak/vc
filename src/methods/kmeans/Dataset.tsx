import { useState, useEffect, useRef } from 'react'
import { Slider, Button, SectionHeading, InfoBox } from '../../components/Controls'
import { getConfig, setConfig, regenerate, subscribe, getData } from './dataStore'
import type { KMeansPoint } from './kmeans'

// Soft cluster colours — we don't know assignments in the preview, so all grey
const PREVIEW_COLOR = '#818cf8'

function DataPreview({ data }: { data: KMeansPoint[] }) {
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
      ctx.fillStyle = PREVIEW_COLOR
      ctx.globalAlpha = 0.75
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

export function Dataset() {
  const [cfg, setCfg] = useState(getConfig)
  const [data, setData] = useState<KMeansPoint[]>(getData)

  useEffect(() => {
    return subscribe(() => {
      setCfg(getConfig())
      setData(getData())
    })
  }, [])

  function update<K extends keyof typeof cfg>(key: K, value: (typeof cfg)[K]) {
    const next = { ...cfg, [key]: value }
    setCfg(next)
    setConfig({ [key]: value })
    regenerate()
  }

  function handleRegenerate() {
    const newSeed = Math.floor(Math.random() * 100000)
    const next = { ...cfg, seed: newSeed }
    setCfg(next)
    setConfig(next)
    regenerate()
  }

  return (
    <div className="max-w-md space-y-6">
      <SectionHeading>Dataset Generation</SectionHeading>

      <div className="space-y-4">
        <Slider
          label="True cluster count"
          value={cfg.clusters}
          min={2}
          max={6}
          step={1}
          onChange={v => update('clusters', v)}
        />
        <Slider
          label="Separation (radius)"
          value={cfg.separation}
          min={0.4}
          max={3.0}
          step={0.1}
          onChange={v => update('separation', v)}
          format={v => v.toFixed(1)}
        />
        <Slider
          label="Noise (σ)"
          value={cfg.noise}
          min={0.05}
          max={0.5}
          step={0.05}
          onChange={v => update('noise', v)}
          format={v => v.toFixed(2)}
        />
        <Slider
          label="Samples (n)"
          value={cfg.n}
          min={30}
          max={300}
          step={10}
          onChange={v => update('n', v)}
        />
      </div>

      <Button label="Regenerate (new seed)" onClick={handleRegenerate} variant="primary" />

      <DataPreview data={data} />

      <InfoBox>
        <p className="font-medium text-indigo-300 mb-1">Unsupervised — labels are ignored</p>
        <p className="text-slate-400">
          The dataset is generated with{' '}
          <span className="metric text-cyan-300">{cfg.clusters}</span> true Gaussian blobs,
          but the cluster labels are <em>discarded</em>. The K-Means algorithm sees only
          coordinates (x, y) and must discover the groupings from scratch. Set{' '}
          <span className="metric text-cyan-300">k</span> in the Visualization tab to match
          or differ from the true count to see what happens.
        </p>
      </InfoBox>

      <div className="text-xs text-slate-500 space-y-1">
        <div>Seed: <span className="metric text-slate-400">{cfg.seed}</span></div>
        <div>Points: <span className="metric text-slate-400">{data.length}</span></div>
      </div>
    </div>
  )
}
