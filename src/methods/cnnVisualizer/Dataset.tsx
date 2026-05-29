import { useEffect, useRef } from 'react'
import { SectionHeading, Select, InfoBox } from '../../components/Controls'
import { Heatmap } from '../../lib/heatmap'
import {
  SAMPLE_IMAGES,
  getCNNConfig,
  setCNNConfig,
  type ImageKey,
} from './dataStore'
import type { DatasetProps } from '../registry'
import { useState } from 'react'

const IMAGE_OPTIONS = Object.entries(SAMPLE_IMAGES).map(([k, v]) => ({
  value: k,
  label: v.label,
}))

export function Dataset({ onDataChange }: DatasetProps) {
  const [imageKey, setImageKey] = useState<ImageKey>(getCNNConfig().imageKey)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const hmRef = useRef<Heatmap | null>(null)

  // Draw heatmap whenever image changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (!hmRef.current) {
      hmRef.current = new Heatmap(canvas, {
        colormap: 'grayscale',
        gridLines: false,
        cellLabels: false,
        padding: 0,
        normMin: 0,
        normMax: 255,
      })
    }
    hmRef.current.resize()
    hmRef.current.render(SAMPLE_IMAGES[imageKey].data, { normMin: 0, normMax: 255 })
  }, [imageKey])

  function handleSelect(v: string) {
    const key = v as ImageKey
    setImageKey(key)
    setCNNConfig({ imageKey: key })
    onDataChange?.()
  }

  const img = SAMPLE_IMAGES[imageKey]
  const rows = img.data.length
  const cols = img.data[0]?.length ?? 0

  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Input Image</SectionHeading>

      <div className="max-w-xs">
        <Select
          label="Sample image"
          value={imageKey}
          options={IMAGE_OPTIONS}
          onChange={handleSelect}
        />
      </div>

      {/* Preview */}
      <div className="flex gap-6 items-start">
        <div className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 uppercase tracking-wider">Preview ({rows}×{cols})</span>
          <div
            className="border border-slate-600 rounded overflow-hidden"
            style={{ width: 160, height: 160 }}
          >
            <canvas
              ref={canvasRef}
              className="w-full h-full block"
              style={{ imageRendering: 'pixelated' }}
            />
          </div>
        </div>

        <div className="flex-1 text-sm text-slate-400 space-y-2">
          <p>
            <strong className="text-slate-300">{img.label}</strong> — a{' '}
            {rows}×{cols} grayscale image with pixel values in [0, 255].
          </p>
          <p>
            {imageKey === 'diagonal' && (
              <>The bright diagonal band clearly responds to Sobel-X (which detects edges tilted in the opposite direction) and also to Sobel-Y. The 45° structure makes it a great test for directional edge kernels.</>
            )}
            {imageKey === 'cross' && (
              <>The cross has both a horizontal bar and a vertical bar. Sobel-X lights up the vertical edges of the horizontal bar; Sobel-Y lights up the horizontal edges of the vertical bar — a clear demonstration of directional sensitivity.</>
            )}
            {imageKey === 'digit7' && (
              <>A digit-like "7" with diagonal strokes. Edge kernels reveal the stroke structure. Compare blur (smooths strokes) vs. sharpen (amplifies fine detail) vs. outline (finds the stroke boundary).</>
            )}
            {imageKey === 'checkerboard' && (
              <>High-frequency alternating pattern. Blur kernels dramatically smooth it; outline and Sobel kernels produce strong uniform responses everywhere. Useful for seeing frequency-dependent behavior.</>
            )}
          </p>
        </div>
      </div>

      <InfoBox>
        <strong className="text-indigo-300">How to use:</strong> Select an image here,
        then go to <em>Visualization</em> to watch the kernel sweep in real time. Each
        image is designed to produce a visibly different feature map depending on the kernel —
        especially the difference between <span className="metric">Sobel-X</span> and{' '}
        <span className="metric">Sobel-Y</span>.
      </InfoBox>

      <SectionHeading>Pixel Values</SectionHeading>
      <div className="text-sm text-slate-400">
        <p>
          All images are <strong className="text-slate-300">grayscale</strong> with 8-bit values
          (0 = black, 255 = white). The heatmap uses a grayscale colormap for the input, and a{' '}
          <strong className="text-slate-300">diverging blue→white→red</strong> colormap for signed
          convolution outputs (e.g. Sobel, where negative values indicate one edge polarity and
          positive values the other).
        </p>
      </div>
    </div>
  )
}
