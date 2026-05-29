import {
  useEffect,
  useRef,
  useState,
  useCallback,
} from 'react'
import { Heatmap } from '../../lib/heatmap'
import {
  Select,
  ButtonRow,
  Stat,
  SectionHeading,
} from '../../components/Controls'
import {
  KERNELS,
  SAMPLE_IMAGES,
  getCNNConfig,
  setCNNConfig,
  type KernelKey,
  type ImageKey,
} from './dataStore'

// ─── Pure convolution math ─────────────────────────────────────────────────────

function padImage(img: number[][], pad: number): number[][] {
  if (pad === 0) return img
  const rows = img.length + 2 * pad
  const cols = (img[0]?.length ?? 0) + 2 * pad
  const out: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0))
  for (let r = 0; r < img.length; r++) {
    for (let c = 0; c < (img[0]?.length ?? 0); c++) {
      out[r + pad][c + pad] = img[r][c]
    }
  }
  return out
}

function computeOutputSize(inputSize: number, kSize: number, stride: number, pad: number): number {
  return Math.floor((inputSize - kSize + 2 * pad) / stride) + 1
}

function convolveAt(padded: number[][], kernel: number[][], rStart: number, cStart: number): number {
  const k = kernel.length
  let sum = 0
  for (let m = 0; m < k; m++)
    for (let n = 0; n < k; n++)
      sum += padded[rStart + m][cStart + n] * kernel[m][n]
  return sum
}

// ─── Options ──────────────────────────────────────────────────────────────────

const KERNEL_OPTIONS = Object.entries(KERNELS).map(([k, v]) => ({ value: k, label: v.label }))
const IMAGE_OPTIONS  = Object.entries(SAMPLE_IMAGES).map(([k, v]) => ({ value: k, label: v.label }))
const STRIDE_OPTIONS  = [1,2,3].map(v => ({ value: String(v), label: String(v) }))
const PADDING_OPTIONS = [
  { value: 'valid', label: 'Valid (no padding)' },
  { value: 'same',  label: 'Same (preserve size)' },
]
const SPEED_OPTIONS = [
  { value: '20',  label: 'Very fast' },
  { value: '60',  label: 'Fast' },
  { value: '120', label: 'Normal' },
  { value: '300', label: 'Slow' },
  { value: '600', label: 'Very slow' },
]

// ─── Sweep state ──────────────────────────────────────────────────────────────

interface SweepState {
  outputRow: number
  outputCol: number
  outRows: number
  outCols: number
  convOutput: number[][] // NaN = not yet computed
  reluOutput: number[][]
  poolOutput: number[][]
  normLo: number
  normHi: number
  done: boolean
}

// ─── Component ────────────────────────────────────────────────────────────────

export function Visualization() {
  const cfg = getCNNConfig()

  const [imageKey, setImageKey]   = useState<ImageKey>(cfg.imageKey)
  const [kernelKey, setKernelKey] = useState<KernelKey>(cfg.kernelKey)
  const [stride, setStride]       = useState(cfg.stride)
  const [padding, setPadding]     = useState<'valid' | 'same'>(cfg.padding)
  const [speedMs, setSpeedMs]     = useState(cfg.speedMs)
  const [isPlaying, setIsPlaying] = useState(false)

  const [step, setStep]               = useState(0)
  const [totalSteps, setTotalSteps]   = useState(0)
  const [convVal, setConvVal]         = useState<number | null>(null)
  const [outSizeStr, setOutSizeStr]   = useState('')
  const [poolSizeStr, setPoolSizeStr] = useState('')

  // Five unique canvases: input, kernel, conv, relu, pool
  const inputCanvasRef  = useRef<HTMLCanvasElement>(null)
  const kernelCanvasRef = useRef<HTMLCanvasElement>(null)
  const convCanvasRef   = useRef<HTMLCanvasElement>(null)
  const reluCanvasRef   = useRef<HTMLCanvasElement>(null)
  const poolCanvasRef   = useRef<HTMLCanvasElement>(null)

  const inputHmRef  = useRef<Heatmap | null>(null)
  const kernelHmRef = useRef<Heatmap | null>(null)
  const convHmRef   = useRef<Heatmap | null>(null)
  const reluHmRef   = useRef<Heatmap | null>(null)
  const poolHmRef   = useRef<Heatmap | null>(null)

  const sweepRef     = useRef<SweepState | null>(null)
  const rafRef       = useRef<number | null>(null)
  const lastTimeRef  = useRef<number>(0)
  const isPlayingRef = useRef(false)
  const speedMsRef   = useRef(speedMs)
  const strideRef    = useRef(stride)

  useEffect(() => { speedMsRef.current = speedMs }, [speedMs])
  useEffect(() => { strideRef.current = stride }, [stride])

  // ─── Derived config ───────────────────────────────────────────────────────
  const getComputed = useCallback(() => {
    const image  = SAMPLE_IMAGES[imageKey].data
    const kernel = KERNELS[kernelKey].matrix
    const kSize  = kernel.length
    const inRows = image.length
    const inCols = image[0]?.length ?? 0
    const pad    = padding === 'same' ? Math.floor(kSize / 2) : 0
    const outRows = computeOutputSize(inRows, kSize, stride, pad)
    const outCols = computeOutputSize(inCols, kSize, stride, pad)
    const padded  = padImage(image, pad)
    return { image, kernel, kSize, inRows, inCols, pad, outRows, outCols, padded }
  }, [imageKey, kernelKey, stride, padding])

  // ─── Init heatmaps once ───────────────────────────────────────────────────
  useEffect(() => {
    if (inputCanvasRef.current)
      inputHmRef.current = new Heatmap(inputCanvasRef.current,  { colormap: 'grayscale', gridLines: false, padding: 0, normMin: 0, normMax: 255 })
    if (kernelCanvasRef.current)
      kernelHmRef.current = new Heatmap(kernelCanvasRef.current, { colormap: 'diverging', gridLines: true,  cellLabels: true, padding: 6 })
    if (convCanvasRef.current)
      convHmRef.current = new Heatmap(convCanvasRef.current,    { colormap: 'diverging', gridLines: false, padding: 0 })
    if (reluCanvasRef.current)
      reluHmRef.current = new Heatmap(reluCanvasRef.current,    { colormap: 'grayscale', gridLines: false, padding: 0, normMin: 0 })
    if (poolCanvasRef.current)
      poolHmRef.current = new Heatmap(poolCanvasRef.current,    { colormap: 'grayscale', gridLines: false, padding: 0, normMin: 0 })
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [])

  // ─── Render all panels ────────────────────────────────────────────────────
  const renderAll = useCallback((sweep: SweepState) => {
    const { image, kernel, kSize, inRows, inCols, pad } = getComputed()
    const s  = strideRef.current
    const lo = isFinite(sweep.normLo) ? sweep.normLo : -1
    const hi = isFinite(sweep.normHi) ? sweep.normHi :  1
    const absMax = Math.max(Math.abs(lo), Math.abs(hi), 0.01)
    const { outputRow: r, outputCol: c, outRows, outCols, convOutput, reluOutput, poolOutput } = sweep

    // Input + receptive field
    if (inputHmRef.current) {
      inputHmRef.current.resize()
      inputHmRef.current.render(image, { normMin: 0, normMax: 255 })
      if (!sweep.done && outRows > 0) {
        const rfR = r * s - pad, rfC = c * s - pad
        const vR = Math.max(0, rfR), vC = Math.max(0, rfC)
        const eR = Math.min(inRows, rfR + kSize), eC = Math.min(inCols, rfC + kSize)
        if (eR > vR && eC > vC)
          inputHmRef.current.highlightRect(vR, vC, eR - vR, eC - vC)
      }
    }

    // Kernel (diverging with cell labels)
    if (kernelHmRef.current) {
      kernelHmRef.current.resize()
      kernelHmRef.current.render(kernel)
    }

    // Conv output (diverging, symmetric ±absMax)
    if (convHmRef.current) {
      const disp = convOutput.map(row => row.map(v => isNaN(v) ? 0 : v))
      convHmRef.current.resize()
      convHmRef.current.render(disp, { normMin: -absMax, normMax: absMax })
      if (!sweep.done && r < outRows && c < outCols)
        convHmRef.current.highlightCell(r, c)
    }

    // ReLU
    if (reluHmRef.current) {
      const disp = reluOutput.map(row => row.map(v => isNaN(v) ? 0 : v))
      reluHmRef.current.resize()
      reluHmRef.current.render(disp, { normMin: 0, normMax: absMax })
    }

    // MaxPool
    if (poolHmRef.current) {
      if (poolOutput.length > 0 && (poolOutput[0]?.length ?? 0) > 0) {
        const disp = poolOutput.map(row => row.map(v => isNaN(v) ? 0 : v))
        poolHmRef.current.resize()
        poolHmRef.current.render(disp, { normMin: 0, normMax: absMax })
      } else {
        poolHmRef.current.resize()
        poolHmRef.current.clear()
      }
    }
  }, [getComputed])

  // ─── Build fresh sweep ────────────────────────────────────────────────────
  const buildSweep = useCallback((): SweepState => {
    const { outRows, outCols } = getComputed()
    const pR = Math.max(0, Math.floor(outRows / 2))
    const pC = Math.max(0, Math.floor(outCols / 2))
    return {
      outputRow: 0, outputCol: 0, outRows, outCols,
      convOutput: Array.from({ length: outRows }, () => new Array(outCols).fill(NaN)),
      reluOutput: Array.from({ length: outRows }, () => new Array(outCols).fill(NaN)),
      poolOutput: Array.from({ length: pR },       () => new Array(pC).fill(NaN)),
      normLo: Infinity, normHi: -Infinity,
      done: false,
    }
  }, [getComputed])

  // ─── Reset ────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    isPlayingRef.current = false
    setIsPlaying(false)
    const { outRows, outCols } = getComputed()
    const sweep = buildSweep()
    sweepRef.current = sweep
    setStep(0)
    setTotalSteps(outRows * outCols)
    setConvVal(null)
    setOutSizeStr(`${outRows}×${outCols}`)
    const pR = Math.floor(outRows / 2), pC = Math.floor(outCols / 2)
    setPoolSizeStr(pR > 0 && pC > 0 ? `${pR}×${pC}` : '—')
    renderAll(sweep)
  }, [getComputed, buildSweep, renderAll])

  // ─── Advance one step ────────────────────────────────────────────────────
  const stepOnce = useCallback((): boolean => {
    const sweep = sweepRef.current
    if (!sweep || sweep.done) return false
    const { kernel, padded } = getComputed()
    const s = strideRef.current
    const { outputRow: r, outputCol: c, outCols } = sweep

    const val = convolveAt(padded, kernel, r * s, c * s)
    sweep.convOutput[r][c] = val
    sweep.reluOutput[r][c] = Math.max(0, val)
    if (val < sweep.normLo) sweep.normLo = val
    if (val > sweep.normHi) sweep.normHi = val
    setConvVal(val)
    setStep(r * outCols + c + 1)

    // Update pool block if all 4 cells ready
    const bR = Math.floor(r / 2), bC = Math.floor(c / 2)
    if (bR < sweep.poolOutput.length && bC < (sweep.poolOutput[0]?.length ?? 0)) {
      let maxVal = -Infinity, complete = true
      outer: for (let dr = 0; dr < 2; dr++) {
        for (let dc = 0; dc < 2; dc++) {
          const pr = bR * 2 + dr, pc = bC * 2 + dc
          if (pr < sweep.outRows && pc < sweep.outCols) {
            const v = sweep.convOutput[pr][pc]
            if (isNaN(v)) { complete = false; break outer }
            const rv = Math.max(0, v)
            if (rv > maxVal) maxVal = rv
          }
        }
      }
      if (complete && isFinite(maxVal)) sweep.poolOutput[bR][bC] = maxVal
    }

    // Render while outputRow/outputCol still equal (r,c) so the just-filled
    // cell and its receptive field are highlighted.
    renderAll(sweep)

    let nr = r, nc = c + 1
    if (nc >= sweep.outCols) { nc = 0; nr++ }
    if (nr >= sweep.outRows) {
      // Sweep finished — leave the last computed cell highlighted.
      sweep.done = true
      return false
    }
    // Advance pointer for the next stepOnce call (no re-render here).
    sweep.outputRow = nr
    sweep.outputCol = nc
    return true
  }, [getComputed, renderAll])

  // ─── RAF loop ────────────────────────────────────────────────────────────
  const animLoop = useCallback((ts: number) => {
    if (!isPlayingRef.current) return
    if (ts - lastTimeRef.current >= speedMsRef.current) {
      lastTimeRef.current = ts
      if (!stepOnce()) { isPlayingRef.current = false; setIsPlaying(false); return }
    }
    rafRef.current = requestAnimationFrame(animLoop)
  }, [stepOnce])

  const handlePlay = () => {
    if (sweepRef.current?.done) reset()
    isPlayingRef.current = true; setIsPlaying(true)
    lastTimeRef.current = performance.now()
    rafRef.current = requestAnimationFrame(animLoop)
  }
  const handlePause = () => {
    isPlayingRef.current = false; setIsPlaying(false)
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
  }
  const handleStep = () => { handlePause(); if (sweepRef.current?.done) reset(); else stepOnce() }
  const handleReset = () => reset()

  // Sync config + reset on control changes
  useEffect(() => {
    setCNNConfig({ imageKey, kernelKey, stride, padding, speedMs })
    reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageKey, kernelKey, stride, padding])

  const inputRows = SAMPLE_IMAGES[imageKey].data.length
  const inputCols = SAMPLE_IMAGES[imageKey].data[0]?.length ?? 0

  // ─── JSX ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-4">

      {/* Controls */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 shrink-0">
        <Select label="Image"   value={imageKey}        options={IMAGE_OPTIONS}   onChange={v => setImageKey(v as ImageKey)}      disabled={isPlaying} />
        <Select label="Kernel"  value={kernelKey}       options={KERNEL_OPTIONS}  onChange={v => setKernelKey(v as KernelKey)}    disabled={isPlaying} />
        <Select label="Stride"  value={String(stride)}  options={STRIDE_OPTIONS}  onChange={v => setStride(Number(v))}             disabled={isPlaying} />
        <Select label="Padding" value={padding}         options={PADDING_OPTIONS} onChange={v => setPadding(v as 'valid'|'same')}  disabled={isPlaying} />
        <Select label="Speed"   value={String(speedMs)} options={SPEED_OPTIONS}   onChange={v => setSpeedMs(Number(v))} />
        <div className="flex flex-col justify-end">
          <ButtonRow buttons={[
            isPlaying
              ? { label:'Pause', onClick:handlePause, variant:'danger'   as const }
              : { label:'Play',  onClick:handlePlay,  variant:'primary'  as const },
            { label:'Step',  onClick:handleStep,  disabled:isPlaying },
            { label:'Reset', onClick:handleReset, disabled:isPlaying },
          ]} />
        </div>
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-3 shrink-0">
        <Stat label="Step"      value={`${step}/${totalSteps}`}                                 color="text-slate-300" />
        <Stat label="Conv val"  value={convVal !== null ? convVal.toFixed(1) : '—'}             color="text-amber-300" />
        <Stat label="Out size"  value={outSizeStr || `${inputRows}×…`}                          color="text-indigo-300" />
        <Stat label="Pool size" value={poolSizeStr || '—'}                                       color="text-cyan-300" />
      </div>

      {/* Main three-panel sweep */}
      <div className="grid grid-cols-3 gap-3">
        {/* Input */}
        <div className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 uppercase tracking-wider text-center">
            Input ({inputRows}×{inputCols})
          </span>
          <div className="rounded border border-slate-600 overflow-hidden bg-slate-950" style={{ aspectRatio: '1/1' }}>
            <canvas ref={inputCanvasRef} className="w-full h-full block" style={{ imageRendering:'pixelated' }} />
          </div>
          <p className="text-[10px] text-slate-600 text-center">Receptive field — indigo border</p>
        </div>

        {/* Kernel */}
        <div className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 uppercase tracking-wider text-center">
            Kernel 3×3
          </span>
          <div className="rounded border border-slate-600 overflow-hidden bg-slate-950" style={{ aspectRatio: '1/1' }}>
            <canvas ref={kernelCanvasRef} className="w-full h-full block" />
          </div>
          <p className="text-[10px] text-slate-600 text-center">Blue=neg · White=0 · Red=pos</p>
        </div>

        {/* Conv output */}
        <div className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 uppercase tracking-wider text-center">
            Conv Output ({outSizeStr})
          </span>
          <div className="rounded border border-slate-600 overflow-hidden bg-slate-950" style={{ aspectRatio: '1/1' }}>
            <canvas ref={convCanvasRef} className="w-full h-full block" style={{ imageRendering:'pixelated' }} />
          </div>
          <p className="text-[10px] text-slate-600 text-center">Active cell — cyan border</p>
        </div>
      </div>

      {/* Pipeline: ReLU + MaxPool (conv already shown above) */}
      <div>
        <SectionHeading>Pipeline: Conv → ReLU → 2×2 MaxPool</SectionHeading>
        <div className="grid grid-cols-3 gap-3 items-center">
          {/* Conv result note */}
          <div className="flex flex-col items-center justify-center h-full gap-2 py-4 rounded border border-slate-700/50 bg-slate-900/50">
            <span className="text-3xl select-none">↑</span>
            <span className="text-xs text-slate-500 text-center px-2">
              Conv output shown above (diverging colormap — signed values)
            </span>
          </div>

          {/* ReLU */}
          <div className="flex flex-col gap-1">
            <span className="text-xs text-slate-500 uppercase tracking-wider text-center">
              After ReLU
            </span>
            <div className="rounded border border-slate-600 overflow-hidden bg-slate-950" style={{ aspectRatio: '1/1' }}>
              <canvas ref={reluCanvasRef} className="w-full h-full block" style={{ imageRendering:'pixelated' }} />
            </div>
            <p className="text-[10px] text-slate-600 text-center">Negatives zeroed</p>
          </div>

          {/* MaxPool */}
          <div className="flex flex-col gap-1">
            <span className="text-xs text-slate-500 uppercase tracking-wider text-center">
              2×2 MaxPool ({poolSizeStr})
            </span>
            <div className="rounded border border-slate-600 overflow-hidden bg-slate-950" style={{ aspectRatio: '1/1' }}>
              <canvas ref={poolCanvasRef} className="w-full h-full block" style={{ imageRendering:'pixelated' }} />
            </div>
            <p className="text-[10px] text-slate-600 text-center">Halved spatial size</p>
          </div>
        </div>
      </div>

      {/* Step detail */}
      {convVal !== null && (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-md px-4 py-2 text-sm text-slate-400">
          Last computed:&nbsp;
          <span className="metric text-amber-300">{convVal.toFixed(3)}</span>
          &nbsp;→ ReLU:&nbsp;
          <span className="metric text-green-300">{Math.max(0, convVal).toFixed(3)}</span>
          <span className="text-xs text-slate-600 ml-3">(kernel ⊙ input patch dot product)</span>
        </div>
      )}
    </div>
  )
}
