// ─── LossChart: canvas line chart for a loss series over epochs ───────────────

export interface LossChartOptions {
  color?: string
  bg?: string
  label?: string
  maxPoints?: number
}

export class LossChart {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private dpr: number
  private losses: number[] = []
  private opts: Required<LossChartOptions>

  constructor(canvas: HTMLCanvasElement, opts: LossChartOptions = {}) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Could not get 2D context')
    this.ctx = ctx
    this.dpr = window.devicePixelRatio || 1
    this.opts = {
      color: opts.color ?? '#818cf8',
      bg: opts.bg ?? '#0f172a',
      label: opts.label ?? 'Loss',
      maxPoints: opts.maxPoints ?? 500,
    }
    this.resize()
  }

  resize(): void {
    const { canvas, dpr } = this
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    if (
      canvas.width !== Math.round(w * dpr) ||
      canvas.height !== Math.round(h * dpr)
    ) {
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      this.ctx.scale(dpr, dpr)
    }
  }

  push(loss: number): void {
    this.losses.push(loss)
    if (this.losses.length > this.opts.maxPoints) {
      // Downsample: keep every other
      const next: number[] = []
      for (let i = 0; i < this.losses.length; i += 2) next.push(this.losses[i])
      this.losses = next
    }
  }

  reset(): void {
    this.losses = []
  }

  draw(): void {
    const { ctx, canvas, opts } = this
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    const pad = { top: 14, right: 10, bottom: 24, left: 46 }

    // Background
    ctx.fillStyle = opts.bg
    ctx.fillRect(0, 0, w, h)

    if (this.losses.length < 2) {
      ctx.fillStyle = '#475569'
      ctx.font = '10px JetBrains Mono, monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('No data yet', w / 2, h / 2)
      return
    }

    const data = this.losses
    const yMax = Math.max(...data) * 1.05
    const yMin = Math.min(...data) * 0.95

    const plotW = w - pad.left - pad.right
    const plotH = h - pad.top - pad.bottom

    const toX = (i: number) => pad.left + (i / (data.length - 1)) * plotW
    const toY = (v: number) =>
      pad.top + (1 - (v - yMin) / (yMax - yMin + 1e-12)) * plotH

    // Grid
    ctx.strokeStyle = 'rgba(148,163,184,0.08)'
    ctx.lineWidth = 1
    const nLines = 3
    for (let i = 0; i <= nLines; i++) {
      const y = pad.top + (i / nLines) * plotH
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(pad.left + plotW, y)
      ctx.stroke()
    }

    // Y labels
    ctx.fillStyle = '#64748b'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let i = 0; i <= nLines; i++) {
      const v = yMax - (i / nLines) * (yMax - yMin)
      const y = pad.top + (i / nLines) * plotH
      ctx.fillText(fmtLoss(v), pad.left - 4, y)
    }

    // X labels (epoch count)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText('0', pad.left, pad.top + plotH + 4)
    ctx.fillText(String(data.length - 1), pad.left + plotW, pad.top + plotH + 4)

    // Loss line with gradient fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH)
    grad.addColorStop(0, opts.color + '55')
    grad.addColorStop(1, opts.color + '00')

    ctx.beginPath()
    ctx.moveTo(toX(0), toY(data[0]))
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(toX(i), toY(data[i]))
    }
    ctx.lineTo(toX(data.length - 1), pad.top + plotH)
    ctx.lineTo(toX(0), pad.top + plotH)
    ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()

    ctx.beginPath()
    ctx.moveTo(toX(0), toY(data[0]))
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(toX(i), toY(data[i]))
    }
    ctx.strokeStyle = opts.color
    ctx.lineWidth = 1.5
    ctx.lineJoin = 'round'
    ctx.stroke()

    // Label
    ctx.fillStyle = opts.color
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    ctx.fillText(opts.label, pad.left + 4, pad.top + 2)

    // Border
    ctx.strokeStyle = 'rgba(148,163,184,0.2)'
    ctx.lineWidth = 1
    ctx.strokeRect(pad.left, pad.top, plotW, plotH)
  }
}

function fmtLoss(v: number): string {
  if (v === 0) return '0'
  if (Math.abs(v) < 0.001 || Math.abs(v) >= 10000) return v.toExponential(1)
  return v.toPrecision(3)
}
