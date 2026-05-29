// ─── Plot: canvas 2D wrapper for ML visualizations ───────────────────────────

export interface PlotBounds {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export interface ScatterOptions {
  color?: string
  r?: number
  alpha?: number
}

export interface LineOptions {
  color?: string
  width?: number
  dash?: number[]
  alpha?: number
}

export interface TextOptions {
  color?: string
  font?: string
  align?: CanvasTextAlign
  baseline?: CanvasTextBaseline
}

export class Plot {
  private ctx: CanvasRenderingContext2D
  private canvas: HTMLCanvasElement
  private dpr: number
  private bounds: PlotBounds = { xMin: -1, xMax: 1, yMin: -1, yMax: 1 }
  // Padding in CSS pixels
  private pad = { top: 20, right: 16, bottom: 36, left: 48 }

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Could not get 2D context')
    this.ctx = ctx
    this.dpr = window.devicePixelRatio || 1
    this.resize()
  }

  // ─── Resize to fill display size ─────────────────────────────────────────
  resize(): void {
    const { canvas, dpr } = this
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      this.ctx.scale(dpr, dpr)
    }
  }

  setBounds(bounds: PlotBounds): void {
    this.bounds = bounds
  }

  setBoundsFromData(xs: number[], ys: number[], padFraction = 0.1): void {
    if (xs.length === 0) return
    let xMin = Math.min(...xs), xMax = Math.max(...xs)
    let yMin = Math.min(...ys), yMax = Math.max(...ys)
    const xPad = (xMax - xMin) * padFraction || 1
    const yPad = (yMax - yMin) * padFraction || 1
    this.bounds = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad }
  }

  // ─── Coordinate transforms ────────────────────────────────────────────────
  private cssW(): number { return this.canvas.clientWidth }
  private cssH(): number { return this.canvas.clientHeight }

  toPixelX(x: number): number {
    const { xMin, xMax } = this.bounds
    const plotW = this.cssW() - this.pad.left - this.pad.right
    return this.pad.left + ((x - xMin) / (xMax - xMin)) * plotW
  }

  toPixelY(y: number): number {
    const { yMin, yMax } = this.bounds
    const plotH = this.cssH() - this.pad.top - this.pad.bottom
    return this.pad.top + (1 - (y - yMin) / (yMax - yMin)) * plotH
  }

  toPixel(x: number, y: number): [number, number] {
    return [this.toPixelX(x), this.toPixelY(y)]
  }

  toDataX(px: number): number {
    const { xMin, xMax } = this.bounds
    const plotW = this.cssW() - this.pad.left - this.pad.right
    return xMin + ((px - this.pad.left) / plotW) * (xMax - xMin)
  }

  toDataY(py: number): number {
    const { yMin, yMax } = this.bounds
    const plotH = this.cssH() - this.pad.top - this.pad.bottom
    return yMin + (1 - (py - this.pad.top) / plotH) * (yMax - yMin)
  }

  // ─── Clear ────────────────────────────────────────────────────────────────
  clear(bg = '#0f172a'): void {
    const { ctx } = this
    const w = this.cssW()
    const h = this.cssH()
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, w, h)
  }

  // ─── Axes and grid ────────────────────────────────────────────────────────
  drawAxesGrid(opts: { gridColor?: string; axisColor?: string; labelColor?: string } = {}): void {
    const {
      gridColor = 'rgba(148,163,184,0.1)',
      axisColor = 'rgba(148,163,184,0.4)',
      labelColor = '#64748b',
    } = opts
    const { ctx, bounds, pad } = this
    const w = this.cssW()
    const h = this.cssH()
    const plotW = w - pad.left - pad.right
    const plotH = h - pad.top - pad.bottom

    ctx.save()
    ctx.font = '10px JetBrains Mono, Consolas, monospace'
    ctx.fillStyle = labelColor
    ctx.strokeStyle = gridColor
    ctx.lineWidth = 1

    // Vertical grid lines (x axis ticks)
    const xTicks = niceTickMarks(bounds.xMin, bounds.xMax, 6)
    for (const xv of xTicks) {
      const px = this.toPixelX(xv)
      if (px < pad.left || px > pad.left + plotW) continue
      ctx.beginPath()
      ctx.moveTo(px, pad.top)
      ctx.lineTo(px, pad.top + plotH)
      ctx.stroke()
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText(fmtTick(xv), px, pad.top + plotH + 4)
    }

    // Horizontal grid lines (y axis ticks)
    const yTicks = niceTickMarks(bounds.yMin, bounds.yMax, 5)
    for (const yv of yTicks) {
      const py = this.toPixelY(yv)
      if (py < pad.top || py > pad.top + plotH) continue
      ctx.beginPath()
      ctx.moveTo(pad.left, py)
      ctx.lineTo(pad.left + plotW, py)
      ctx.stroke()
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      ctx.fillText(fmtTick(yv), pad.left - 6, py)
    }

    // Axis lines
    ctx.strokeStyle = axisColor
    ctx.lineWidth = 1.5
    // x-axis
    if (bounds.yMin <= 0 && bounds.yMax >= 0) {
      const py = this.toPixelY(0)
      ctx.beginPath()
      ctx.moveTo(pad.left, py)
      ctx.lineTo(pad.left + plotW, py)
      ctx.stroke()
    }
    // y-axis
    if (bounds.xMin <= 0 && bounds.xMax >= 0) {
      const px = this.toPixelX(0)
      ctx.beginPath()
      ctx.moveTo(px, pad.top)
      ctx.lineTo(px, pad.top + plotH)
      ctx.stroke()
    }

    // Plot border
    ctx.strokeStyle = 'rgba(148,163,184,0.25)'
    ctx.lineWidth = 1
    ctx.strokeRect(pad.left, pad.top, plotW, plotH)

    ctx.restore()
  }

  // ─── Scatter ──────────────────────────────────────────────────────────────
  scatter(
    points: Array<{ x: number; y: number }>,
    opts: ScatterOptions = {},
  ): void {
    const { ctx } = this
    const { color = '#818cf8', r = 4, alpha = 0.85 } = opts
    ctx.save()
    ctx.globalAlpha = alpha
    ctx.fillStyle = color
    for (const p of points) {
      const [px, py] = this.toPixel(p.x, p.y)
      ctx.beginPath()
      ctx.arc(px, py, r, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.restore()
  }

  // ─── Line (array of points or fn) ────────────────────────────────────────
  line(
    pointsOrFn: Array<{ x: number; y: number }> | ((x: number) => number),
    opts: LineOptions = {},
  ): void {
    const { ctx, bounds } = this
    const { color = '#06b6d4', width = 2, dash = [], alpha = 1 } = opts

    let points: Array<{ x: number; y: number }>
    if (typeof pointsOrFn === 'function') {
      const fn = pointsOrFn
      const steps = 300
      points = []
      for (let i = 0; i <= steps; i++) {
        const x = bounds.xMin + (i / steps) * (bounds.xMax - bounds.xMin)
        const y = fn(x)
        if (isFinite(y)) points.push({ x, y })
      }
    } else {
      points = pointsOrFn
    }

    if (points.length < 2) return

    ctx.save()
    ctx.globalAlpha = alpha
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    if (dash.length) ctx.setLineDash(dash)

    ctx.beginPath()
    let started = false
    for (const p of points) {
      const [px, py] = this.toPixel(p.x, p.y)
      if (!started) {
        ctx.moveTo(px, py)
        started = true
      } else {
        ctx.lineTo(px, py)
      }
    }
    ctx.stroke()
    ctx.restore()
  }

  // ─── Text ─────────────────────────────────────────────────────────────────
  text(
    label: string,
    x: number,
    y: number,
    opts: TextOptions = {},
  ): void {
    const {
      color = '#94a3b8',
      font = '11px JetBrains Mono, monospace',
      align = 'left',
      baseline = 'middle',
    } = opts
    const { ctx } = this
    const [px, py] = this.toPixel(x, y)
    ctx.save()
    ctx.fillStyle = color
    ctx.font = font
    ctx.textAlign = align
    ctx.textBaseline = baseline
    ctx.fillText(label, px, py)
    ctx.restore()
  }

  /** Draw text at pixel coordinates (bypasses data transform) */
  textPx(label: string, px: number, py: number, opts: TextOptions = {}): void {
    const {
      color = '#94a3b8',
      font = '11px JetBrains Mono, monospace',
      align = 'left',
      baseline = 'middle',
    } = opts
    const { ctx } = this
    ctx.save()
    ctx.fillStyle = color
    ctx.font = font
    ctx.textAlign = align
    ctx.textBaseline = baseline
    ctx.fillText(label, px, py)
    ctx.restore()
  }

  getCtx(): CanvasRenderingContext2D { return this.ctx }
  getBounds(): PlotBounds { return { ...this.bounds } }
  getPad() { return { ...this.pad } }
  getCssSize(): { w: number; h: number } { return { w: this.cssW(), h: this.cssH() } }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function niceTickMarks(min: number, max: number, count: number): number[] {
  const range = max - min
  if (range === 0) return [min]
  const rawStep = range / count
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const normalized = rawStep / mag
  let step: number
  if (normalized <= 1.5) step = 1 * mag
  else if (normalized <= 3.5) step = 2 * mag
  else if (normalized <= 7.5) step = 5 * mag
  else step = 10 * mag

  const start = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let v = start; v <= max + step * 0.01; v += step) {
    ticks.push(parseFloat(v.toPrecision(10)))
  }
  return ticks
}

function fmtTick(v: number): string {
  if (Math.abs(v) >= 10000 || (Math.abs(v) < 0.01 && v !== 0)) {
    return v.toExponential(1)
  }
  // Remove trailing zeros
  return parseFloat(v.toPrecision(4)).toString()
}
