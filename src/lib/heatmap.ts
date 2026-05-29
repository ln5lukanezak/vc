/**
 * heatmap.ts — Reusable 2-D matrix renderer for CNN, MLP, and NLP visualizations.
 *
 * Public API:
 *   new Heatmap(canvas, options?)   — attach to a <canvas> element
 *   heatmap.render(matrix, opts?)   — draw the full matrix
 *   heatmap.highlightRect(r,c,h,w)  — draw a receptive-field border
 *   heatmap.highlightCell(r,c)      — pulse-highlight a single output cell
 *   heatmap.resize()                — call when canvas CSS size changes
 *   heatmap.clear()                 — fill with background color
 *
 * Colormaps: 'grayscale' (0=black,1=white) | 'diverging' (neg=blue,0=white,pos=red)
 */

export type ColorMap = 'grayscale' | 'diverging'

export interface HeatmapOptions {
  /** background fill color (default #0f172a) */
  bg?: string
  /** draw thin lines between cells (default false) */
  gridLines?: boolean
  /** show numeric value inside each cell; works best for small matrices (default false) */
  cellLabels?: boolean
  /** font size for cell labels in px; 0 = auto (default 0) */
  labelFontSize?: number
  /** colormap (default 'grayscale') */
  colormap?: ColorMap
  /** if provided, normalize values to [min,max]; otherwise auto-range per render() call */
  normMin?: number
  normMax?: number
  /** padding around the grid in CSS px (default 2) */
  padding?: number
}

export interface RenderOptions {
  /** override instance colormap for this render only */
  colormap?: ColorMap
  /** override normalization min */
  normMin?: number
  /** override normalization max */
  normMax?: number
}

// ─── Color helpers ────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function clamp(v: number, lo = 0, hi = 1): number {
  return Math.max(lo, Math.min(hi, v))
}

/** grayscale: 0 → #000, 1 → #fff */
function grayscaleColor(t: number): [number, number, number] {
  const v = Math.round(clamp(t) * 255)
  return [v, v, v]
}

/**
 * diverging: -1 → blue(59,130,246), 0 → white, +1 → red(239,68,68)
 * (Tailwind blue-500 / red-500 palette)
 */
function divergingColor(t: number): [number, number, number] {
  const c = clamp(t, -1, 1)
  if (c < 0) {
    const s = -c // 0..1 toward blue
    return [
      Math.round(lerp(255, 59, s)),
      Math.round(lerp(255, 130, s)),
      Math.round(lerp(255, 246, s)),
    ]
  } else {
    return [
      Math.round(lerp(255, 239, c)),
      Math.round(lerp(255, 68, c)),
      Math.round(lerp(255, 68, c)),
    ]
  }
}

function toRgbString(r: number, g: number, b: number, a = 1): string {
  return `rgba(${r},${g},${b},${a})`
}

// ─── Heatmap class ────────────────────────────────────────────────────────────

export class Heatmap {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private dpr: number
  private opts: Required<HeatmapOptions>

  // cached layout from last render
  private _cellW = 0
  private _cellH = 0
  private _offsetX = 0
  private _offsetY = 0

  constructor(canvas: HTMLCanvasElement, options: HeatmapOptions = {}) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Heatmap: could not get 2D context')
    this.ctx = ctx
    this.dpr = window.devicePixelRatio || 1

    this.opts = {
      bg: options.bg ?? '#0f172a',
      gridLines: options.gridLines ?? false,
      cellLabels: options.cellLabels ?? false,
      labelFontSize: options.labelFontSize ?? 0,
      colormap: options.colormap ?? 'grayscale',
      normMin: options.normMin ?? NaN,
      normMax: options.normMax ?? NaN,
      padding: options.padding ?? 2,
    }

    this.resize()
  }

  /** Sync canvas backing store to its CSS display size. Call on mount + resize. */
  resize(): void {
    const { canvas, dpr } = this
    const w = canvas.clientWidth || 1
    const h = canvas.clientHeight || 1
    const targetW = Math.round(w * dpr)
    const targetH = Math.round(h * dpr)
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW
      canvas.height = targetH
      this.ctx.scale(dpr, dpr)
    }
  }

  /** Fill with background color. */
  clear(): void {
    const w = this.canvas.clientWidth || 1
    const h = this.canvas.clientHeight || 1
    this.ctx.fillStyle = this.opts.bg
    this.ctx.fillRect(0, 0, w, h)
  }

  /**
   * Render a 2-D numeric matrix as a colored grid.
   * @param matrix  row-major: matrix[row][col]
   * @param opts    per-call overrides for colormap / normalization
   */
  render(matrix: number[][], opts: RenderOptions = {}): void {
    const rows = matrix.length
    if (rows === 0) return
    const cols = matrix[0].length
    if (cols === 0) return

    this.resize()
    this.clear()

    const colormap = opts.colormap ?? this.opts.colormap
    const pad = this.opts.padding

    const cssW = this.canvas.clientWidth
    const cssH = this.canvas.clientHeight
    const gridW = cssW - pad * 2
    const gridH = cssH - pad * 2
    const cellW = gridW / cols
    const cellH = gridH / rows

    // Cache for highlight helpers
    this._cellW = cellW
    this._cellH = cellH
    this._offsetX = pad
    this._offsetY = pad

    // Determine normalization range
    let normMin = opts.normMin ?? (isNaN(this.opts.normMin) ? NaN : this.opts.normMin)
    let normMax = opts.normMax ?? (isNaN(this.opts.normMax) ? NaN : this.opts.normMax)

    if (isNaN(normMin) || isNaN(normMax)) {
      let lo = Infinity, hi = -Infinity
      for (const row of matrix) {
        for (const v of row) {
          if (isFinite(v)) {
            if (v < lo) lo = v
            if (v > hi) hi = v
          }
        }
      }
      if (!isFinite(lo)) { lo = 0; hi = 1 }
      if (lo === hi) { lo -= 1; hi += 1 }
      if (isNaN(normMin)) normMin = lo
      if (isNaN(normMax)) normMax = hi
    }

    const range = normMax - normMin || 1
    const { ctx } = this

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const raw = matrix[r][c]
        const px = pad + c * cellW
        const py = pad + r * cellH

        let rgb: [number, number, number]
        if (colormap === 'diverging') {
          // map normMin..0..normMax to -1..0..+1
          const t = clamp((raw - normMin) / range * 2 - 1, -1, 1)
          rgb = divergingColor(t)
        } else {
          const t = clamp((raw - normMin) / range)
          rgb = grayscaleColor(t)
        }

        ctx.fillStyle = toRgbString(...rgb)
        ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5) // +0.5 to avoid sub-pixel gaps
      }
    }

    // Grid lines
    if (this.opts.gridLines) {
      ctx.strokeStyle = 'rgba(148,163,184,0.15)'
      ctx.lineWidth = 0.5
      for (let r = 1; r < rows; r++) {
        const y = pad + r * cellH
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + gridW, y); ctx.stroke()
      }
      for (let c = 1; c < cols; c++) {
        const x = pad + c * cellW
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, pad + gridH); ctx.stroke()
      }
    }

    // Cell labels
    if (this.opts.cellLabels) {
      const fontSize = this.opts.labelFontSize > 0
        ? this.opts.labelFontSize
        : Math.max(7, Math.min(cellW * 0.35, cellH * 0.35, 13))
      ctx.font = `${fontSize}px JetBrains Mono, monospace`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = matrix[r][c]
          const px = pad + c * cellW
          const py = pad + r * cellH
          const cx = px + cellW / 2
          const cy = py + cellH / 2

          // Pick label color with contrast
          const t = clamp((v - normMin) / range)
          const brightness = colormap === 'grayscale' ? t : 0.5
          ctx.fillStyle = brightness > 0.5 ? 'rgba(0,0,0,0.75)' : 'rgba(255,255,255,0.85)'

          const label = Math.abs(v) < 100 ? v.toFixed(2) : v.toFixed(0)
          ctx.fillText(label, cx, cy)
        }
      }
    }
  }

  /**
   * Draw an accent-colored border around a rectangular region of cells.
   * Used to highlight the current receptive field on the input matrix.
   * @param startRow  top-left cell row (0-indexed)
   * @param startCol  top-left cell col (0-indexed)
   * @param height    number of rows to cover
   * @param width     number of cols to cover
   * @param color     border color (default indigo-400 #818cf8)
   * @param lineWidth border width in CSS px (default 2.5)
   */
  highlightRect(
    startRow: number,
    startCol: number,
    height: number,
    width: number,
    color = '#818cf8',
    lineWidth = 2.5,
  ): void {
    const { ctx, _cellW, _cellH, _offsetX, _offsetY } = this
    const x = _offsetX + startCol * _cellW
    const y = _offsetY + startRow * _cellH
    const w = width * _cellW
    const h = height * _cellH

    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth
    ctx.shadowColor = color
    ctx.shadowBlur = 6
    ctx.strokeRect(x + lineWidth / 2, y + lineWidth / 2, w - lineWidth, h - lineWidth)
    ctx.restore()
  }

  /**
   * Draw a bright filled border on a single output cell (the cell currently
   * being written by the sweep animation).
   * @param row  cell row
   * @param col  cell col
   * @param color  default cyan-400 (#22d3ee)
   */
  highlightCell(row: number, col: number, color = '#22d3ee'): void {
    const { ctx, _cellW, _cellH, _offsetX, _offsetY } = this
    const x = _offsetX + col * _cellW
    const y = _offsetY + row * _cellH

    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = 2.5
    ctx.shadowColor = color
    ctx.shadowBlur = 8
    ctx.strokeRect(x + 1.5, y + 1.5, _cellW - 3, _cellH - 3)
    ctx.restore()
  }

  /** Return cell pixel bounds (CSS px, top-left + size). Useful for overlay labels. */
  getCellBounds(row: number, col: number): { x: number; y: number; w: number; h: number } {
    return {
      x: this._offsetX + col * this._cellW,
      y: this._offsetY + row * this._cellH,
      w: this._cellW,
      h: this._cellH,
    }
  }

  /** Update per-instance options at runtime (e.g. toggle cellLabels). */
  setOptions(patch: Partial<HeatmapOptions>): void {
    Object.assign(this.opts, patch)
  }

  getCtx(): CanvasRenderingContext2D {
    return this.ctx
  }
}
