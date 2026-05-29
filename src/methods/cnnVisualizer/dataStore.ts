/**
 * dataStore.ts — CNN Visualizer method-scoped singleton.
 * Holds the selected input image and kernel configuration so they
 * persist across tab switches without lifting state to App.
 */

// ─── Built-in sample images (grayscale, row-major, 0=black 255=white) ─────────

/** 16×16 diagonal edge image */
const DIAGONAL_EDGE: number[][] = (() => {
  const img: number[][] = []
  for (let r = 0; r < 16; r++) {
    const row: number[] = []
    for (let c = 0; c < 16; c++) {
      // Bright diagonal band ±2 cells wide
      row.push(Math.abs(r - c) <= 2 ? 220 : 30)
    }
    img.push(row)
  }
  return img
})()

/** 16×16 cross / plus shape */
const CROSS: number[][] = (() => {
  const img: number[][] = []
  for (let r = 0; r < 16; r++) {
    const row: number[] = []
    for (let c = 0; c < 16; c++) {
      const cx = Math.abs(c - 7) <= 1
      const cy = Math.abs(r - 7) <= 1
      row.push(cx || cy ? 220 : 30)
    }
    img.push(row)
  }
  return img
})()

/** 16×16 digit-like "7" glyph */
const DIGIT_7: number[][] = (() => {
  const ON = 220, OFF = 30
  // 0=off, 1=on
  const template = [
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
  ]
  return template.map(row => row.map(v => v ? ON : OFF))
})()

/** 16×16 checkerboard — good for showing high-frequency response */
const CHECKERBOARD: number[][] = (() => {
  const img: number[][] = []
  for (let r = 0; r < 16; r++) {
    const row: number[] = []
    for (let c = 0; c < 16; c++) {
      row.push((r + c) % 2 === 0 ? 200 : 30)
    }
    img.push(row)
  }
  return img
})()

export type ImageKey = 'diagonal' | 'cross' | 'digit7' | 'checkerboard'

export const SAMPLE_IMAGES: Record<ImageKey, { label: string; data: number[][] }> = {
  diagonal:    { label: 'Diagonal Edge',  data: DIAGONAL_EDGE  },
  cross:       { label: 'Cross / Plus',   data: CROSS          },
  digit7:      { label: 'Digit "7"',      data: DIGIT_7        },
  checkerboard:{ label: 'Checkerboard',   data: CHECKERBOARD   },
}

// ─── Kernel definitions ───────────────────────────────────────────────────────

export type KernelKey =
  | 'identity'
  | 'box_blur'
  | 'gaussian'
  | 'sharpen'
  | 'sobel_x'
  | 'sobel_y'
  | 'emboss'
  | 'outline'

export const KERNELS: Record<KernelKey, { label: string; matrix: number[][] }> = {
  identity: {
    label: 'Identity',
    matrix: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
  },
  box_blur: {
    label: 'Box Blur',
    matrix: [
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9],
    ],
  },
  gaussian: {
    label: 'Gaussian Blur',
    matrix: [
      [1/16, 2/16, 1/16],
      [2/16, 4/16, 2/16],
      [1/16, 2/16, 1/16],
    ],
  },
  sharpen: {
    label: 'Sharpen',
    matrix: [
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0],
    ],
  },
  sobel_x: {
    label: 'Sobel-X (vertical edges)',
    matrix: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
  },
  sobel_y: {
    label: 'Sobel-Y (horizontal edges)',
    matrix: [
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1],
    ],
  },
  emboss: {
    label: 'Emboss',
    matrix: [
      [-2, -1, 0],
      [-1,  1, 1],
      [ 0,  1, 2],
    ],
  },
  outline: {
    label: 'Outline / Laplacian',
    matrix: [
      [-1, -1, -1],
      [-1,  8, -1],
      [-1, -1, -1],
    ],
  },
}

// ─── State ────────────────────────────────────────────────────────────────────

export interface CNNConfig {
  imageKey: ImageKey
  kernelKey: KernelKey
  stride: number
  padding: 'valid' | 'same'
  speedMs: number // ms between sweep steps
}

let _config: CNNConfig = {
  imageKey: 'diagonal',
  kernelKey: 'sobel_x',
  stride: 1,
  padding: 'valid',
  speedMs: 80,
}

export function getCNNConfig(): CNNConfig {
  return { ..._config }
}

export function setCNNConfig(patch: Partial<CNNConfig>): void {
  _config = { ..._config, ...patch }
}
