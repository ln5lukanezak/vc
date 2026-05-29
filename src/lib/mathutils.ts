// ─── Seeded RNG (mulberry32) ──────────────────────────────────────────────────
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0
  return function () {
    s |= 0
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ─── Box–Muller gaussian (mean=0, std=1) ─────────────────────────────────────
export function randnPair(rand: () => number): [number, number] {
  const u1 = rand()
  const u2 = rand()
  const r = Math.sqrt(-2 * Math.log(u1 + 1e-12))
  const theta = 2 * Math.PI * u2
  return [r * Math.cos(theta), r * Math.sin(theta)]
}

/** Build a randn function backed by a seeded RNG */
export function makeRandn(seed: number): () => number {
  const rand = mulberry32(seed)
  let spare: number | null = null
  return () => {
    if (spare !== null) {
      const v = spare
      spare = null
      return v
    }
    const [a, b] = randnPair(rand)
    spare = b
    return a
  }
}

// ─── Vector helpers ───────────────────────────────────────────────────────────
export function dot(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export function mean(arr: number[]): number {
  if (arr.length === 0) return 0
  let s = 0
  for (const v of arr) s += v
  return s / arr.length
}

export function variance(arr: number[]): number {
  const m = mean(arr)
  let s = 0
  for (const v of arr) s += (v - m) ** 2
  return s / arr.length
}

export function std(arr: number[]): number {
  return Math.sqrt(variance(arr))
}

export function sum(arr: number[]): number {
  let s = 0
  for (const v of arr) s += v
  return s
}

/** Clip value to [lo, hi] */
export function clip(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

// ─── Standardize helpers ─────────────────────────────────────────────────────
export interface Scaler {
  means: number[]
  stds: number[]
}

/**
 * Fit a standard scaler on a 2-D matrix (rows = samples, cols = features).
 * Returns the scaler (means + stds).
 */
export function fitScaler(X: number[][]): Scaler {
  if (X.length === 0) return { means: [], stds: [] }
  const nFeatures = X[0].length
  const means: number[] = new Array(nFeatures).fill(0)
  const stds: number[] = new Array(nFeatures).fill(1)
  for (let j = 0; j < nFeatures; j++) {
    const col = X.map((row) => row[j])
    means[j] = mean(col)
    const s = std(col)
    stds[j] = s < 1e-12 ? 1 : s
  }
  return { means, stds }
}

/** Apply a fitted scaler to a matrix. Returns a new matrix. */
export function applyScaler(X: number[][], scaler: Scaler): number[][] {
  return X.map((row) =>
    row.map((v, j) => (v - scaler.means[j]) / scaler.stds[j]),
  )
}

/** Standardize columns of X in-place and return { scaled, scaler }. */
export function standardize(X: number[][]): { scaled: number[][]; scaler: Scaler } {
  const scaler = fitScaler(X)
  const scaled = applyScaler(X, scaler)
  return { scaled, scaler }
}
