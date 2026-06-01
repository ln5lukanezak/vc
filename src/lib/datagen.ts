import { mulberry32, makeRandn } from './mathutils'

export interface RegressionData {
  x: number[]
  y: number[]
}

// ─── 2-D Classification types ─────────────────────────────────────────────────

/** A single labeled 2-D point. Coords are in roughly [-1,1] for all generators. */
export interface Point2D {
  x: number
  y: number
  label: number
}

export interface ClassificationOptions {
  /** number of points total */
  n: number
  /** noise std added to each coordinate (0 = clean, 0.3 = noisy) */
  noise: number
  /** RNG seed for reproducibility */
  seed?: number
}

// ─── Two Moons ────────────────────────────────────────────────────────────────

/**
 * Classic two-moon dataset. Two interleaved half-circles in [-1.5, 1.5].
 * Label 0 = upper moon, label 1 = lower moon.
 */
export function generateTwoMoons({
  n,
  noise,
  seed = 42,
}: ClassificationOptions): Point2D[] {
  const rand = mulberry32(seed)
  const randn = makeRandn(seed + 1)
  const half = Math.floor(n / 2)
  const pts: Point2D[] = []

  for (let i = 0; i < half; i++) {
    const angle = Math.PI * (i / (half - 1 || 1))
    pts.push({
      x: Math.cos(angle) + randn() * noise,
      y: Math.sin(angle) + randn() * noise,
      label: 0,
    })
  }
  for (let i = 0; i < n - half; i++) {
    const angle = Math.PI * (i / (n - half - 1 || 1))
    pts.push({
      x: 1 - Math.cos(angle) + randn() * noise,
      y: -0.5 - Math.sin(angle) + randn() * noise,
      label: 1,
    })
  }

  // Shuffle
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]]
  }
  return pts
}

// ─── Concentric Circles ───────────────────────────────────────────────────────

/**
 * Two concentric circles in [-1.2, 1.2]. Label 0 = inner, label 1 = outer.
 * The inner circle has radius ~0.4, outer ~1.0.
 */
export function generateCircles({
  n,
  noise,
  seed = 42,
}: ClassificationOptions): Point2D[] {
  const rand = mulberry32(seed)
  const randn = makeRandn(seed + 1)
  const half = Math.floor(n / 2)
  const pts: Point2D[] = []

  for (let i = 0; i < half; i++) {
    const angle = 2 * Math.PI * (i / (half - 1 || 1))
    const r = 0.4
    pts.push({
      x: r * Math.cos(angle) + randn() * noise,
      y: r * Math.sin(angle) + randn() * noise,
      label: 0,
    })
  }
  for (let i = 0; i < n - half; i++) {
    const angle = 2 * Math.PI * (i / (n - half - 1 || 1))
    const r = 1.0
    pts.push({
      x: r * Math.cos(angle) + randn() * noise,
      y: r * Math.sin(angle) + randn() * noise,
      label: 1,
    })
  }

  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]]
  }
  return pts
}

// ─── XOR (4 quadrants) ───────────────────────────────────────────────────────

/**
 * XOR pattern: points in [-1,1]×[-1,1], label = (x>0) XOR (y>0).
 * Quadrants: Q1&Q3 label 0, Q2&Q4 label 1 (or vice versa).
 */
export function generateXOR({
  n,
  noise,
  seed = 42,
}: ClassificationOptions): Point2D[] {
  const rand = mulberry32(seed)
  const randn = makeRandn(seed + 1)
  const pts: Point2D[] = []

  for (let i = 0; i < n; i++) {
    const x = (rand() * 2 - 1)
    const y = (rand() * 2 - 1)
    const label = ((x > 0) !== (y > 0)) ? 1 : 0
    pts.push({
      x: x + randn() * noise,
      y: y + randn() * noise,
      label,
    })
  }
  return pts
}

// ─── Spiral ───────────────────────────────────────────────────────────────────

/**
 * Archimedean spiral dataset. For numClasses=2 gives two interleaved spirals;
 * numClasses=3 gives three-class spiral (each arm label 0/1/2).
 * All points in approximately [-1,1]×[-1,1].
 */
export function generateSpiral({
  n,
  noise,
  seed = 42,
  numClasses = 2,
}: ClassificationOptions & { numClasses?: number }): Point2D[] {
  const rand = mulberry32(seed)
  const randn = makeRandn(seed + 1)
  const perClass = Math.floor(n / numClasses)
  const pts: Point2D[] = []

  for (let c = 0; c < numClasses; c++) {
    const count = c < numClasses - 1 ? perClass : n - pts.length
    for (let i = 0; i < count; i++) {
      const t = i / (perClass - 1 || 1)
      const angle = t * 2.5 * Math.PI + (2 * Math.PI * c) / numClasses
      const r = 0.1 + 0.85 * t
      pts.push({
        x: r * Math.cos(angle) + randn() * noise,
        y: r * Math.sin(angle) + randn() * noise,
        label: c,
      })
    }
  }

  // Shuffle
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]]
  }
  return pts
}

// ─── Gaussian Blobs ───────────────────────────────────────────────────────────

export interface BlobsOptions {
  /** total number of points (distributed evenly across classes) */
  n: number
  /** number of classes / clusters (2–4) */
  classes: number
  /**
   * inter-cluster distance multiplier (1 = clusters placed on a unit circle,
   * larger values spread them further apart). Typical range: 0.5–3.
   */
  separation: number
  /** per-cluster isotropic std (controls cluster spread). Typical range: 0.05–0.5 */
  noise: number
  /** RNG seed for reproducibility */
  seed?: number
}

/**
 * Isotropic Gaussian blobs for multi-class classification.
 *
 * `classes` (2–4) cluster centres are placed evenly around a circle of radius
 * `separation`.  Each point is drawn as centre + N(0, noise²) in both axes.
 * Coordinates are in roughly [-(separation+noise·3), +(separation+noise·3)].
 *
 * Label 0…classes-1.  Points are shuffled before return.
 */
export function generateBlobs({
  n,
  classes,
  separation,
  noise,
  seed = 42,
}: BlobsOptions): Point2D[] {
  const k = Math.max(2, Math.min(4, Math.round(classes)))
  const rand = mulberry32(seed)
  const randn = makeRandn(seed + 1)
  const perClass = Math.floor(n / k)
  const pts: Point2D[] = []

  for (let c = 0; c < k; c++) {
    const angle = (2 * Math.PI * c) / k
    const cx = separation * Math.cos(angle)
    const cy = separation * Math.sin(angle)
    const count = c < k - 1 ? perClass : n - pts.length
    for (let i = 0; i < count; i++) {
      pts.push({
        x: cx + randn() * noise,
        y: cy + randn() * noise,
        label: c,
      })
    }
  }

  // Shuffle
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [pts[i], pts[j]] = [pts[j], pts[i]]
  }
  return pts
}

export interface GenerateRegressionOptions {
  n: number       // number of samples
  noise: number   // gaussian noise std
  degree: number  // degree of the TRUE polynomial (1–6)
  seed?: number
}

/**
 * Generate a synthetic regression dataset.
 * The true function is a random polynomial of the given degree over x in [-3, 3].
 * Returns { x, y } where each is a flat array of length n.
 */
export function generateRegression({
  n,
  noise,
  degree,
  seed = 42,
}: GenerateRegressionOptions): RegressionData {
  const rand = mulberry32(seed)
  const randn = makeRandn(seed + 1)

  // Random coefficients for the true polynomial
  const coeffs: number[] = []
  for (let d = 0; d <= degree; d++) {
    // Scale down higher-degree terms to keep y in a reasonable range
    const scale = 1.5 / (d + 1)
    coeffs.push((rand() * 2 - 1) * scale)
  }

  const x: number[] = []
  const y: number[] = []

  for (let i = 0; i < n; i++) {
    // Uniformly spaced + slight jitter, x in [-3, 3]
    const xi = -3 + (6 * i) / (n - 1 || 1) + (rand() - 0.5) * (6 / n) * 0.5
    let yi = 0
    for (let d = 0; d <= degree; d++) {
      yi += coeffs[d] * Math.pow(xi, d)
    }
    yi += randn() * noise
    x.push(xi)
    y.push(yi)
  }

  // Sort by x for cleaner plotting
  const indices = x.map((_, i) => i).sort((a, b) => x[a] - x[b])
  return {
    x: indices.map((i) => x[i]),
    y: indices.map((i) => y[i]),
  }
}
