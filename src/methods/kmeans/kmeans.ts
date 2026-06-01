// ─── K-Means / K-Means++ clustering (from scratch, no TF.js) ─────────────────

import { mulberry32 } from '../../lib/mathutils'

export type InitMethod = 'random' | 'kmeanspp'

export interface KMeansPoint {
  x: number
  y: number
}

function dist2(a: KMeansPoint, b: KMeansPoint): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return dx * dx + dy * dy
}

export class KMeans {
  private k: number
  private points: KMeansPoint[]
  private _centroids: KMeansPoint[]
  private _prevCentroids: KMeansPoint[]
  private _assignments: number[]
  private _iteration: number
  private rng: () => number

  constructor(k: number, points: KMeansPoint[], init: InitMethod = 'kmeanspp', seed = 42) {
    this.k = k
    this.points = points
    this.rng = mulberry32(seed)
    this._assignments = new Array(points.length).fill(0)
    this._centroids = []
    this._prevCentroids = []
    this._iteration = 0
    this._initCentroids(init)
    // Initial assignment (no update yet)
    this._assign()
  }

  // ─── Initialization ───────────────────────────────────────────────────────

  private _initCentroids(init: InitMethod): void {
    const pts = this.points
    if (pts.length === 0) {
      this._centroids = Array.from({ length: this.k }, () => ({ x: 0, y: 0 }))
      this._prevCentroids = this._centroids.map(c => ({ ...c }))
      return
    }

    if (init === 'random') {
      // Pick k distinct random points
      const indices = Array.from({ length: pts.length }, (_, i) => i)
      // Fisher-Yates shuffle (first k elements)
      for (let i = 0; i < this.k; i++) {
        const j = i + Math.floor(this.rng() * (indices.length - i))
        ;[indices[i], indices[j]] = [indices[j], indices[i]]
      }
      this._centroids = indices.slice(0, this.k).map(i => ({ ...pts[i] }))
    } else {
      // k-means++ seeding
      const chosen: KMeansPoint[] = []
      // First centroid: random point
      const firstIdx = Math.floor(this.rng() * pts.length)
      chosen.push({ ...pts[firstIdx] })

      for (let c = 1; c < this.k; c++) {
        // D²-weighted probability
        const dists = pts.map(p => {
          let minD2 = Infinity
          for (const cent of chosen) {
            const d = dist2(p, cent)
            if (d < minD2) minD2 = d
          }
          return minD2
        })
        const total = dists.reduce((a, b) => a + b, 0)
        let r = this.rng() * total
        let idx = 0
        for (let i = 0; i < dists.length; i++) {
          r -= dists[i]
          if (r <= 0) { idx = i; break }
          idx = i
        }
        chosen.push({ ...pts[idx] })
      }
      this._centroids = chosen
    }

    this._prevCentroids = this._centroids.map(c => ({ ...c }))
  }

  // ─── Assignment step ──────────────────────────────────────────────────────

  private _assign(): boolean {
    let changed = false
    for (let i = 0; i < this.points.length; i++) {
      let best = 0
      let bestD = Infinity
      for (let c = 0; c < this.k; c++) {
        const d = dist2(this.points[i], this._centroids[c])
        if (d < bestD) { bestD = d; best = c }
      }
      if (this._assignments[i] !== best) {
        this._assignments[i] = best
        changed = true
      }
    }
    return changed
  }

  // ─── Update step ──────────────────────────────────────────────────────────

  private _update(): void {
    this._prevCentroids = this._centroids.map(c => ({ ...c }))
    const sums: { sx: number; sy: number; count: number }[] = Array.from(
      { length: this.k }, () => ({ sx: 0, sy: 0, count: 0 })
    )

    for (let i = 0; i < this.points.length; i++) {
      const c = this._assignments[i]
      sums[c].sx += this.points[i].x
      sums[c].sy += this.points[i].y
      sums[c].count++
    }

    for (let c = 0; c < this.k; c++) {
      if (sums[c].count === 0) {
        // Empty cluster: re-seed to farthest point from any centroid
        let maxD = -1
        let farIdx = 0
        for (let i = 0; i < this.points.length; i++) {
          let minD2 = Infinity
          for (let cc = 0; cc < this.k; cc++) {
            const d = dist2(this.points[i], this._centroids[cc])
            if (d < minD2) minD2 = d
          }
          if (minD2 > maxD) { maxD = minD2; farIdx = i }
        }
        this._centroids[c] = { ...this.points[farIdx] }
      } else {
        this._centroids[c] = {
          x: sums[c].sx / sums[c].count,
          y: sums[c].sy / sums[c].count,
        }
      }
    }
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  /** One assign+update step. Returns {changed, inertia}. */
  step(): { changed: boolean; inertia: number } {
    const changed = this._assign()
    this._update()
    this._iteration++
    return { changed, inertia: this.inertia() }
  }

  /** Current inertia (WCSS). */
  inertia(): number {
    let sum = 0
    for (let i = 0; i < this.points.length; i++) {
      sum += dist2(this.points[i], this._centroids[this._assignments[i]])
    }
    return sum
  }

  assignments(): number[] { return this._assignments }
  centroids(): KMeansPoint[] { return this._centroids }
  prevCentroids(): KMeansPoint[] { return this._prevCentroids }
  iteration(): number { return this._iteration }
  getK(): number { return this.k }

  /** Reset and re-initialize centroids with a new seed. */
  reset(init: InitMethod, seed: number): void {
    this.rng = mulberry32(seed)
    this._iteration = 0
    this._assignments = new Array(this.points.length).fill(0)
    this._initCentroids(init)
    this._assign()
  }

  /** Update point cloud (for dataset change). */
  setPoints(pts: KMeansPoint[], init: InitMethod, seed: number): void {
    this.points = pts
    this._assignments = new Array(pts.length).fill(0)
    this.rng = mulberry32(seed)
    this._iteration = 0
    this._initCentroids(init)
    this._assign()
  }
}

// ─── Elbow helper: run full fit for given k, return final inertia ─────────────

export function fitInertia(
  pts: KMeansPoint[],
  k: number,
  init: InitMethod = 'kmeanspp',
  seed = 42,
  maxIter = 100,
): number {
  if (pts.length === 0 || k < 1) return 0
  const km = new KMeans(Math.min(k, pts.length), pts, init, seed)
  for (let i = 0; i < maxIter; i++) {
    const { changed } = km.step()
    if (!changed) break
  }
  return km.inertia()
}
