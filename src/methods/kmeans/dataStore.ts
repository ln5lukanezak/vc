/**
 * Method-scoped singleton data store for the K-Means method.
 */
import { generateBlobs } from '../../lib/datagen'
import type { KMeansPoint } from './kmeans'

export interface KMeansDataConfig {
  /** True cluster count used by generateBlobs */
  clusters: number
  /** Inter-cluster separation (radius) */
  separation: number
  /** Per-cluster noise std */
  noise: number
  /** Number of points */
  n: number
  /** RNG seed for data generation */
  seed: number
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

let _cfg: KMeansDataConfig = {
  clusters: 3,
  separation: 1.4,
  noise: 0.18,
  n: 120,
  seed: 42,
}

let _data: KMeansPoint[] = []
let _listeners: Array<() => void> = []

// ─── Dataset generation ───────────────────────────────────────────────────────

export function generateData(cfg: KMeansDataConfig = _cfg): KMeansPoint[] {
  // generateBlobs labels are ignored — we cluster purely on coordinates
  const pts = generateBlobs({
    n: cfg.n,
    classes: cfg.clusters,
    separation: cfg.separation,
    noise: cfg.noise,
    seed: cfg.seed,
  })
  return pts.map(p => ({ x: p.x, y: p.y }))
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function getConfig(): KMeansDataConfig { return { ..._cfg } }
export function getData(): KMeansPoint[] { return _data }

export function setConfig(cfg: Partial<KMeansDataConfig>): void {
  _cfg = { ..._cfg, ...cfg }
}

export function regenerate(): void {
  _data = generateData(_cfg)
  _listeners.forEach(l => l())
}

export function subscribe(fn: () => void): () => void {
  _listeners.push(fn)
  return () => { _listeners = _listeners.filter(l => l !== fn) }
}

// Initialize on module load
regenerate()
