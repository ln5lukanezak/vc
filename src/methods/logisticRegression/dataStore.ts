/**
 * Method-scoped singleton data store for Logistic & Softmax Regression.
 * Persists between tab switches without lifting state to App.
 */
import { generateBlobs, type Point2D } from '../../lib/datagen'
import type { OptimizerName } from '../../lib/optimizers'

export interface LRDataConfig {
  classes: number      // 2–4
  separation: number   // inter-cluster distance
  noise: number        // cluster spread
  n: number            // total samples
  seed: number
}

export interface LRTrainConfig {
  optimizer: OptimizerName
  lr: number
  l2: number
  batchSize: number
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

let _dataConfig: LRDataConfig = {
  classes: 2,
  separation: 1.2,
  noise: 0.25,
  n: 200,
  seed: 42,
}

let _trainConfig: LRTrainConfig = {
  optimizer: 'Adam',
  lr: 0.05,
  l2: 0,
  batchSize: 32,
}

let _data: Point2D[] = []
let _listeners: Array<() => void> = []

// ─── Dataset generation ───────────────────────────────────────────────────────

export function generateData(cfg: LRDataConfig = _dataConfig): Point2D[] {
  return generateBlobs({
    n: cfg.n,
    classes: cfg.classes,
    separation: cfg.separation,
    noise: cfg.noise,
    seed: cfg.seed,
  })
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function getDataConfig(): LRDataConfig { return { ..._dataConfig } }
export function getTrainConfig(): LRTrainConfig { return { ..._trainConfig } }
export function getData(): Point2D[] { return _data }

export function setDataConfig(cfg: Partial<LRDataConfig>): void {
  _dataConfig = { ..._dataConfig, ...cfg }
}

export function setTrainConfig(cfg: Partial<LRTrainConfig>): void {
  _trainConfig = { ..._trainConfig, ...cfg }
}

export function regenerate(): void {
  _data = generateData(_dataConfig)
  _listeners.forEach(l => l())
}

export function subscribe(fn: () => void): () => void {
  _listeners.push(fn)
  return () => { _listeners = _listeners.filter(l => l !== fn) }
}

// Initialize on module load
regenerate()
