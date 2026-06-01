/**
 * Method-scoped singleton data store for the SVM method.
 */
import {
  generateBlobs,
  generateCircles,
  generateTwoMoons,
  type Point2D,
} from '../../lib/datagen'
import type { KernelType } from './svm'

export type DatasetType = 'blobs-sep' | 'blobs-over' | 'circles' | 'moons'

export interface SVMDataConfig {
  dataset: DatasetType
  noise: number
  n: number
  seed: number
}

export interface SVMTrainConfig {
  kernel: KernelType
  C: number
  gamma: number
  degree: number
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

let _dataConfig: SVMDataConfig = {
  dataset: 'blobs-sep',
  noise: 0.15,
  n: 80,
  seed: 42,
}

let _trainConfig: SVMTrainConfig = {
  kernel: 'rbf',
  C: 1,
  gamma: 0.5,
  degree: 3,
}

let _data: Point2D[] = []
let _listeners: Array<() => void> = []

// ─── Dataset generation ───────────────────────────────────────────────────────

export function generateData(cfg: SVMDataConfig = _dataConfig): Point2D[] {
  switch (cfg.dataset) {
    case 'blobs-sep':
      return generateBlobs({ n: cfg.n, classes: 2, separation: 1.8, noise: cfg.noise, seed: cfg.seed })
    case 'blobs-over':
      return generateBlobs({ n: cfg.n, classes: 2, separation: 0.7, noise: cfg.noise + 0.1, seed: cfg.seed })
    case 'circles':
      return generateCircles({ n: cfg.n, noise: cfg.noise, seed: cfg.seed })
    case 'moons':
      return generateTwoMoons({ n: cfg.n, noise: cfg.noise, seed: cfg.seed })
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function getDataConfig(): SVMDataConfig { return { ..._dataConfig } }
export function getTrainConfig(): SVMTrainConfig { return { ..._trainConfig } }
export function getData(): Point2D[] { return _data }

export function setDataConfig(cfg: Partial<SVMDataConfig>): void {
  _dataConfig = { ..._dataConfig, ...cfg }
}

export function setTrainConfig(cfg: Partial<SVMTrainConfig>): void {
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
