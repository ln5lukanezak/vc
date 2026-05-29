/**
 * Method-scoped singleton data store for MLP Playground.
 * Persists between tab switches without lifting state to App.
 */
import {
  generateTwoMoons,
  generateCircles,
  generateXOR,
  generateSpiral,
  type Point2D,
} from '../../lib/datagen'
import type { ActivationName, LayerConfig } from './mlp'
import type { OptimizerName } from '../../lib/optimizers'

export type DatasetName = 'moons' | 'circles' | 'xor' | 'spiral' | 'spiral3'

export interface MLPDataConfig {
  dataset: DatasetName
  n: number
  noise: number
  seed: number
}

export interface MLPTrainConfig {
  hiddenLayers: LayerConfig[]
  activation: ActivationName
  optimizer: OptimizerName
  lr: number
  l2: number
  dropout: number
  batchSize: number
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

let _dataConfig: MLPDataConfig = {
  dataset: 'moons',
  n: 200,
  noise: 0.1,
  seed: 42,
}

let _trainConfig: MLPTrainConfig = {
  hiddenLayers: [{ neurons: 8 }, { neurons: 8 }],
  activation: 'relu',
  optimizer: 'Adam',
  lr: 0.01,
  l2: 0,
  dropout: 0,
  batchSize: 32,
}

let _data: Point2D[] = []
let _listeners: Array<() => void> = []

// ─── Dataset generation ───────────────────────────────────────────────────────

export function numClasses(dataset: DatasetName): number {
  return dataset === 'spiral3' ? 3 : 2
}

export function generateData(cfg: MLPDataConfig = _dataConfig): Point2D[] {
  const { dataset, n, noise, seed } = cfg
  switch (dataset) {
    case 'moons':   return generateTwoMoons({ n, noise, seed })
    case 'circles': return generateCircles({ n, noise, seed })
    case 'xor':     return generateXOR({ n, noise, seed })
    case 'spiral':  return generateSpiral({ n, noise, seed, numClasses: 2 })
    case 'spiral3': return generateSpiral({ n, noise, seed, numClasses: 3 })
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function getDataConfig(): MLPDataConfig { return { ..._dataConfig } }
export function getTrainConfig(): MLPTrainConfig {
  return {
    ..._trainConfig,
    hiddenLayers: _trainConfig.hiddenLayers.map(l => ({ ...l })),
  }
}
export function getData(): Point2D[] { return _data }

export function setDataConfig(cfg: Partial<MLPDataConfig>): void {
  _dataConfig = { ..._dataConfig, ...cfg }
}

export function setTrainConfig(cfg: Partial<MLPTrainConfig>): void {
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

// Initialize data on module load
regenerate()
