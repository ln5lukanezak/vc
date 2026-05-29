// Method-scoped singleton: each method gets its own store module — do not import or share this across methods.
/**
 * Shared data store for Linear Regression method.
 * Persists between tab switches without lifting state to App.
 */
import { generateRegression, type RegressionData } from '../../lib/datagen'

export interface DataConfig {
  n: number
  noise: number
  degree: number
  seed: number
}

let _config: DataConfig = { n: 80, noise: 0.4, degree: 2, seed: 42 }
let _data: RegressionData = generateRegression(_config)
let _listeners: Array<() => void> = []

export function getDataConfig(): DataConfig {
  return { ..._config }
}

export function getData(): RegressionData {
  return _data
}

export function setDataConfig(cfg: Partial<DataConfig>): void {
  _config = { ..._config, ...cfg }
}

export function regenerate(): void {
  _data = generateRegression(_config)
  _listeners.forEach((l) => l())
}

export function subscribe(fn: () => void): () => void {
  _listeners.push(fn)
  return () => {
    _listeners = _listeners.filter((l) => l !== fn)
  }
}
