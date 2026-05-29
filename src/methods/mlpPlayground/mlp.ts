/**
 * mlp.ts — From-scratch MLP with backprop, dropout, L2, mini-batch training.
 *
 * Architecture:
 *   Input dim = 2 → 1–4 hidden layers (1–16 neurons each) → K classes (softmax)
 *
 * Optimizer integration:
 *   ALL weights and biases for ALL layers are FLATTENED into a single number[]
 *   param vector. Gradients are flattened in the same order. We call
 *   optimizer.step(params, grads) once per mini-batch, then UNFLATTEN back
 *   into the layer weight/bias arrays. This reuses optimizers.ts cleanly.
 *
 * Layout of the flat param vector:
 *   [layer0_W row-major, layer0_b, layer1_W row-major, layer1_b, ...]
 */

import { mulberry32, makeRandn } from '../../lib/mathutils'
import type { Optimizer } from '../../lib/optimizers'
import type { Point2D } from '../../lib/datagen'

// ─── Types ────────────────────────────────────────────────────────────────────

export type ActivationName = 'relu' | 'leakyrelu' | 'tanh' | 'sigmoid'

export interface LayerConfig {
  neurons: number
}

export interface MLPConfig {
  hiddenLayers: LayerConfig[]   // 1–4 entries
  activation: ActivationName
  numClasses: number            // 2 or 3
  l2: number                    // weight-decay coefficient
  dropout: number               // dropout rate (0 = off, 0.5 = 50% dropped)
}

interface Layer {
  /** W[outNeuron][inNeuron] */
  W: number[][]
  /** b[outNeuron] */
  b: number[]
  inDim: number
  outDim: number
}

interface ForwardCache {
  /** pre-activation z for each layer (including output) */
  zs: number[][]
  /** post-activation a for each hidden layer; a[0] = input x */
  as: number[][]
  /** dropout masks: 1 = keep, 0 = dropped (null for output layer) */
  masks: (number[] | null)[]
}

// ─── Activation functions ─────────────────────────────────────────────────────

function applyActivation(z: number[], name: ActivationName): number[] {
  switch (name) {
    case 'relu':      return z.map(v => Math.max(0, v))
    case 'leakyrelu': return z.map(v => v >= 0 ? v : 0.01 * v)
    case 'tanh':      return z.map(v => Math.tanh(v))
    case 'sigmoid':   return z.map(v => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v)))))
  }
}

function activationGrad(_a: number[], z: number[], name: ActivationName): number[] {
  // All cases derive from z (pre-activation), so dropout never corrupts the derivative.
  // _a is unused but kept in signature for call-site compatibility.
  switch (name) {
    case 'relu':      return z.map(v => v > 0 ? 1 : 0)
    case 'leakyrelu': return z.map(v => v >= 0 ? 1 : 0.01)
    case 'tanh':      return z.map(v => { const t = Math.tanh(v); return 1 - t * t })
    case 'sigmoid':   return z.map(v => { const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v)))); return s * (1 - s) })
  }
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

function softmax(z: number[]): number[] {
  const max = Math.max(...z)
  const exp = z.map(v => Math.exp(v - max))
  const sum = exp.reduce((s, v) => s + v, 0) + 1e-12
  return exp.map(v => v / sum)
}

// ─── Xavier / He init ─────────────────────────────────────────────────────────

function initLayer(inDim: number, outDim: number, activation: ActivationName, randn: () => number): Layer {
  // He for ReLU/LeakyReLU, Xavier for Tanh/Sigmoid
  const scale = (activation === 'relu' || activation === 'leakyrelu')
    ? Math.sqrt(2 / inDim)
    : Math.sqrt(1 / inDim)

  const W: number[][] = []
  for (let o = 0; o < outDim; o++) {
    const row: number[] = []
    for (let i = 0; i < inDim; i++) row.push(randn() * scale)
    W.push(row)
  }
  return {
    W,
    b: new Array(outDim).fill(0),
    inDim,
    outDim,
  }
}

// ─── MLP class ────────────────────────────────────────────────────────────────

export class MLP {
  private layers: Layer[] = []
  private config: MLPConfig
  private optimizer: Optimizer
  private rngSeed: number
  private _randDropout: (() => number) | null = null

  constructor(config: MLPConfig, optimizer: Optimizer, seed = 42) {
    this.config = config
    this.optimizer = optimizer
    this.rngSeed = seed
    this._buildAndInit(seed)
  }

  private _buildAndInit(seed: number) {
    const randn = makeRandn(seed)
    this.layers = []
    const { hiddenLayers, activation, numClasses } = this.config

    let inDim = 2 // input features: x, y
    for (const hl of hiddenLayers) {
      this.layers.push(initLayer(inDim, hl.neurons, activation, randn))
      inDim = hl.neurons
    }
    // Output layer — softmax; init with Xavier regardless of hidden activation
    this.layers.push(initLayer(inDim, numClasses, 'sigmoid', randn))

    // Dropout RNG
    this._randDropout = mulberry32(seed + 999)
    this.optimizer.reset()
  }

  reset(seed?: number): void {
    this.rngSeed = seed ?? this.rngSeed
    this._buildAndInit(this.rngSeed)
  }

  // ── Param flattening ──────────────────────────────────────────────────────

  private _flatten(): number[] {
    const out: number[] = []
    for (const layer of this.layers) {
      for (const row of layer.W) for (const v of row) out.push(v)
      for (const v of layer.b) out.push(v)
    }
    return out
  }

  private _unflatten(flat: number[]): void {
    let idx = 0
    for (const layer of this.layers) {
      for (let o = 0; o < layer.outDim; o++) {
        for (let i = 0; i < layer.inDim; i++) {
          layer.W[o][i] = flat[idx++]
        }
      }
      for (let o = 0; o < layer.outDim; o++) {
        layer.b[o] = flat[idx++]
      }
    }
  }

  // ── Forward pass ──────────────────────────────────────────────────────────

  /**
   * Forward pass for a single sample.
   * @param x input [x0, x1]
   * @param training if true, apply dropout
   */
  private _forward(x: number[], training: boolean): { probs: number[]; cache: ForwardCache } {
    const { activation, dropout } = this.config
    const zs: number[][] = []
    const as: number[][] = [x]
    const masks: (number[] | null)[] = []

    let a = x
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l]
      const isOutput = l === this.layers.length - 1

      // z = W·a + b
      const z: number[] = []
      for (let o = 0; o < layer.outDim; o++) {
        let sum = layer.b[o]
        for (let i = 0; i < layer.inDim; i++) sum += layer.W[o][i] * a[i]
        z.push(sum)
      }
      zs.push(z)

      if (isOutput) {
        // Softmax
        const probs = softmax(z)
        as.push(probs)
        masks.push(null)
        return { probs, cache: { zs, as, masks } }
      }

      // Hidden activation
      let act = applyActivation(z, activation)

      // Dropout (training only, not output layer)
      let mask: number[] | null = null
      if (training && dropout > 0) {
        mask = act.map(() => (this._randDropout!() > dropout ? 1 : 0))
        const scale = 1 / (1 - dropout + 1e-12)
        act = act.map((v, i) => v * mask![i] * scale)
      }
      masks.push(mask)
      as.push(act)
      a = act
    }

    // Should never reach here
    return { probs: [], cache: { zs, as, masks } }
  }

  // ── Backprop ──────────────────────────────────────────────────────────────

  /**
   * Backprop for a single sample.
   * Returns flat gradient vector in the same layout as _flatten().
   */
  private _backward(
    cache: ForwardCache,
    label: number,
  ): number[] {
    const { activation, l2, dropout } = this.config
    const { zs, as, masks } = cache

    // Softmax + cross-entropy: d(loss)/dz_output = probs - one_hot
    const probs = as[as.length - 1]
    let delta: number[] = probs.map((p, k) => p - (k === label ? 1 : 0))

    // We collect each layer's flat grads in reverse order, then reverse at end
    const nLayers = this.layers.length
    // blocks[0] = last layer's grads, blocks[nLayers-1] = first layer's grads
    const blocks: number[][] = []

    for (let l = nLayers - 1; l >= 0; l--) {
      const layer = this.layers[l]
      // a_in is the input activations for layer l: as[l] (since as[0]=input, as[1]=after layer 0, ...)
      const a_in = as[l]

      // dW[o][i] = delta[o] * a_in[i] + l2 * W[o][i]
      const blockGrads: number[] = []
      for (let o = 0; o < layer.outDim; o++) {
        for (let i = 0; i < layer.inDim; i++) {
          blockGrads.push(delta[o] * a_in[i] + l2 * layer.W[o][i])
        }
      }
      // db[o] = delta[o]
      for (let o = 0; o < layer.outDim; o++) {
        blockGrads.push(delta[o])
      }
      blocks.push(blockGrads)

      if (l === 0) break // no need to propagate to inputs

      // Propagate delta backward: d_a_prev[i] = Σ_o delta[o] * W[o][i]
      const d_a_prev: number[] = new Array(layer.inDim).fill(0)
      for (let o = 0; o < layer.outDim; o++) {
        for (let i = 0; i < layer.inDim; i++) {
          d_a_prev[i] += delta[o] * layer.W[o][i]
        }
      }

      // Apply dropout mask for the previous hidden layer
      const prevMask = masks[l - 1]
      if (prevMask) {
        const scale = 1 / (1 - dropout + 1e-12)
        for (let i = 0; i < d_a_prev.length; i++) {
          d_a_prev[i] *= prevMask[i] * scale
        }
      }

      // Apply activation gradient: delta_prev = d_a_prev ⊙ φ'(z_{l-1})
      // as[l] = φ(z_{l-1}) (activation output of layer l-1)
      const prevZ = zs[l - 1]
      const prevA = as[l]  // = φ(z_{l-1})
      const dActGrad = activationGrad(prevA, prevZ, activation)
      delta = d_a_prev.map((v, i) => v * dActGrad[i])
    }

    // blocks is in reverse layer order; reverse to get layer-0-first order
    const orderedGrads: number[] = []
    // blocks[0] is last layer, blocks[nLayers-1] is first layer
    for (let l = blocks.length - 1; l >= 0; l--) {
      orderedGrads.push(...blocks[l])
    }
    return orderedGrads
  }

  // ── Training ──────────────────────────────────────────────────────────────

  /**
   * Run one epoch of mini-batch SGD. Returns {loss, acc}.
   */
  trainEpoch(data: Point2D[], batchSize: number): { loss: number; acc: number } {
    const n = data.length
    if (n === 0) return { loss: 0, acc: 0 }

    // Shuffle data in-place copy
    const shuffled = [...data]
    const rng = this._randDropout! // reuse for shuffling
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }

    let totalLoss = 0
    let totalCorrect = 0

    for (let start = 0; start < n; start += batchSize) {
      const batch = shuffled.slice(start, start + batchSize)
      const bLen = batch.length

      // Accumulate flat gradients over the batch
      const paramLen = this._flatten().length
      const sumGrads = new Float64Array(paramLen)

      for (const pt of batch) {
        const { probs, cache } = this._forward([pt.x, pt.y], true)
        const grads = this._backward(cache, pt.label)

        for (let i = 0; i < paramLen; i++) sumGrads[i] += grads[i]

        // Loss: -log(probs[label])
        const p = Math.max(probs[pt.label], 1e-12)
        totalLoss -= Math.log(p)

        // Accuracy
        const pred = probs.indexOf(Math.max(...probs))
        if (pred === pt.label) totalCorrect++
      }

      // Average gradients
      const avgGrads = Array.from(sumGrads).map(v => v / bLen)

      // Optimizer step
      const params = this._flatten()
      const newParams = this.optimizer.step(params, avgGrads)
      this._unflatten(newParams)
    }

    return { loss: totalLoss / n, acc: totalCorrect / n }
  }

  // ── Inference ──────────────────────────────────────────────────────────────

  /** Predict class probabilities for a single point (no dropout). */
  predict(x: number, y: number): number[] {
    const { probs } = this._forward([x, y], false)
    return probs
  }

  /**
   * Evaluate the net over a grid and return probabilities for class 0.
   * xs and ys are 1-D coordinate arrays; result is [row][col] = prob(class 0).
   */
  predictGrid(xs: number[], ys: number[]): number[][] {
    const rows = ys.length
    const cols = xs.length
    const grid: number[][] = []
    for (let r = 0; r < rows; r++) {
      const row: number[] = []
      for (let c = 0; c < cols; c++) {
        const probs = this.predict(xs[c], ys[r])
        row.push(probs[0])
      }
      grid.push(row)
    }
    return grid
  }

  /**
   * Return per-hidden-neuron activation grids.
   * Returns an array of length (total hidden neurons), each being a [rows][cols] matrix
   * of that neuron's activation over the xs×ys grid.
   * Also returns neuronInfo: [{layerIdx, neuronIdx}] parallel to the outer array.
   */
  neuronActivations(
    xs: number[],
    ys: number[],
  ): { grids: number[][][]; neuronInfo: Array<{ layerIdx: number; neuronIdx: number }> } {
    const rows = ys.length
    const cols = xs.length
    const nHidden = this.layers.length - 1 // exclude output layer

    // Count total hidden neurons
    const neuronInfo: Array<{ layerIdx: number; neuronIdx: number }> = []
    for (let l = 0; l < nHidden; l++) {
      for (let n = 0; n < this.layers[l].outDim; n++) {
        neuronInfo.push({ layerIdx: l, neuronIdx: n })
      }
    }

    const grids: number[][][] = neuronInfo.map(() =>
      Array.from({ length: rows }, () => new Array(cols).fill(0)),
    )

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Forward without dropout, collect hidden activations
        const { cache } = this._forward([xs[c], ys[r]], false)
        // as[0] = input, as[1] = after layer 0, ..., as[nHidden] = after last hidden
        let idx = 0
        for (let l = 0; l < nHidden; l++) {
          const layerActs = cache.as[l + 1] // activations after layer l
          for (let n = 0; n < layerActs.length; n++) {
            grids[idx][r][c] = layerActs[n]
            idx++
          }
        }
      }
    }

    return { grids, neuronInfo }
  }

  /** Total number of trainable parameters. */
  get paramCount(): number {
    return this.layers.reduce((s, l) => s + l.outDim * l.inDim + l.outDim, 0)
  }

  /** Number of hidden neurons per layer */
  get hiddenSizes(): number[] {
    return this.layers.slice(0, -1).map(l => l.outDim)
  }
}
