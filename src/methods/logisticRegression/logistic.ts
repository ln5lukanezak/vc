/**
 * logistic.ts — From-scratch Softmax/Logistic Regression with L2 regularization.
 *
 * Model: a single linear layer  z[k] = W[k]·x + b[k]  followed by softmax.
 * For K=2 this is exactly logistic/sigmoid regression (softmax over 2 classes
 * is equivalent to the sigmoid binary classifier).
 *
 * Optimizer integration (same pattern as mlp.ts):
 *   Flatten W (row-major, K×2) then b (K) into a single number[].
 *   Call optimizer.step(params, grads), then unflatten back.
 *
 * Flat param layout:
 *   [ W[0][0], W[0][1],   // class 0 weights
 *     W[1][0], W[1][1],   // class 1 weights
 *     ...
 *     b[0], b[1], ... ]   // all biases
 */

import { mulberry32 } from '../../lib/mathutils'
import type { Optimizer } from '../../lib/optimizers'
import type { Point2D } from '../../lib/datagen'

// ─── Softmax (numerically stable) ────────────────────────────────────────────

function softmax(z: number[]): number[] {
  const max = Math.max(...z)
  const exp = z.map(v => Math.exp(v - max))
  const sum = exp.reduce((s, v) => s + v, 0) + 1e-12
  return exp.map(v => v / sum)
}

// ─── SoftmaxClassifier ────────────────────────────────────────────────────────

export class SoftmaxClassifier {
  /** W[k][f]: weight of class k for feature f (f=0→x, f=1→y) */
  private W: number[][]
  /** b[k]: bias for class k */
  private b: number[]

  private numClasses: number
  private l2: number
  private optimizer: Optimizer
  private rng: () => number

  constructor(
    numClasses: number,
    l2: number,
    optimizer: Optimizer,
    seed = 42,
  ) {
    this.numClasses = numClasses
    this.l2 = l2
    this.optimizer = optimizer
    this.rng = mulberry32(seed)

    // Xavier-style init: small random weights, zero bias
    const scale = Math.sqrt(1 / 2) * 0.1
    this.W = Array.from({ length: numClasses }, () =>
      [(this.rng() * 2 - 1) * scale, (this.rng() * 2 - 1) * scale],
    )
    this.b = new Array(numClasses).fill(0)
    this.optimizer.reset()
  }

  // ── Param flattening ────────────────────────────────────────────────────────

  private flatten(): number[] {
    const out: number[] = []
    for (const row of this.W) out.push(...row)
    out.push(...this.b)
    return out
  }

  private unflatten(flat: number[]): void {
    let idx = 0
    for (let k = 0; k < this.numClasses; k++) {
      this.W[k][0] = flat[idx++]
      this.W[k][1] = flat[idx++]
    }
    for (let k = 0; k < this.numClasses; k++) {
      this.b[k] = flat[idx++]
    }
  }

  // ── Forward ─────────────────────────────────────────────────────────────────

  /** Returns softmax probability vector [p0, p1, ...] for a 2-D point. */
  predict(x: number, y: number): number[] {
    const z = this.W.map((w, k) => w[0] * x + w[1] * y + this.b[k])
    return softmax(z)
  }

  // ── Training ────────────────────────────────────────────────────────────────

  /**
   * Run one epoch over `data` with mini-batches of size `batchSize`.
   * Returns { loss, acc } averaged over all samples.
   */
  trainEpoch(data: Point2D[], batchSize: number): { loss: number; acc: number } {
    const n = data.length
    if (n === 0) return { loss: 0, acc: 0 }

    // Shuffle
    const shuffled = [...data]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }

    let totalLoss = 0
    let totalCorrect = 0

    for (let start = 0; start < n; start += batchSize) {
      const batch = shuffled.slice(start, start + batchSize)
      const bLen = batch.length

      // Number of params: K*2 (weights) + K (biases) = K*3
      const paramLen = this.numClasses * 3
      const sumGrads = new Float64Array(paramLen)

      for (const pt of batch) {
        const probs = this.predict(pt.x, pt.y)

        // Loss: -log(probs[label])
        const p = Math.max(probs[pt.label], 1e-12)
        totalLoss -= Math.log(p)

        // Accuracy
        let bestK = 0
        for (let k = 1; k < this.numClasses; k++) {
          if (probs[k] > probs[bestK]) bestK = k
        }
        if (bestK === pt.label) totalCorrect++

        // Gradient of cross-entropy + softmax w.r.t. logits: delta[k] = probs[k] - one_hot[k]
        const delta = probs.map((p_k, k) => p_k - (k === pt.label ? 1 : 0))

        // dW[k][f] = delta[k] * x[f]  (+ L2 added after averaging)
        // db[k]    = delta[k]
        let gi = 0
        for (let k = 0; k < this.numClasses; k++) {
          sumGrads[gi++] += delta[k] * pt.x   // dW[k][0]
          sumGrads[gi++] += delta[k] * pt.y   // dW[k][1]
        }
        for (let k = 0; k < this.numClasses; k++) {
          sumGrads[gi++] += delta[k]           // db[k]
        }
      }

      // Average + L2 on weights
      const avgGrads: number[] = new Array(paramLen)
      let gi2 = 0
      for (let k = 0; k < this.numClasses; k++) {
        avgGrads[gi2] = sumGrads[gi2] / bLen + this.l2 * this.W[k][0]; gi2++
        avgGrads[gi2] = sumGrads[gi2] / bLen + this.l2 * this.W[k][1]; gi2++
      }
      for (let k = 0; k < this.numClasses; k++) {
        avgGrads[gi2] = sumGrads[gi2] / bLen; gi2++  // no L2 on biases
      }

      // Optimizer step
      const params = this.flatten()
      const newParams = this.optimizer.step(params, avgGrads)
      this.unflatten(newParams)
    }

    return { loss: totalLoss / n, acc: totalCorrect / n }
  }

  /**
   * Evaluate over a grid of (xs × ys) coordinates.
   * Returns a [rows][cols] matrix where each cell = prob(class 0) for K=2,
   * or the argmax class index for K>2 (encoded as k / (K-1) so it's in [0,1]).
   * The caller (Visualization) uses argmax coloring for K>2.
   */
  predictGrid(
    xs: number[],
    ys: number[],
  ): { grid: number[][]; classGrid: number[][] } {
    const rows = ys.length
    const cols = xs.length
    const grid: number[][] = []
    const classGrid: number[][] = []
    for (let r = 0; r < rows; r++) {
      const row: number[] = []
      const classRow: number[] = []
      for (let c = 0; c < cols; c++) {
        const probs = this.predict(xs[c], ys[r])
        row.push(probs[0])
        let bestK = 0
        for (let k = 1; k < this.numClasses; k++) {
          if (probs[k] > probs[bestK]) bestK = k
        }
        classRow.push(bestK)
      }
      grid.push(row)
      classGrid.push(classRow)
    }
    return { grid, classGrid }
  }

  /** Compute confusion matrix (K×K). matrix[true][pred] = count. */
  confusionMatrix(data: Point2D[]): number[][] {
    const K = this.numClasses
    const cm: number[][] = Array.from({ length: K }, () => new Array(K).fill(0))
    for (const pt of data) {
      const probs = this.predict(pt.x, pt.y)
      let bestK = 0
      for (let k = 1; k < K; k++) {
        if (probs[k] > probs[bestK]) bestK = k
      }
      cm[pt.label][bestK]++
    }
    return cm
  }

  /** Current weight matrix copy (for display). */
  getWeights(): number[][] {
    return this.W.map(row => [...row])
  }

  /** Total trainable parameters: K*2 + K = K*3 */
  get paramCount(): number {
    return this.numClasses * 3
  }
}
