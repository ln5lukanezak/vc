/**
 * svm.ts — Kernel soft-margin SVM trained via Simplified SMO (Platt/CS229).
 *
 * Binary classification only. Labels mapped to ±1 internally.
 * Kernels: linear, poly (xᵀx'+c)^d, rbf exp(−γ‖x−x'‖²).
 * Exposes runPasses(), decisionGrid(), supportVectorIndices(), reset().
 */

export type KernelType = 'linear' | 'poly' | 'rbf'

export interface SVMParams {
  C: number           // soft-margin slack penalty
  kernel: KernelType
  gamma: number       // RBF γ
  degree: number      // poly degree
  polyCoef: number    // poly coef c (fixed at 1)
  tol: number         // KKT violation tolerance
  eps: number         // alpha change threshold
}

export interface SVMState {
  passes: number
  alphaChanges: number   // changes in last pass (convergence indicator)
  converged: boolean
  svCount: number
}

const DEFAULT_PARAMS: SVMParams = {
  C: 1,
  kernel: 'rbf',
  gamma: 0.5,
  degree: 3,
  polyCoef: 1,
  tol: 1e-3,
  eps: 1e-5,
}

// ─── Kernel functions ─────────────────────────────────────────────────────────

function kernelLinear(xi: [number, number], xj: [number, number]): number {
  return xi[0] * xj[0] + xi[1] * xj[1]
}

function kernelPoly(
  xi: [number, number],
  xj: [number, number],
  degree: number,
  coef: number,
): number {
  const dot = xi[0] * xj[0] + xi[1] * xj[1]
  return Math.pow(dot + coef, degree)
}

function kernelRBF(
  xi: [number, number],
  xj: [number, number],
  gamma: number,
): number {
  const dx = xi[0] - xj[0]
  const dy = xi[1] - xj[1]
  return Math.exp(-gamma * (dx * dx + dy * dy))
}

// ─── SVM class ────────────────────────────────────────────────────────────────

export class SVM {
  private xs: Array<[number, number]> = []
  private ys: number[] = []   // ±1
  private alpha: number[] = []
  private b = 0
  private K: number[][] = []   // precomputed kernel matrix
  private n = 0

  private params: SVMParams
  private _state: SVMState = { passes: 0, alphaChanges: 0, converged: false, svCount: 0 }

  constructor(params: Partial<SVMParams> = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params }
  }

  // ── Load training data ────────────────────────────────────────────────────

  setData(xs: Array<[number, number]>, labels: number[]): void {
    // Map labels: class 0 → -1, class 1 → +1
    this.xs = xs
    this.ys = labels.map(l => l === 0 ? -1 : 1)
    this.n = xs.length
    this._precomputeKernel()
    this._initAlpha()
  }

  // ── Update params (resets α but keeps data) ───────────────────────────────

  setParams(patch: Partial<SVMParams>): void {
    this.params = { ...this.params, ...patch }
    if (
      patch.kernel !== undefined ||
      patch.gamma !== undefined ||
      patch.degree !== undefined ||
      patch.polyCoef !== undefined
    ) {
      // kernel changed — must recompute matrix
      this._precomputeKernel()
    }
    this._initAlpha()
  }

  reset(): void {
    this._precomputeKernel()
    this._initAlpha()
  }

  private _initAlpha(): void {
    this.alpha = new Array(this.n).fill(0)
    this.b = 0
    this._state = { passes: 0, alphaChanges: 0, converged: false, svCount: 0 }
  }

  private _kernelVal(i: number, j: number): number {
    const xi = this.xs[i]
    const xj = this.xs[j]
    const { kernel, gamma, degree, polyCoef } = this.params
    switch (kernel) {
      case 'linear': return kernelLinear(xi, xj)
      case 'poly':   return kernelPoly(xi, xj, degree, polyCoef)
      case 'rbf':    return kernelRBF(xi, xj, gamma)
    }
  }

  private _kernelPoint(x: [number, number], j: number): number {
    const xj = this.xs[j]
    const { kernel, gamma, degree, polyCoef } = this.params
    switch (kernel) {
      case 'linear': return kernelLinear(x, xj)
      case 'poly':   return kernelPoly(x, xj, degree, polyCoef)
      case 'rbf':    return kernelRBF(x, xj, gamma)
    }
  }

  private _precomputeKernel(): void {
    const n = this.n
    this.K = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => this._kernelVal(i, j))
    )
  }

  // ── Decision function ────────────────────────────────────────────────────

  /** f(i) using precomputed K (fast, for training) */
  private _fCache(i: number): number {
    const { alpha, ys, b, n, K } = this
    let s = 0
    for (let j = 0; j < n; j++) {
      if (alpha[j] > 0) s += alpha[j] * ys[j] * K[j][i]
    }
    return s + b
  }

  /** f(x) for an arbitrary test point (for grid eval) */
  decisionValue(x: [number, number]): number {
    const { alpha, ys, b, n } = this
    let s = 0
    for (let j = 0; j < n; j++) {
      if (alpha[j] > 0) s += alpha[j] * ys[j] * this._kernelPoint(x, j)
    }
    return s + b
  }

  // ── Simplified SMO ────────────────────────────────────────────────────────

  /**
   * Run `numPasses` full passes of Simplified SMO over all training pairs.
   * Returns the number of alpha changes in the last pass.
   */
  runPasses(numPasses: number): number {
    const { alpha, ys, n, K, params } = this
    const { C, tol, eps } = params

    let totalChanges = 0

    for (let pass = 0; pass < numPasses; pass++) {
      let alphaChangedThisPass = 0

      for (let i = 0; i < n; i++) {
        const fi = this._fCache(i)
        const Ei = fi - ys[i]

        // KKT violation check
        const violates =
          (ys[i] * Ei < -tol && alpha[i] < C) ||
          (ys[i] * Ei > tol  && alpha[i] > 0)

        if (!violates) continue

        // Choose j ≠ i randomly
        let j = i
        while (j === i) j = Math.floor(Math.random() * n)

        const fj = this._fCache(j)
        const Ej = fj - ys[j]

        const ai_old = alpha[i]
        const aj_old = alpha[j]

        // Compute bounds L and H
        let L: number, H: number
        if (ys[i] !== ys[j]) {
          L = Math.max(0, alpha[j] - alpha[i])
          H = Math.min(C, C + alpha[j] - alpha[i])
        } else {
          L = Math.max(0, alpha[i] + alpha[j] - C)
          H = Math.min(C, alpha[i] + alpha[j])
        }
        if (L >= H) continue

        // eta = 2 K(i,j) − K(i,i) − K(j,j)
        const eta = 2 * K[i][j] - K[i][i] - K[j][j]

        // Guard: eta must be < 0 for a maximum to exist in the interior
        if (eta >= 0) continue

        // Update alpha_j
        let aj_new = alpha[j] - (ys[j] * (Ei - Ej)) / eta
        aj_new = Math.min(H, Math.max(L, aj_new))

        if (Math.abs(aj_new - aj_old) < eps) continue

        alpha[j] = aj_new

        // Update alpha_i so sum stays satisfied
        alpha[i] = ai_old + ys[i] * ys[j] * (aj_old - aj_new)

        // Update bias b
        const b1 =
          this.b - Ei -
          ys[i] * (alpha[i] - ai_old) * K[i][i] -
          ys[j] * (alpha[j] - aj_old) * K[i][j]
        const b2 =
          this.b - Ej -
          ys[i] * (alpha[i] - ai_old) * K[i][j] -
          ys[j] * (alpha[j] - aj_old) * K[j][j]

        if (alpha[i] > 0 && alpha[i] < C) {
          this.b = b1
        } else if (alpha[j] > 0 && alpha[j] < C) {
          this.b = b2
        } else {
          this.b = (b1 + b2) / 2
        }

        // Guard b against NaN
        if (!isFinite(this.b)) this.b = 0

        alphaChangedThisPass++
      }

      totalChanges += alphaChangedThisPass

      // Track for state
      if (pass === numPasses - 1) {
        this._state.alphaChanges = alphaChangedThisPass
      }
    }

    this._state.passes += numPasses
    this._state.converged = this._state.alphaChanges === 0
    this._state.svCount = this.supportVectorIndices().length
    return totalChanges
  }

  // ── Grid evaluation (for heatmap) ─────────────────────────────────────────

  /**
   * Evaluate f over an (nx × ny) grid.
   * xs: array of x coords (length nx), ys: array of y coords (length ny).
   * Returns matrix[row][col] = f(xs[col], ys[row]).
   * Row 0 is the TOP (highest y) — matches canvas top-down orientation.
   */
  decisionGrid(xCoords: number[], yCoords: number[]): number[][] {
    const ny = yCoords.length
    const nx = xCoords.length
    const grid: number[][] = []

    for (let r = 0; r < ny; r++) {
      // row 0 = highest y → yCoords reversed
      const y = yCoords[ny - 1 - r]
      const row: number[] = []
      for (let c = 0; c < nx; c++) {
        const x = xCoords[c]
        const v = this.decisionValue([x, y])
        row.push(isFinite(v) ? v : 0)
      }
      grid.push(row)
    }

    return grid
  }

  // ── Support vectors ───────────────────────────────────────────────────────

  supportVectorIndices(): number[] {
    const eps = this.params.eps
    const indices: number[] = []
    for (let i = 0; i < this.n; i++) {
      if (this.alpha[i] > eps) indices.push(i)
    }
    return indices
  }

  getState(): SVMState { return { ...this._state } }
  getAlpha(): number[] { return this.alpha }
  getB(): number { return this.b }
  getN(): number { return this.n }
}
