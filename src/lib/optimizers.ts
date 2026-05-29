// ─── Optimizer interface (generic parameter vector) ──────────────────────────

export interface Optimizer {
  name: string
  reset(): void
  step(params: number[], grads: number[]): number[]
}

// ─── SGD ─────────────────────────────────────────────────────────────────────

export interface SGDOptions {
  lr?: number
}

export class SGD implements Optimizer {
  readonly name = 'SGD'
  private lr: number

  constructor(opts: SGDOptions = {}) {
    this.lr = opts.lr ?? 0.01
  }

  reset(): void { /* stateless */ }

  setLr(lr: number): void { this.lr = lr }

  step(params: number[], grads: number[]): number[] {
    return params.map((p, i) => p - this.lr * grads[i])
  }
}

// ─── Momentum ─────────────────────────────────────────────────────────────────

export interface MomentumOptions {
  lr?: number
  momentum?: number
}

export class MomentumOptimizer implements Optimizer {
  readonly name = 'Momentum'
  private lr: number
  private beta: number
  private velocity: number[] = []

  constructor(opts: MomentumOptions = {}) {
    this.lr = opts.lr ?? 0.01
    this.beta = opts.momentum ?? 0.9
  }

  reset(): void { this.velocity = [] }

  setLr(lr: number): void { this.lr = lr }

  step(params: number[], grads: number[]): number[] {
    if (this.velocity.length !== params.length) {
      this.velocity = new Array(params.length).fill(0)
    }
    return params.map((p, i) => {
      this.velocity[i] = this.beta * this.velocity[i] + (1 - this.beta) * grads[i]
      return p - this.lr * this.velocity[i]
    })
  }
}

// ─── Adam ─────────────────────────────────────────────────────────────────────

export interface AdamOptions {
  lr?: number
  beta1?: number
  beta2?: number
  epsilon?: number
}

export class Adam implements Optimizer {
  readonly name = 'Adam'
  private lr: number
  private beta1: number
  private beta2: number
  private epsilon: number
  private m: number[] = []
  private v: number[] = []
  private t = 0

  constructor(opts: AdamOptions = {}) {
    this.lr = opts.lr ?? 0.01
    this.beta1 = opts.beta1 ?? 0.9
    this.beta2 = opts.beta2 ?? 0.999
    this.epsilon = opts.epsilon ?? 1e-8
  }

  reset(): void {
    this.m = []
    this.v = []
    this.t = 0
  }

  setLr(lr: number): void { this.lr = lr }

  step(params: number[], grads: number[]): number[] {
    this.t++
    if (this.m.length !== params.length) {
      this.m = new Array(params.length).fill(0)
      this.v = new Array(params.length).fill(0)
    }
    const { beta1, beta2, epsilon, lr, t } = this
    return params.map((p, i) => {
      this.m[i] = beta1 * this.m[i] + (1 - beta1) * grads[i]
      this.v[i] = beta2 * this.v[i] + (1 - beta2) * grads[i] ** 2
      const mHat = this.m[i] / (1 - beta1 ** t)
      const vHat = this.v[i] / (1 - beta2 ** t)
      return p - lr * mHat / (Math.sqrt(vHat) + epsilon)
    })
  }
}

// ─── RMSProp ──────────────────────────────────────────────────────────────────

export interface RMSPropOptions {
  lr?: number
  /** smoothing constant for the squared-gradient EMA (default 0.9) */
  rho?: number
  epsilon?: number
}

export class RMSProp implements Optimizer {
  readonly name = 'RMSProp'
  private lr: number
  private rho: number
  private epsilon: number
  private v: number[] = []

  constructor(opts: RMSPropOptions = {}) {
    this.lr = opts.lr ?? 0.001
    this.rho = opts.rho ?? 0.9
    this.epsilon = opts.epsilon ?? 1e-8
  }

  reset(): void { this.v = [] }

  setLr(lr: number): void { this.lr = lr }

  step(params: number[], grads: number[]): number[] {
    if (this.v.length !== params.length) {
      this.v = new Array(params.length).fill(0)
    }
    return params.map((p, i) => {
      this.v[i] = this.rho * this.v[i] + (1 - this.rho) * grads[i] ** 2
      return p - this.lr * grads[i] / (Math.sqrt(this.v[i]) + this.epsilon)
    })
  }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

export type OptimizerName = 'SGD' | 'Momentum' | 'RMSProp' | 'Adam'

export function createOptimizer(name: OptimizerName, lr: number): Optimizer {
  switch (name) {
    case 'SGD':      return new SGD({ lr })
    case 'Momentum': return new MomentumOptimizer({ lr })
    case 'RMSProp':  return new RMSProp({ lr })
    case 'Adam':     return new Adam({ lr })
  }
}
