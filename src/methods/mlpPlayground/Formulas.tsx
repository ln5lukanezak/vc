import { BlockMath, InlineMath } from 'react-katex'
import { SectionHeading } from '../../components/Controls'

interface FormulaRowProps {
  formula: string
  annotation: string
}

function FormulaRow({ formula, annotation }: FormulaRowProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center gap-2 py-3 border-b border-slate-700/50 last:border-0">
      <div className="flex-1 overflow-x-auto">
        <BlockMath math={formula} />
      </div>
      <div className="text-xs text-slate-500 sm:w-64 shrink-0 italic">
        {annotation}
      </div>
    </div>
  )
}

export function Formulas() {
  return (
    <div className="max-w-2xl space-y-8">

      <SectionHeading>Forward Pass (layer l)</SectionHeading>
      <div className="text-sm text-slate-400 mb-2">
        Given the input activation{' '}
        <InlineMath math="\mathbf{a}^{(l-1)}" />{' '}
        from the previous layer:
      </div>
      <FormulaRow
        formula="\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}"
        annotation="Pre-activation: linear transform. W^(l) has shape [d_out × d_in]."
      />
      <FormulaRow
        formula="\mathbf{a}^{(l)} = \varphi\!\left(\mathbf{z}^{(l)}\right)"
        annotation="Post-activation: element-wise nonlinearity φ (ReLU, Tanh, etc.) applied to hidden layers."
      />

      <SectionHeading>Activation Functions</SectionHeading>
      <FormulaRow
        formula="\text{ReLU}(z) = \max(0,\, z), \quad \text{ReLU}'(z) = \mathbf{1}[z > 0]"
        annotation="Rectified Linear Unit — most popular; avoids vanishing gradient for z > 0."
      />
      <FormulaRow
        formula="\text{LeakyReLU}(z) = \begin{cases} z & z \ge 0 \\ 0.01\,z & z < 0 \end{cases}"
        annotation="Prevents 'dead neurons' by allowing a small gradient for negative z."
      />
      <FormulaRow
        formula="\text{Tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \text{Tanh}'(z) = 1 - \tanh^2(z)"
        annotation="Zero-centered, smooth; gradients can vanish in very deep nets."
      />

      <SectionHeading>Output Layer — Softmax</SectionHeading>
      <FormulaRow
        formula="\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1,\ldots,K"
        annotation="Converts raw scores z into a probability distribution over K classes. Numerically stable with max subtraction."
      />

      <SectionHeading>Loss Function — Categorical Cross-Entropy</SectionHeading>
      <FormulaRow
        formula="\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}_{y_i}^{(i)}"
        annotation="Average negative log-probability of the true class. Minimizing this pushes p(true class) → 1."
      />

      <SectionHeading>Backpropagation (chain rule)</SectionHeading>
      <FormulaRow
        formula="\boldsymbol{\delta}^{(L)} = \hat{\mathbf{p}} - \mathbf{e}_{y}"
        annotation="Output layer delta: softmax probability minus one-hot label. The elegant cancellation of softmax + cross-entropy."
      />
      <FormulaRow
        formula="\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)}\right)^\top \boldsymbol{\delta}^{(l+1)} \odot \varphi'\!\left(\mathbf{z}^{(l)}\right)"
        annotation="Hidden layer delta: propagate error backward through weights, then gate by activation derivative (⊙ = element-wise product)."
      />
      <FormulaRow
        formula="\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} \left(\mathbf{a}^{(l-1)}\right)^\top + \lambda\, \mathbf{W}^{(l)}"
        annotation="Weight gradient: outer product of delta and input activations, plus L2 regularization term λW."
      />
      <FormulaRow
        formula="\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}"
        annotation="Bias gradient equals the delta vector directly."
      />

      <SectionHeading>Gradient Descent Update</SectionHeading>
      <FormulaRow
        formula="\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}"
        annotation="All parameters θ (weights + biases) are flattened into one vector, updated by the chosen optimizer (SGD / Momentum / RMSProp / Adam)."
      />

      <SectionHeading>L2 Regularization</SectionHeading>
      <FormulaRow
        formula="\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{l} \|\mathbf{W}^{(l)}\|_F^2"
        annotation="Penalizes large weights; pushes the network toward smoother, more generalizable decision boundaries."
      />

      <SectionHeading>Dropout</SectionHeading>
      <FormulaRow
        formula="\tilde{\mathbf{a}}^{(l)} = \frac{1}{1-p}\,\mathbf{m} \odot \mathbf{a}^{(l)}, \quad m_j \sim \text{Bernoulli}(1-p)"
        annotation="Each neuron is independently masked with probability p during training. Scaled by 1/(1−p) so expected value is unchanged. Disabled at inference."
      />
    </div>
  )
}
