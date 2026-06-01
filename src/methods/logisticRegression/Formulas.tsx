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

      <SectionHeading>Binary Logistic (Sigmoid) — K = 2</SectionHeading>
      <div className="text-sm text-slate-400 mb-2">
        When there are only 2 classes the softmax simplifies to the sigmoid:
      </div>
      <FormulaRow
        formula="\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^\top \mathbf{x} + b"
        annotation="Squashes a single linear score z into a probability in (0,1). p(y=1|x) = σ(z)."
      />
      <FormulaRow
        formula="\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \bigl[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\bigr]"
        annotation="Binary cross-entropy: penalises low probability on the true class. Equivalent to categorical CE for K=2."
      />

      <SectionHeading>Softmax Regression — K Classes</SectionHeading>
      <div className="text-sm text-slate-400 mb-2">
        One linear score per class, then a probability simplex via softmax:
      </div>
      <FormulaRow
        formula="z_k = \mathbf{W}_k \mathbf{x} + b_k, \quad k = 1,\ldots,K"
        annotation="K linear functions share the input x ∈ ℝ². W ∈ ℝ^(K×2), b ∈ ℝ^K."
      />
      <FormulaRow
        formula="\hat{p}_k = \frac{e^{z_k - z_{\max}}}{\displaystyle\sum_{j=1}^{K} e^{z_j - z_{\max}}}"
        annotation="Numerically stable softmax: subtract max logit before exp to prevent overflow."
      />
      <FormulaRow
        formula="\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^N \log \hat{p}_{y_i}^{(i)}"
        annotation="Categorical cross-entropy: average negative log-probability of the true class."
      />

      <SectionHeading>Gradient (per sample)</SectionHeading>
      <div className="text-sm text-slate-400 mb-2">
        The softmax + cross-entropy gradient simplifies elegantly — no chain rule
        product needed:
      </div>
      <FormulaRow
        formula="\frac{\partial \mathcal{L}}{\partial z_k} = \hat{p}_k - \mathbf{1}[k = y]"
        annotation="Predicted probability minus one-hot label. Zero when the model is perfectly confident on the true class."
      />
      <FormulaRow
        formula="\frac{\partial \mathcal{L}}{\partial \mathbf{W}_k} = \bigl(\hat{p}_k - \mathbf{1}[k=y]\bigr)\,\mathbf{x}^\top"
        annotation="Outer product of the logit gradient and the input features. Averaged over the mini-batch."
      />
      <FormulaRow
        formula="\frac{\partial \mathcal{L}}{\partial b_k} = \hat{p}_k - \mathbf{1}[k = y]"
        annotation="Bias gradient equals the logit gradient directly."
      />

      <SectionHeading>L2 Regularization</SectionHeading>
      <FormulaRow
        formula="\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2}\sum_k \|\mathbf{W}_k\|^2"
        annotation="Penalises large weights; shrinks the boundary toward the origin, reducing overfitting and bias."
      />
      <FormulaRow
        formula="\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{W}_k} = \frac{1}{B}\sum_{i \in \text{batch}} (\hat{p}_k^{(i)} - \mathbf{1}[k=y_i])\,\mathbf{x}^{(i)\top} + \lambda\,\mathbf{W}_k"
        annotation="Mini-batch average gradient plus L2 penalty. No L2 on biases."
      />

      <SectionHeading>Parameter Update</SectionHeading>
      <FormulaRow
        formula="\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla_\theta \mathcal{L}_{\text{reg}}"
        annotation="All K×3 params flattened into one vector θ, updated by the chosen optimizer (SGD/Momentum/RMSProp/Adam)."
      />

      <SectionHeading>2-Class Softmax = Sigmoid Identity</SectionHeading>
      <FormulaRow
        formula="p_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}} = \frac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)"
        annotation="With K=2, softmax collapses to the sigmoid of the score difference. Binary logistic and 2-class softmax are the same model."
      />
      <div className="text-xs text-slate-500 italic mt-2">
        Where <InlineMath math="\sigma(z) = 1/(1+e^{-z})" /> is the sigmoid / logistic function.
      </div>
    </div>
  )
}
