import { BlockMath, InlineMath } from 'react-katex'
import { SectionHeading } from '../../components/Controls'

// KaTeX CSS is imported once in main.tsx

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
      <SectionHeading>Model</SectionHeading>
      <div className="text-sm text-slate-400 mb-2">
        A degree-<InlineMath math="d" /> polynomial model with weights{' '}
        <InlineMath math="\mathbf{w} \in \mathbb{R}^{d+1}" />:
      </div>
      <FormulaRow
        formula="\hat{y} = \sum_{j=0}^{d} w_j \, x^j = w_0 + w_1 x + w_2 x^2 + \cdots + w_d x^d"
        annotation="Prediction: dot product of weights with polynomial features. w₀ is the bias (intercept)."
      />
      <FormulaRow
        formula="\boldsymbol{\phi}(x) = \begin{bmatrix} 1 \\ x \\ x^2 \\ \vdots \\ x^d \end{bmatrix}, \quad \hat{y} = \mathbf{w}^\top \boldsymbol{\phi}(x)"
        annotation="Feature map: scalar x lifted to a (d+1)-dimensional vector. The model is linear in these features."
      />

      <SectionHeading>Loss Function (MSE)</SectionHeading>
      <FormulaRow
        formula="\mathcal{L}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \bigl(\hat{y}_i - y_i\bigr)^2"
        annotation="Mean Squared Error — convex in w, unique global minimum."
      />

      <SectionHeading>Gradient</SectionHeading>
      <FormulaRow
        formula="\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{2}{n} \, \Phi^\top (\Phi \mathbf{w} - \mathbf{y})"
        annotation="Full-batch gradient, where Φ is the n×(d+1) design matrix with rows φ(xᵢ)."
      />

      <SectionHeading>Update Rule (Gradient Descent)</SectionHeading>
      <FormulaRow
        formula="\mathbf{w} \leftarrow \mathbf{w} - \eta \, \frac{\partial \mathcal{L}}{\partial \mathbf{w}}"
        annotation="Subtract the gradient scaled by learning rate η. Repeated until convergence."
      />

      <SectionHeading>Optimizer Variants</SectionHeading>
      <FormulaRow
        formula="v_t = \beta v_{t-1} + (1-\beta)\,g_t, \quad \mathbf{w} \leftarrow \mathbf{w} - \eta\, v_t"
        annotation="Momentum: exponential moving average of gradients reduces oscillation."
      />
      <FormulaRow
        formula="m_t = \hat{m}_t / (1 - \beta_1^t), \quad v_t = \hat{v}_t / (1 - \beta_2^t), \quad \mathbf{w} \leftarrow \mathbf{w} - \eta\, \frac{m_t}{\sqrt{v_t} + \epsilon}"
        annotation="Adam: bias-corrected first and second moment estimates. Adaptive per-parameter learning rate."
      />

      <SectionHeading>R² (Coefficient of Determination)</SectionHeading>
      <FormulaRow
        formula="R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (\bar{y} - y_i)^2}"
        annotation="R² = 1 means perfect fit; R² = 0 means no better than the mean; R² < 0 means worse than predicting the mean."
      />
    </div>
  )
}
