import { BlockMath } from 'react-katex'
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

      <SectionHeading>Primal (soft-margin)</SectionHeading>
      <FormulaRow
        formula="\min_{\mathbf{w},b,\boldsymbol{\xi}} \;\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}\xi_i"
        annotation="Minimise margin width ½‖w‖² (= maximise margin 2/‖w‖) plus slack penalty C·Σξ."
      />
      <FormulaRow
        formula="\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \geq 1 - \xi_i,\quad \xi_i \geq 0 \;\forall i"
        annotation="Each point must be on the correct side of its margin, with slack ξ_i ≥ 0 for violations."
      />

      <SectionHeading>Dual (Lagrangian)</SectionHeading>
      <FormulaRow
        formula="\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i,\mathbf{x}_j)"
        annotation="Dual QP: maximise over α ∈ [0, C]^N, with the equality constraint Σ α_i y_i = 0."
      />
      <FormulaRow
        formula="\text{s.t.}\quad \sum_{i=1}^N \alpha_i y_i = 0,\quad 0 \leq \alpha_i \leq C"
        annotation="Box constraint keeps α in [0, C]. Points with α_i > 0 are support vectors."
      />

      <SectionHeading>Kernel Definitions</SectionHeading>
      <FormulaRow
        formula="K_{\text{lin}}(\mathbf{x},\mathbf{x}') = \mathbf{x}^\top\mathbf{x}'"
        annotation="Linear kernel — dot product in input space. Gives a linear boundary."
      />
      <FormulaRow
        formula="K_{\text{poly}}(\mathbf{x},\mathbf{x}') = (\mathbf{x}^\top\mathbf{x}' + 1)^d"
        annotation="Polynomial kernel of degree d. Implicitly maps to all monomials up to degree d."
      />
      <FormulaRow
        formula="K_{\text{rbf}}(\mathbf{x},\mathbf{x}') = \exp\!\bigl(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2\bigr)"
        annotation="Radial Basis Function (Gaussian) kernel. γ controls locality — high γ = narrow influence."
      />

      <SectionHeading>Decision Function</SectionHeading>
      <FormulaRow
        formula="f(\mathbf{x}) = \sum_{i=1}^N \alpha_i\, y_i\, K(\mathbf{x}_i, \mathbf{x}) + b"
        annotation="Predicted class = sign(f). Only support vectors (α_i > 0) contribute. Margin boundaries at f = ±1."
      />
      <FormulaRow
        formula="\hat{y} = \operatorname{sign}(f(\mathbf{x}))"
        annotation="+1 if f > 0 (class 1), −1 if f < 0 (class 0). f = 0 is the decision boundary."
      />

      <SectionHeading>SMO Update Rule</SectionHeading>
      <FormulaRow
        formula="\eta = 2K_{ij} - K_{ii} - K_{jj}"
        annotation="Second derivative of the objective w.r.t. α_j. Must be < 0 for a valid interior maximum."
      />
      <FormulaRow
        formula="\alpha_j^{\text{new}} = \operatorname{clip}\!\Bigl(\alpha_j - \frac{y_j(E_i - E_j)}{\eta},\; L,\; H\Bigr)"
        annotation="Analytic update for α_j, clipped to feasible box [L, H]. E_i = f(x_i) − y_i."
      />
      <FormulaRow
        formula="\alpha_i^{\text{new}} = \alpha_i + y_i y_j(\alpha_j^{\text{old}} - \alpha_j^{\text{new}})"
        annotation="α_i adjusted to satisfy the equality constraint Σ α_k y_k = 0."
      />
    </div>
  )
}
