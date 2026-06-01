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

      <SectionHeading>Objective — Inertia (WCSS)</SectionHeading>
      <FormulaRow
        formula="J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2"
        annotation="Minimise the total squared distance of every point to its assigned centroid. Called within-cluster sum of squares (WCSS) or inertia."
      />

      <SectionHeading>Assignment Step</SectionHeading>
      <FormulaRow
        formula="c_i = \underset{k}{\operatorname{argmin}}\; \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2"
        annotation="Each point x_i is assigned to the cluster whose centroid μ_k is nearest. This partitions the plane into k Voronoi cells."
      />

      <SectionHeading>Update Step</SectionHeading>
      <FormulaRow
        formula="\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}"
        annotation="Each centroid is the mean of all points currently in its cluster. This is the unique minimiser of inertia for fixed assignments."
      />

      <SectionHeading>Convergence Guarantee</SectionHeading>
      <FormulaRow
        formula="J^{(t+1)} \leq J^{(t)}"
        annotation="Inertia is non-increasing: the assign step can only lower or maintain J (best assignment for current centroids); the update step can only lower or maintain J (mean minimises squared distances). So J strictly decreases or the algorithm has converged."
      />

      <SectionHeading>K-Means++ Seeding Probability</SectionHeading>
      <FormulaRow
        formula="P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}"
        annotation="D(x) = distance from x to its nearest already-chosen centroid. High D² → high probability of selection. Spreads initial seeds across the data."
      />
      <FormulaRow
        formula="D(\mathbf{x}) = \min_{c \in \text{chosen}} \|\mathbf{x} - \boldsymbol{\mu}_c\|"
        annotation="After placing each new centroid, D(x) is updated for all remaining points."
      />

      <SectionHeading>Elbow Heuristic</SectionHeading>
      <FormulaRow
        formula="\hat{k} = \underset{k}{\operatorname{argmax}}\; \bigl(J(k-1) - J(k)\bigr) - \bigl(J(k) - J(k+1)\bigr)"
        annotation="Informal elbow: k where the second difference of the inertia curve is largest — i.e., adding one more cluster stops helping much. In practice, visual inspection of the curve is often used."
      />

    </div>
  )
}
