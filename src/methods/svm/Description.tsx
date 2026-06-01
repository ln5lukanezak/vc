import { Prose, SectionHeading, InfoBox } from '../../components/Controls'

export function Description() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>The Primal Problem</SectionHeading>
      <Prose>
        <p>
          A hard-margin SVM seeks the separating hyperplane that maximises the
          margin between the two classes. The margin width equals{' '}
          <span className="metric text-cyan-300">2/‖w‖</span>, so maximising it means
          minimising <span className="metric text-cyan-300">‖w‖²/2</span>. The
          soft-margin variant adds slack variables{' '}
          <span className="metric text-cyan-300">ξ_i ≥ 0</span> that allow points to
          violate the margin, penalised by <span className="metric text-cyan-300">C</span>:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          min ½‖w‖² + C Σ ξ_i
        </p>
        <p>
          subject to{' '}
          <span className="metric text-cyan-300">y_i(wᵀx_i+b) ≥ 1 − ξ_i</span>{' '}
          and{' '}
          <span className="metric text-cyan-300">ξ_i ≥ 0</span>.
        </p>
      </Prose>

      <SectionHeading>The Dual & Kernel Substitution</SectionHeading>
      <Prose>
        <p>
          Via Lagrangian duality, the primal problem transforms to a QP over dual
          variables <span className="metric text-cyan-300">α_i ∈ [0, C]</span>:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          max Σ α_i − ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
        </p>
        <p>
          subject to <span className="metric text-cyan-300">Σ α_i y_i = 0</span>,{' '}
          <span className="metric text-cyan-300">0 ≤ α_i ≤ C</span>.
        </p>
        <p>
          The primal weight vector{' '}
          <span className="metric text-cyan-300">w = Σ α_i y_i x_i</span> only appears
          inside dot products, so we can replace every dot product with{' '}
          <span className="metric text-cyan-300">K(x_i, x_j)</span> — the kernel trick.
          This gives non-linear boundaries at no extra cost.
        </p>
      </Prose>

      <SectionHeading>Simplified SMO Algorithm</SectionHeading>
      <Prose>
        <p>
          Sequential Minimal Optimisation (Platt 1998 / CS229 simplified version) solves
          the dual by updating two <span className="metric text-cyan-300">α</span> variables
          at a time — the smallest sub-problem that respects the equality constraint
          <span className="metric text-cyan-300"> Σ α_i y_i = 0</span>.
        </p>
        <p>
          Each pass loops over all training points. For point{' '}
          <span className="metric text-cyan-300">i</span> it checks the KKT condition:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          y_i · f(x_i) {'<'} 1 and α_i {'<'} C, or y_i · f(x_i) {'>'} 1 and α_i {'>'} 0
        </p>
        <p>
          If violated, it picks a random second index{' '}
          <span className="metric text-cyan-300">j ≠ i</span>, computes the analytic
          optimum for the pair, clips it to{' '}
          <span className="metric text-cyan-300">[L, H]</span>, and updates{' '}
          <span className="metric text-cyan-300">α_i, α_j,</span> and bias{' '}
          <span className="metric text-cyan-300">b</span>.
        </p>
      </Prose>

      <SectionHeading>Bias Update (b)</SectionHeading>
      <Prose>
        <p>
          After updating each pair, two candidate bias values{' '}
          <span className="metric text-cyan-300">b₁, b₂</span> are computed from the
          KKT conditions for <span className="metric text-cyan-300">i</span> and{' '}
          <span className="metric text-cyan-300">j</span> respectively. If{' '}
          <span className="metric text-cyan-300">α_i</span> is a free SV (strictly inside
          [0, C]) then <span className="metric text-cyan-300">b = b₁</span>; if{' '}
          <span className="metric text-cyan-300">α_j</span> is free then{' '}
          <span className="metric text-cyan-300">b = b₂</span>; otherwise{' '}
          <span className="metric text-cyan-300">b = (b₁+b₂)/2</span>.
        </p>
      </Prose>

      <SectionHeading>KKT / eta Guards</SectionHeading>
      <Prose>
        <p>
          The denominator <span className="metric text-cyan-300">η = 2K(i,j) − K(i,i) − K(j,j)</span>{' '}
          must be negative for the quadratic to have an interior maximum. If{' '}
          <span className="metric text-cyan-300">η ≥ 0</span> (degenerate kernel matrix
          row, or duplicate points) the pair is skipped. <span className="metric text-cyan-300">b</span>{' '}
          is guarded against NaN after every update. If bounds{' '}
          <span className="metric text-cyan-300">L ≥ H</span> the pair is also skipped.
        </p>
      </Prose>

      <SectionHeading>Decision Function</SectionHeading>
      <Prose>
        <p>
          Once training is complete, the decision function for a new point{' '}
          <span className="metric text-cyan-300">x</span> is:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          f(x) = Σ_i α_i y_i K(x_i, x) + b
        </p>
        <p>
          The predicted class is <span className="metric text-cyan-300">sign(f(x))</span>.
          The <strong className="text-slate-200">margin bands</strong> are at{' '}
          <span className="metric text-cyan-300">f = ±1</span>; points between them are
          inside the margin.
        </p>
      </Prose>

      <SectionHeading>Complexity</SectionHeading>
      <Prose>
        <p>
          Kernel matrix precomputation is{' '}
          <span className="metric text-cyan-300">O(n²)</span>. Each SMO pass is{' '}
          <span className="metric text-cyan-300">O(n²)</span> (each of n points does an
          O(n) f-eval). Grid evaluation for the heatmap is{' '}
          <span className="metric text-cyan-300">O(grid² × n_sv)</span>. Sample count is
          capped at 120 to keep animation smooth.
        </p>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Implementation note:</strong> No
        TensorFlow.js. The kernel matrix is precomputed once and reused every SMO
        pass. Only support vectors (<span className="metric text-indigo-300">α_i &gt; ε</span>)
        participate in the decision function and grid evaluation.
      </InfoBox>
    </div>
  )
}
