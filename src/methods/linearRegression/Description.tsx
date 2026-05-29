import { Prose, SectionHeading, InfoBox } from '../../components/Controls'

export function Description() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>The Model</SectionHeading>
      <Prose>
        <p>
          For polynomial regression of degree <em>d</em>, we build a feature
          vector from a scalar input x:
        </p>
        <div className="metric bg-slate-800 rounded px-3 py-2 text-cyan-300 text-sm my-2">
          φ(x) = [1, x, x², …, x^d]
        </div>
        <p>
          The prediction is the dot product of learned weights{' '}
          <span className="metric text-cyan-300">w ∈ ℝ^(d+1)</span> with this
          feature vector:
        </p>
        <div className="metric bg-slate-800 rounded px-3 py-2 text-cyan-300 text-sm my-2">
          ŷ = w · φ(x) = w₀ + w₁x + w₂x² + … + wdx^d
        </div>
        <p>
          Note: we fold the bias term <em>b</em> into <em>w₀</em> (the
          coefficient of x⁰ = 1). This is the standard "augmented" formulation.
        </p>
      </Prose>

      <SectionHeading>Loss Function</SectionHeading>
      <Prose>
        <p>
          We minimize the{' '}
          <strong className="text-slate-200">Mean Squared Error (MSE)</strong>:
        </p>
        <div className="metric bg-slate-800 rounded px-3 py-2 text-cyan-300 text-sm my-2">
          L(w) = (1/n) Σ (ŷᵢ − yᵢ)²
        </div>
        <p>
          MSE penalizes large residuals quadratically, is convex in{' '}
          <em>w</em>, and has a unique global minimum.
        </p>
      </Prose>

      <SectionHeading>Gradient Descent</SectionHeading>
      <Prose>
        <p>
          The gradient of MSE w.r.t. the weight vector is:
        </p>
        <div className="metric bg-slate-800 rounded px-3 py-2 text-cyan-300 text-sm my-2">
          ∂L/∂w = (2/n) Φᵀ(Φw − y)
        </div>
        <p>
          where <span className="metric text-cyan-300">Φ</span> is the design
          matrix (each row is φ(xᵢ)). The update rule for each step is:
        </p>
        <div className="metric bg-slate-800 rounded px-3 py-2 text-cyan-300 text-sm my-2">
          w ← w − η · ∂L/∂w
        </div>
        <p>
          with learning rate <em>η</em>. The optimizer implementations
          (SGD / Momentum / Adam) modify exactly how this update is applied.
        </p>
      </Prose>

      <SectionHeading>Feature Standardization (important!)</SectionHeading>
      <Prose>
        <p>
          For degree &gt; 1, the polynomial features can differ by many orders of
          magnitude (e.g., x ≈ 2 gives x⁵ ≈ 32). Without standardization,
          gradient descent is extremely ill-conditioned: a step size that works
          for x⁵ will explode for x.
        </p>
        <p>
          We therefore{' '}
          <strong className="text-slate-200">z-score each feature column</strong>{' '}
          before training (subtract column mean, divide by column std), then
          transform the learned weights back to the original scale for plotting.
        </p>
      </Prose>

      <SectionHeading>Assumptions and Failure Modes</SectionHeading>
      <Prose>
        <ul className="list-disc list-inside space-y-1">
          <li>
            <strong className="text-slate-200">Linearity:</strong> the model cannot
            capture truly non-linear relationships (e.g., periodicity) without
            appropriate basis functions.
          </li>
          <li>
            <strong className="text-slate-200">Outliers:</strong> MSE squares
            residuals, so a single outlier can dominate the loss.
          </li>
          <li>
            <strong className="text-slate-200">Overfitting:</strong> a high-degree
            polynomial can interpolate the training set perfectly while
            generalizing poorly. Ridge/Lasso regularization mitigate this.
          </li>
          <li>
            <strong className="text-slate-200">Learning rate:</strong> too large →
            the loss diverges; too small → training is very slow. Adam adapts the
            rate per parameter and is usually most robust.
          </li>
        </ul>
      </Prose>

      <InfoBox>
        The visualization runs <strong className="text-indigo-300">full-batch</strong>{' '}
        gradient descent (all points in each step). For noisy or large datasets,
        mini-batch SGD would be more efficient but would show more variance in the
        loss curve.
      </InfoBox>
    </div>
  )
}
