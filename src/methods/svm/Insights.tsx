import { InfoBox, SectionHeading, Prose } from '../../components/Controls'

export function Insights() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Try These Experiments</SectionHeading>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          1. Linear kernel on circles → watch it fail, then switch to RBF
        </p>
        <Prose>
          <p>
            Select the <em>Concentric circles</em> dataset, choose <em>Linear</em> kernel,
            and hit <em>Train</em>. The boundary will be a straight line that cuts through
            both rings — accuracy stuck around 50%. No matter how long you train, a line
            cannot separate an inner ring from an outer one.
          </p>
          <p>
            Now switch to <em>RBF</em> (γ ≈ 0.5–1.0) and hit <em>Reset → Train</em>.
            The surface curves into a circular boundary that perfectly separates the two
            rings. The kernel trick has implicitly mapped the data to a space where they
            are linearly separable — without ever computing that high-dimensional space.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          2. Raise C to reduce margin violations / lower C for a wider, softer margin
        </p>
        <Prose>
          <p>
            Use the <em>Overlapping blobs</em> dataset with <em>Linear</em> kernel. At the
            default <span className="metric text-cyan-300">C = 1</span> the boundary finds a
            compromise margin. Raise <span className="metric text-cyan-300">C</span> to 10 or
            20 — the SVM tries harder to classify every point correctly; the margin narrows,
            more points become support vectors, and the boundary can overfit.
          </p>
          <p>
            Lower <span className="metric text-cyan-300">C</span> to 0.1 — the soft margin
            widens, tolerating more misclassifications in favour of a broader safety buffer.
            Watch how the number of support vectors and the margin band thickness change.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          3. Increase γ on RBF and watch overfitting — tight islands around points
        </p>
        <Prose>
          <p>
            Select <em>Two moons</em> with <em>RBF</em> kernel, start with{' '}
            <span className="metric text-cyan-300">γ = 0.3</span> and train. The boundary
            smoothly follows the moon shapes.
          </p>
          <p>
            Now raise <span className="metric text-cyan-300">γ</span> to 3.0 or 5.0 and
            retrain. Each training point becomes its own tiny "island" — the surface shows
            isolated dots of each class surrounded by the opposite colour. High γ makes the
            RBF kernel very local: each support vector influences only its immediate
            neighbourhood. This is severe overfitting; the model memorises training points
            instead of learning the structure.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          4. Count support vectors — less is generally better
        </p>
        <Prose>
          <p>
            On <em>Separable blobs</em> with <em>Linear</em> kernel and moderate{' '}
            <span className="metric text-cyan-300">C</span>, after training you might see only
            3–6 support vectors (shown ringed). These are the critical boundary points;
            all other training data is irrelevant to the final boundary. This sparsity is
            what makes SVMs memory-efficient and interpretable.
          </p>
          <p>
            Switch to <em>RBF</em> with high γ — now almost every point is a support vector.
            Many SVs = complex, data-dependent boundary; few SVs = clean, generalizing
            boundary.
          </p>
        </Prose>
      </InfoBox>
    </div>
  )
}
