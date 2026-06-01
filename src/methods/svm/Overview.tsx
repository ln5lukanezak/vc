import { Prose, InfoBox, SectionHeading } from '../../components/Controls'

export function Overview() {
  return (
    <div className="max-w-2xl space-y-6">
      <Prose>
        <p>
          <strong className="text-slate-200">Support Vector Machines</strong> find the{' '}
          <em>maximum-margin</em> decision boundary: the hyperplane that keeps the
          nearest training points (the <strong className="text-slate-200">support
          vectors</strong>) as far apart as possible. That margin is the safety
          buffer — large margins generalize better than boundaries that hug the data.
        </p>
        <p>
          The <strong className="text-slate-200">soft-margin</strong> extension (parameter{' '}
          <span className="metric text-cyan-300">C</span>) allows a controlled number of
          points to sit inside or even on the wrong side of the margin. High{' '}
          <span className="metric text-cyan-300">C</span> = tight fit (small or no violations
          allowed); low <span className="metric text-cyan-300">C</span> = wider, softer margin
          that tolerates more overlap.
        </p>
      </Prose>

      <SectionHeading>The Kernel Trick</SectionHeading>
      <Prose>
        <p>
          A linear SVM draws straight boundaries — useless for circles or spirals. The{' '}
          <strong className="text-slate-200">kernel trick</strong> fixes this without ever
          explicitly computing a high-dimensional feature map. We replace every dot product{' '}
          <span className="metric text-cyan-300">xᵀx'</span> with a{' '}
          <strong className="text-slate-200">kernel function</strong>{' '}
          <span className="metric text-cyan-300">K(x, x')</span> that implicitly computes{' '}
          <span className="metric text-cyan-300">φ(x)ᵀφ(x')</span> in some (possibly
          infinite-dimensional) feature space — at the cost of just evaluating the kernel.
        </p>
      </Prose>

      <SectionHeading>Kernels Available</SectionHeading>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Linear</div>
          <div className="metric text-cyan-300 mb-1">K = xᵀx'</div>
          <div className="text-slate-400">Straight-line boundary. Fast, interpretable. Use when classes are (nearly) linearly separable.</div>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Polynomial</div>
          <div className="metric text-cyan-300 mb-1">K = (xᵀx' + 1)ᵈ</div>
          <div className="text-slate-400">Curved polynomial boundary of degree d. Intermediate flexibility. Sensitive to scaling.</div>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">RBF / Gaussian</div>
          <div className="metric text-cyan-300 mb-1">K = exp(−γ‖x−x'‖²)</div>
          <div className="text-slate-400">Infinitely dimensional feature space. Handles any shape. γ controls locality — high γ = tight clusters.</div>
        </div>
      </div>

      <SectionHeading>Support Vectors</SectionHeading>
      <Prose>
        <p>
          Only the training points whose dual variable{' '}
          <span className="metric text-cyan-300">α_i &gt; 0</span> matter — they are the
          support vectors, the "load-bearing" points of the boundary. All other training
          points could be removed and the solution would not change. Fewer SVs usually means
          a cleaner, better-generalizing boundary; many SVs (especially at soft-margin with
          high <span className="metric text-cyan-300">C</span>) means the boundary is heavily
          data-dependent.
        </p>
      </Prose>

      <SectionHeading>When to Use Which Kernel</SectionHeading>
      <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Linear kernel</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Linearly separable blobs</li>
            <li>High-dimensional, sparse features (text)</li>
            <li>When speed matters most</li>
          </ul>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">RBF kernel</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Circles, moons, non-linear shapes</li>
            <li>Default choice for unknown data</li>
            <li>Tune γ + C carefully to avoid overfit</li>
          </ul>
        </div>
      </div>

      <InfoBox>
        <strong className="text-indigo-300">Quick demo:</strong> Select the{' '}
        <em>Circles</em> dataset, use a <em>linear</em> kernel and train — it
        will fail (the boundary can only be a line). Switch to <em>RBF</em> and
        watch the curved boundary emerge and correctly separate the rings.
      </InfoBox>
    </div>
  )
}
