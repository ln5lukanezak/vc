import { Prose, SectionHeading, InfoBox } from '../../components/Controls'

export function Description() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Unsupervised Learning</SectionHeading>
      <Prose>
        <p>
          In <strong className="text-slate-200">supervised learning</strong> (logistic
          regression, SVM) every training point carries a label that tells the model what
          to predict. K-Means is <em>unsupervised</em>: there are no labels. The algorithm
          only sees coordinates and must infer the cluster structure from geometry alone.
        </p>
        <p>
          This means the "cluster numbers" produced by K-Means are arbitrary — cluster 0
          today might be called cluster 2 after a different run. What matters is which
          points end up together, not the numeric label.
        </p>
      </Prose>

      <SectionHeading>The Objective: Inertia (WCSS)</SectionHeading>
      <Prose>
        <p>
          K-Means minimises the{' '}
          <strong className="text-slate-200">within-cluster sum of squares</strong> (WCSS),
          also called inertia:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          {'J = Σ_k Σ_{x ∈ C_k} ‖x − μ_k‖²'}
        </p>
        <p>
          Every assign→update step guarantees inertia is non-increasing. Convergence is
          detected when no point changes cluster between two successive assign steps.
        </p>
      </Prose>

      <SectionHeading>K-Means++ in Detail</SectionHeading>
      <Prose>
        <p>
          After placing the first centroid uniformly at random, each subsequent centroid{' '}
          <span className="metric text-cyan-300">μ_{'{c}'}</span> is sampled from the data
          with probability proportional to the squared distance from the nearest already-placed
          centroid:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          {'P(x) ∝ min_{c already placed} ‖x − μ_c‖²'}
        </p>
        <p>
          This "spread-out" seeding means that initial centroids are unlikely to land in the
          same natural cluster. Empirically, K-Means++ achieves inertia within an{' '}
          <span className="metric text-cyan-300">O(log k)</span> factor of the global optimum,
          compared to potentially arbitrarily bad random seeds.
        </p>
      </Prose>

      <SectionHeading>Empty Clusters</SectionHeading>
      <Prose>
        <p>
          In rare cases a centroid can end up with zero assigned points (e.g. two initial
          seeds were placed inside the same dense cluster, and one "wins" all points). This
          implementation handles the empty-cluster case by re-seeding the empty centroid to
          the data point farthest from any existing centroid — a sensible recovery that
          typically recovers a meaningful partition on the next iteration.
        </p>
      </Prose>

      <SectionHeading>Comparison with Related Methods</SectionHeading>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">K-Means</div>
          <div className="text-slate-400">
            Hard assignments (each point belongs to exactly one cluster). Assumes spherical,
            equal-size clusters. Very fast.
          </div>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Gaussian Mixture (GMM)</div>
          <div className="text-slate-400">
            Soft assignments — each point has a probability for every cluster. Full covariance
            handles elongated or rotated clusters. Slower.
          </div>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">SVM Decision Regions</div>
          <div className="text-slate-400">
            Supervised: requires labels. K-Means Voronoi looks similar but is derived purely
            from centroid distances, not a margin objective.
          </div>
        </div>
      </div>

      <SectionHeading>Sensitivity to k</SectionHeading>
      <Prose>
        <p>
          Unlike GMM (which can use BIC/AIC to penalise model complexity), K-Means has no
          built-in mechanism for choosing <span className="metric text-cyan-300">k</span>.
          The elbow plot is a heuristic, not a formal test. When the true clusters are not
          well-separated the elbow is often smooth with no obvious kink — in practice, domain
          knowledge or additional criteria (e.g. silhouette score) are needed.
        </p>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Key insight:</strong> K-Means Voronoi regions
        are <em>always</em> convex polygons (intersection of half-spaces "closer to μ_k than
        to any other centroid"). This is why K-Means cannot discover ring-shaped or crescent
        clusters — the boundary between two rings is not a Voronoi edge.
      </InfoBox>
    </div>
  )
}
