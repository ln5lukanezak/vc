import { Prose, InfoBox, SectionHeading } from '../../components/Controls'

export function Overview() {
  return (
    <div className="max-w-2xl space-y-6">
      <Prose>
        <p>
          <strong className="text-slate-200">K-Means Clustering</strong> is one of the
          most widely used <em>unsupervised</em> learning algorithms. Unlike classification,
          there are no labels — the algorithm discovers structure in data purely from the
          geometry of the points, grouping them into{' '}
          <span className="metric text-cyan-300">k</span> clusters by minimising
          within-cluster squared distances.
        </p>
        <p>
          Because there is no supervisor, the algorithm can find natural groupings in data
          without any human annotation — useful for customer segmentation, image
          compression, anomaly detection, and exploratory analysis.
        </p>
      </Prose>

      <SectionHeading>Lloyd's Algorithm (Assign → Update)</SectionHeading>
      <Prose>
        <p>
          The core loop alternates two steps until convergence:
        </p>
      </Prose>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">1. Assignment Step</div>
          <div className="text-slate-400">
            Each point is assigned to its nearest centroid by Euclidean distance. This
            partitions the space into{' '}
            <span className="metric text-cyan-300">k</span> Voronoi regions.
          </div>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">2. Update Step</div>
          <div className="text-slate-400">
            Each centroid is recomputed as the mean of all points currently assigned to it.
            This moves the centroid toward the centre of its cluster.
          </div>
        </div>
      </div>
      <Prose>
        <p>
          The algorithm is guaranteed to decrease the objective (inertia) on every step and
          converge in a finite number of iterations. However, it may converge to a{' '}
          <em>local optimum</em>, not the global one — the final result depends on the
          initial centroid positions.
        </p>
      </Prose>

      <SectionHeading>Random vs. K-Means++ Initialization</SectionHeading>
      <Prose>
        <p>
          <strong className="text-slate-200">Random init</strong> places the{' '}
          <span className="metric text-cyan-300">k</span> starting centroids at randomly
          chosen data points. This is fast but can produce poor local optima when two
          initial centroids land in the same true cluster.
        </p>
        <p>
          <strong className="text-slate-200">K-Means++</strong> seeds the first centroid
          randomly, then picks each subsequent centroid proportional to its squared distance
          from the nearest already-chosen centroid. This spreads the seeds across the data,
          dramatically reducing the chance of a bad start and typically converging in fewer
          iterations.
        </p>
      </Prose>

      <SectionHeading>Choosing k — The Elbow Method</SectionHeading>
      <Prose>
        <p>
          A fundamental challenge: <strong className="text-slate-200">how many clusters?</strong>{' '}
          As <span className="metric text-cyan-300">k</span> increases, inertia always decreases
          (more clusters = tighter fit). The <em>elbow method</em> plots inertia vs. k and
          looks for the "elbow" — the point of diminishing returns where adding more clusters
          no longer substantially reduces inertia.
        </p>
      </Prose>

      <SectionHeading>Assumptions & Limitations</SectionHeading>
      <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Works well when</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Clusters are roughly spherical (isotropic)</li>
            <li>Clusters are similar in size and density</li>
            <li>k is known or can be estimated by elbow</li>
            <li>Large datasets (fast linear-time steps)</li>
          </ul>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Fails or struggles when</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Clusters are elongated / anisotropic</li>
            <li>Clusters have very different sizes/densities</li>
            <li>Non-convex shapes (rings, moons)</li>
            <li>k is unknown and elbow is unclear</li>
          </ul>
        </div>
      </div>

      <InfoBox>
        <strong className="text-indigo-300">Quick demo:</strong> Set{' '}
        <span className="metric text-cyan-300">k</span> equal to the true cluster count
        and hit <em>Run</em>. Watch the centroids drift toward the blob centres and the
        Voronoi regions snap into place. Then try{' '}
        <span className="metric text-cyan-300">k-means++</span> vs.{' '}
        <span className="metric text-cyan-300">random</span> init on a 4-cluster dataset
        to see how init affects convergence.
      </InfoBox>
    </div>
  )
}
