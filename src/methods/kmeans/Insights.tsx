import { InfoBox, SectionHeading, Prose } from '../../components/Controls'

export function Insights() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Try These Experiments</SectionHeading>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          1. Set k below or above the true cluster count — watch inertia and the partition
        </p>
        <Prose>
          <p>
            Generate data with <strong>True clusters = 4</strong> and good separation (≥ 1.2).
            Set <span className="metric text-cyan-300">k = 4</span> and run — the centroids
            should find the four blobs cleanly, inertia drops quickly, and the Voronoi
            regions match the natural structure.
          </p>
          <p>
            Now try <span className="metric text-cyan-300">k = 2</span> (under-clustering).
            Two large, merged regions appear; inertia is much higher because distant points
            share a centroid. Then try <span className="metric text-cyan-300">k = 7</span>{' '}
            (over-clustering): blobs get split into multiple sub-clusters, inertia decreases
            further but the partition no longer matches meaningful groups.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          2. Compare random vs. K-Means++ init on well-separated blobs
        </p>
        <Prose>
          <p>
            With 4 or more clusters, switch init to <strong>Random</strong> and hit{' '}
            <strong>Reset</strong> several times (each reset uses a new random seed). You
            will occasionally see two initial centroids land inside the same blob — the
            algorithm converges to a suboptimal partition where one true cluster is split
            and another is merged. Inertia is higher than necessary.
          </p>
          <p>
            Switch to <strong>K-Means++</strong> and repeat. The D²-weighted seeding
            actively avoids placing two initial centroids in the same neighbourhood, so it
            almost always converges to the natural partition with lower inertia and in fewer
            iterations.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          3. Find the elbow and compare it to the true k
        </p>
        <Prose>
          <p>
            Set <strong>True clusters = 3</strong> and click <strong>Compute Elbow</strong>.
            The inertia curve should show a sharp bend around k = 3 — the elbow — where
            adding one more cluster stops reducing inertia dramatically.
          </p>
          <p>
            Now increase <strong>Noise</strong> to 0.4 and recompute. With overlapping
            blobs the elbow often becomes a gradual curve with no obvious kink — in
            practice this is the hard problem of model selection for K-Means. Compare the
            elbow with the true count you set; as noise grows they diverge.
          </p>
          <p>
            The highlighted dot on the elbow chart marks your currently selected{' '}
            <span className="metric text-cyan-300">k</span> — use it to see how far your
            chosen k sits from the theoretical optimum.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          4. Watch the centroid trails — large first jump, tiny final adjustments
        </p>
        <Prose>
          <p>
            Hit <strong>Step</strong> repeatedly instead of Run. On the first few
            iterations the dashed trail arrows are long as centroids migrate across the
            canvas toward the blob centres. By iteration 4–6 the arrows shrink to almost
            nothing — the centroids are oscillating near their final positions.
          </p>
          <p>
            This mirrors the inertia curve: big drops early, nearly flat near convergence.
            Lloyd's algorithm is doing the most work in the first few iterations.
          </p>
        </Prose>
      </InfoBox>
    </div>
  )
}
