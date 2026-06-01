import { InfoBox, SectionHeading, Prose } from '../../components/Controls'

export function Insights() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Try These Experiments</SectionHeading>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          1. Lower separation until classes overlap — watch accuracy drop
        </p>
        <Prose>
          <p>
            Start with 2 classes, separation = 2.0, noise = 0.2. Hit{' '}
            <em>Train</em> and let accuracy reach ~100%. The probability surface
            shows a sharp boundary with confident blue/green zones.
          </p>
          <p>
            Now lower separation to 0.6 and hit <em>Regen data + Reset</em>.
            Retrain. The boundary is still a straight line (it always will be),
            but accuracy plateaus below 100% because no line can correctly separate
            the overlapping blobs. The grey uncertainty zone widens — the model
            knows it is unsure near the boundary.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          2. Raise λ and watch the boundary simplify and "shrink"
        </p>
        <Prose>
          <p>
            Use 2 classes, separation = 1.5, then set{' '}
            <strong>L2 λ = 0.3</strong> and train. Compare with λ = 0.
          </p>
          <p>
            With high λ the weights are penalised for being large. The decision
            boundary is forced to pass closer to the origin and at a less extreme
            angle — it "shrinks" toward the midpoint between the two classes
            regardless of the data asymmetry. Accuracy may drop slightly but the
            boundary is more robust to new data.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          3. Logistic can't curve — try a non-linearly-separable layout to see it fail
        </p>
        <Prose>
          <p>
            Set 2 classes, separation = 0.5, noise = 0.05. Regenerate until you
            get two tightly interleaved blobs. Train for 500+ epochs. The accuracy
            will be stuck around 50–70% — the best straight line just can't do better.
          </p>
          <p>
            This is exactly the motivation for the{' '}
            <strong className="text-slate-200">MLP Playground</strong> and{' '}
            <strong className="text-slate-200">Support Vector Machine</strong> (with
            RBF kernel): both can learn curved, non-linear boundaries that logistic
            regression fundamentally cannot.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          4. Scale to 4 classes — see the probability surface tile into regions
        </p>
        <Prose>
          <p>
            Set classes = 4, separation = 1.5, noise = 0.2 and train with Adam.
            The probability surface divides the 2-D plane into 4 "Voronoi-like"
            regions separated by straight lines. Hover mentally over any region —
            its colour tells you the predicted class.
          </p>
          <p>
            The confusion matrix now shows a 4×4 grid. Diagonal cells (correct
            predictions) glow indigo; off-diagonal cells (misclassifications) show
            in red. Lower the separation to see the off-diagonal cells fill up as
            nearby classes start bleeding into each other.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          5. Optimizer shootout: SGD vs. Adam on tight blobs
        </p>
        <Prose>
          <p>
            Use 3 classes, separation = 0.8, noise = 0.3. With{' '}
            <strong>SGD at lr = 0.01</strong> watch the loss curve oscillate.
            Switching to <strong>Adam at lr = 0.05</strong> gives dramatically
            faster convergence with less tuning — Adam's per-parameter adaptive
            learning rates handle the different scales of the weight updates
            automatically.
          </p>
        </Prose>
      </InfoBox>
    </div>
  )
}
