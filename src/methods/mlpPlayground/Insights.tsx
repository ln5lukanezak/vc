import { InfoBox, SectionHeading, Prose } from '../../components/Controls'

export function Insights() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Try These Experiments</SectionHeading>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          1. Spiral needs ≥2 hidden layers — try 1 layer and watch it fail
        </p>
        <Prose>
          <p>
            Select the <strong>Spiral (2-class)</strong> dataset. In the Visualization
            tab, remove all but one hidden layer (8 neurons). Hit <em>Train</em> and
            let it run for several hundred epochs. The network can only carve straight
            or mildly curved lines — the spiral arms will never be separated cleanly.
          </p>
          <p>
            Now add a second hidden layer and reset. Within tens of epochs, the boundary
            starts bending and chasing the spiral arms. This is the power of depth:
            the first layer detects local directional features, the second composes
            them into curves.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          2. Increase dropout to fight overfitting on noisy moons
        </p>
        <Prose>
          <p>
            Switch to <strong>Two Moons</strong> with <strong>noise = 0.35</strong>
            and a large network (2 layers × 16 neurons). Without dropout, the decision
            boundary will jag wildly around the noisy points — 100% training accuracy
            but a fragile boundary.
          </p>
          <p>
            Set <strong>dropout = 0.4</strong> and reset. The boundary becomes much
            smoother; the network is forced to find the underlying moon structure rather
            than memorizing individual noisy points. Compare the neuron activation maps
            — they look less "noisy" too.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          3. Tanh vs. ReLU on the spiral
        </p>
        <Prose>
          <p>
            Use <strong>Spiral (2-class)</strong>, 2 layers × 8 neurons, Adam optimizer
            at lr = 0.01. Train with <strong>ReLU</strong> — notice how quickly the
            boundary starts forming sharp angular segments that approximate the curve.
          </p>
          <p>
            Reset and switch to <strong>Tanh</strong>. The boundary forms more smoothly
            (Tanh is differentiable everywhere), but convergence is often slower for the
            same learning rate. Per-neuron activation maps with Tanh show smooth
            gradients vs. ReLU's piece-wise flat regions.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          4. Optimizer showdown: SGD vs. Adam on XOR
        </p>
        <Prose>
          <p>
            Select <strong>XOR</strong>, 1 hidden layer × 4 neurons.
            With <strong>SGD at lr = 0.1</strong>, watch the loss curve — it may
            oscillate or converge slowly. With <strong>Adam at lr = 0.01</strong>,
            the same architecture converges much faster without any lr tuning.
          </p>
          <p>
            Try <strong>SGD at lr = 0.3</strong> — the loss likely explodes. Adam's
            adaptive per-parameter learning rates make it robust to lr choices.
            RMSProp is a middle ground: adaptive like Adam but with fewer hyperparameters.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          5. L2 regularization and the smooth boundary
        </p>
        <Prose>
          <p>
            Use <strong>Concentric Circles</strong> with 2 hidden layers × 12 neurons.
            Train without regularization — the boundary will tightly trace the data
            (watch the neuron activation maps light up in complex patterns).
          </p>
          <p>
            Reset, set <strong>L2 λ = 0.01</strong>, and retrain. The boundary becomes
            simpler and more circular — the regularizer penalizes large weights,
            discouraging the network from learning overly complex mappings. Compare
            the magnitude of activations in the neuron maps.
          </p>
        </Prose>
      </InfoBox>
    </div>
  )
}
