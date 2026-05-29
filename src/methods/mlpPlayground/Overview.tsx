import { Prose, InfoBox, SectionHeading } from '../../components/Controls'

export function Overview() {
  return (
    <div className="max-w-2xl space-y-6">
      <Prose>
        <p>
          A <strong className="text-slate-200">Multilayer Perceptron (MLP)</strong> is
          the foundational deep learning architecture: a stack of fully-connected layers
          that transforms raw inputs through successive nonlinear mappings until reaching
          a final classification or regression output.
        </p>
        <p>
          Each layer learns to extract progressively more abstract features. The first
          hidden layer might detect simple linear boundaries; deeper layers compose
          those into curves, loops, and spirals. This is the essence of{' '}
          <em>representation learning</em>.
        </p>
      </Prose>

      <SectionHeading>Hidden Layers &amp; Nonlinearity</SectionHeading>
      <Prose>
        <p>
          Without nonlinear activation functions, stacking layers would achieve nothing
          beyond a single linear transform — any composition of linear maps is still
          linear. Activations like <strong className="text-slate-200">ReLU</strong>{' '}
          (Rectified Linear Unit) or <strong className="text-slate-200">Tanh</strong>{' '}
          break this, letting the network carve non-linear decision boundaries.
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li><span className="metric text-cyan-300">ReLU(z) = max(0, z)</span> — fast, sparse, no vanishing gradient for positive z. Default choice.</li>
          <li><span className="metric text-cyan-300">LeakyReLU</span> — small negative slope prevents "dead neurons".</li>
          <li><span className="metric text-cyan-300">Tanh</span> — smooth, zero-centered; slower to converge but sometimes better for shallow nets.</li>
          <li><span className="metric text-cyan-300">Sigmoid</span> — squashes to (0,1); mainly historical, prone to vanishing gradients.</li>
        </ul>
      </Prose>

      <SectionHeading>Why Depth and Width Matter</SectionHeading>
      <Prose>
        <p>
          <strong className="text-slate-200">Width</strong> (neurons per layer) controls
          how many distinct patterns a layer can represent simultaneously.{' '}
          <strong className="text-slate-200">Depth</strong> (number of layers) lets the
          network compose simpler features into complex ones. Empirically, depth is more
          parameter-efficient than width for complex tasks — a spiral that requires
          ≥2 layers cannot be learned with a 1-layer net regardless of width.
        </p>
      </Prose>

      <SectionHeading>Backpropagation Intuition</SectionHeading>
      <Prose>
        <p>
          Backprop applies the <em>chain rule</em> from calculus: starting from the
          loss gradient at the output, it propagates error signals layer-by-layer
          backward through the network, computing how much each weight contributed
          to the loss. Those partial derivatives then guide the optimizer to nudge
          each weight in the direction that reduces the loss.
        </p>
      </Prose>

      <SectionHeading>Overfitting &amp; Regularization</SectionHeading>
      <Prose>
        <p>
          A large MLP can memorize the training data exactly — achieving 0% training
          error while failing on new points. Two key remedies:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>
            <strong className="text-slate-200">L2 regularization (weight decay):</strong>{' '}
            adds <span className="metric text-cyan-300">λ‖W‖²</span> to the loss,
            penalizing large weights and encouraging smoother decision boundaries.
          </li>
          <li>
            <strong className="text-slate-200">Dropout:</strong> randomly zeros out
            neurons during training (with rate <em>p</em>), forcing the network to
            learn redundant representations. Disabled at inference.
          </li>
        </ul>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Quick demo:</strong> Select the{' '}
        <em>spiral</em> dataset, use 2 hidden layers with 8 neurons each, and hit{' '}
        <em>Train</em>. Watch the decision boundary morph from a straight line to
        an intricate spiral-following curve. Then try 1 layer — it can't do it!
      </InfoBox>

      <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Pros</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Universal approximator</li>
            <li>Learns complex boundaries</li>
            <li>Flexible architecture</li>
            <li>Scales with data</li>
          </ul>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Cons</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Many hyperparameters</li>
            <li>Prone to overfitting</li>
            <li>Training is iterative</li>
            <li>Black-box (not interpretable)</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
