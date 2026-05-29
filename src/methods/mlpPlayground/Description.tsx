import { Prose, SectionHeading, InfoBox } from '../../components/Controls'

export function Description() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Architecture</SectionHeading>
      <Prose>
        <p>
          An MLP consists of an <strong className="text-slate-200">input layer</strong>,
          one or more <strong className="text-slate-200">hidden layers</strong>, and an{' '}
          <strong className="text-slate-200">output layer</strong>. In this playground,
          the input is always 2-dimensional (x, y coordinates of a 2-D point), and the
          output is a probability distribution over classes via softmax.
        </p>
        <p>
          Each <em>fully connected</em> layer applies a linear transformation followed by
          a nonlinear activation. Between two layers with dimensions{' '}
          <span className="metric text-cyan-300">d_in</span> and{' '}
          <span className="metric text-cyan-300">d_out</span>, there are{' '}
          <span className="metric text-cyan-300">d_in × d_out</span> weights plus{' '}
          <span className="metric text-cyan-300">d_out</span> biases.
        </p>
      </Prose>

      <SectionHeading>Forward Pass</SectionHeading>
      <Prose>
        <p>
          For each layer <em>l</em>:
        </p>
        <ol className="list-decimal list-inside space-y-1">
          <li>Compute pre-activation: <span className="metric text-cyan-300">z = W·a + b</span></li>
          <li>Apply activation: <span className="metric text-cyan-300">a = φ(z)</span> (hidden layers)</li>
          <li>Output layer uses softmax instead of φ to get class probabilities.</li>
        </ol>
        <p>
          The pre-activations <em>z</em> and activations <em>a</em> for every layer are
          cached during the forward pass — they are needed by backprop.
        </p>
      </Prose>

      <SectionHeading>Loss Function: Cross-Entropy</SectionHeading>
      <Prose>
        <p>
          The network minimizes <strong className="text-slate-200">categorical
          cross-entropy</strong>:
        </p>
        <p className="metric text-cyan-300 text-center py-2">
          L = −(1/N) Σᵢ log p(yᵢ | xᵢ)
        </p>
        <p>
          For a correct label <em>k</em>, this reduces to{' '}
          <span className="metric text-cyan-300">−log(pₖ)</span>. Minimizing it pushes
          the softmax probability for the true class toward 1.
        </p>
      </Prose>

      <SectionHeading>Backpropagation</SectionHeading>
      <Prose>
        <p>
          Starting from the output, we compute{' '}
          <span className="metric text-cyan-300">∂L/∂W</span> and{' '}
          <span className="metric text-cyan-300">∂L/∂b</span> for each layer using the
          chain rule:
        </p>
        <ol className="list-decimal list-inside space-y-1">
          <li>
            For the output layer (softmax + cross-entropy), the gradient simplifies
            elegantly to <span className="metric text-cyan-300">δ = p − one_hot(y)</span>.
          </li>
          <li>
            For each hidden layer: propagate{' '}
            <span className="metric text-cyan-300">δ_prev = Wᵀ δ ⊙ φ′(z)</span>
            where ⊙ is elementwise multiply and φ′ is the activation derivative.
          </li>
          <li>
            Weight gradient: <span className="metric text-cyan-300">∂L/∂W = δ·aᵀ</span>,
            bias gradient: <span className="metric text-cyan-300">∂L/∂b = δ</span>.
          </li>
        </ol>
      </Prose>

      <SectionHeading>Mini-Batch Training</SectionHeading>
      <Prose>
        <p>
          Rather than computing gradients on the full dataset (batch GD) or a single
          sample (SGD), <strong className="text-slate-200">mini-batch</strong> training
          averages gradients over a small subset (e.g. 32 points) per parameter update.
          This provides a good balance: lower variance than pure SGD, lower memory and
          faster iteration than full batch.
        </p>
      </Prose>

      <SectionHeading>Dropout</SectionHeading>
      <Prose>
        <p>
          During each forward pass in training, each hidden activation is independently
          set to 0 with probability <em>p</em> (the dropout rate). Surviving activations
          are scaled up by <span className="metric text-cyan-300">1/(1−p)</span> to keep
          expected output magnitude the same. At inference, dropout is disabled — the
          full network is used.
        </p>
      </Prose>

      <SectionHeading>Failure Modes</SectionHeading>
      <Prose>
        <ul className="list-disc list-inside space-y-1">
          <li>
            <strong className="text-slate-200">Exploding gradients:</strong> lr too high →
            weights diverge; fix: reduce learning rate.
          </li>
          <li>
            <strong className="text-slate-200">Underfitting:</strong> network too small for
            the task, or training stopped too early.
          </li>
          <li>
            <strong className="text-slate-200">Vanishing gradients:</strong> Sigmoid/Tanh
            in deep nets; ReLU largely solves this.
          </li>
          <li>
            <strong className="text-slate-200">Overfitting:</strong> noisy boundary that
            memorizes training data; use dropout or L2.
          </li>
        </ul>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Implementation note:</strong> All weights
        and biases across all layers are flattened into a single parameter vector,
        passed to the shared <span className="metric">Optimizer.step()</span>, then
        unflattened back. This cleanly reuses the same optimizer interface across
        all methods in ML Explorer.
      </InfoBox>
    </div>
  )
}
