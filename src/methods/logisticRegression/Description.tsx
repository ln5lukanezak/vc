import { Prose, SectionHeading, InfoBox } from '../../components/Controls'

export function Description() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>The Model</SectionHeading>
      <Prose>
        <p>
          Softmax regression has a single linear layer with weight matrix{' '}
          <span className="metric text-cyan-300">W ∈ ℝ^(K×2)</span> and bias vector{' '}
          <span className="metric text-cyan-300">b ∈ ℝ^K</span> (for 2-D input, K classes).
          The full parameter count is <span className="metric text-cyan-300">K × 3</span>
          — tiny compared to an MLP. In this playground, training fits those parameters
          by gradient descent on cross-entropy loss.
        </p>
        <p>
          Because the model is one layer deep with no nonlinearity, every decision boundary
          is a straight line. For K classes there are at most K(K−1)/2 pairwise boundaries,
          all linear.
        </p>
      </Prose>

      <SectionHeading>Softmax Stability</SectionHeading>
      <Prose>
        <p>
          Naively computing{' '}
          <span className="metric text-cyan-300">exp(z_k) / Σ exp(z_j)</span> can
          overflow for large{' '}
          <span className="metric text-cyan-300">z_k</span>. The standard fix is to
          subtract the maximum logit before exponentiation:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          softmax(z)_k = exp(z_k − max z) / Σ_j exp(z_j − max z)
        </p>
        <p>
          This shifts values into a numerically safe range without changing the result,
          since the normalization constant cancels.
        </p>
      </Prose>

      <SectionHeading>Cross-Entropy Loss</SectionHeading>
      <Prose>
        <p>
          For N training examples, the average categorical cross-entropy is:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          {'L = −(1/N) Σ_i log p(y_i | x_i)'}
        </p>
        <p>
          where <span className="metric text-cyan-300">{'p(y_i | x_i) = softmax(z)_{y_i}'}</span>{' '}
          is the predicted probability of the true class. The loss{' '}
          <span className="metric text-cyan-300">−log(p)</span> is 0 when p→1 (perfect
          prediction) and diverges to +∞ when p→0. We clamp p ≥ 1e-12 to avoid
          <span className="metric text-cyan-300"> log(0)</span>.
        </p>
      </Prose>

      <SectionHeading>Gradient Derivation</SectionHeading>
      <Prose>
        <p>
          The gradient of cross-entropy + softmax w.r.t. the logit{' '}
          <span className="metric text-cyan-300">z_k</span> simplifies elegantly:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          ∂L/∂z_k = p_k − 1[k = y]  (= p_k − one_hot_k)
        </p>
        <p>
          This is the same elegant cancellation as in the MLP. The weight and bias
          gradients follow immediately:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          ∂L/∂W_k = (p_k − 1[k=y]) · x ,   ∂L/∂b_k = p_k − 1[k=y]
        </p>
        <p>
          Averaged over a mini-batch, with L2 regularization added to the weight
          gradient:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          ∂L_reg/∂W_k = (1/B) Σ_i (p_k^(i) − 1[k=y_i]) · x^(i)  +  λ · W_k
        </p>
      </Prose>

      <SectionHeading>Optimizer Integration</SectionHeading>
      <Prose>
        <p>
          Following the same pattern as the MLP playground, all{' '}
          <span className="metric text-cyan-300">K×3</span> parameters (K×2 weights + K
          biases) are <strong className="text-slate-200">flattened</strong> into a single
          number[], passed to the shared{' '}
          <span className="metric text-cyan-300">Optimizer.step(params, grads)</span>,
          and <strong className="text-slate-200">unflattened</strong> back. This reuses
          the SGD / Momentum / RMSProp / Adam implementations without modification.
        </p>
      </Prose>

      <SectionHeading>L2 Regularization</SectionHeading>
      <Prose>
        <p>
          L2 adds <span className="metric text-cyan-300">λ/2 · ‖W‖²</span> to the loss,
          penalising large weights. Its effect on the decision boundary: with a higher λ,
          the weights are pushed toward zero, making the boundary pass closer to the
          origin and rotate toward a less extreme orientation. For logistic regression
          on blobs, a high λ produces a boundary that bisects the cluster centres even
          when the data is asymmetric — a form of "shrinkage".
        </p>
      </Prose>

      <SectionHeading>Why 2-Class Softmax = Logistic (Sigmoid)</SectionHeading>
      <Prose>
        <p>
          With K=2, softmax of{' '}
          <span className="metric text-cyan-300">[z₀, z₁]</span> gives:
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          p₁ = exp(z₁) / (exp(z₀) + exp(z₁)) = 1 / (1 + exp(z₀ − z₁)) = σ(z₁ − z₀)
        </p>
        <p>
          where σ is the sigmoid function. So 2-class softmax regression and binary
          logistic regression are the same model — the weight vector is the difference
          of the two class weights.
        </p>
      </Prose>

      <SectionHeading>Failure Modes</SectionHeading>
      <Prose>
        <ul className="list-disc list-inside space-y-1">
          <li>
            <strong className="text-slate-200">Non-linearly separable classes:</strong>{' '}
            accuracy plateaus below 100% regardless of training duration. The boundary
            is the best possible straight line, but sometimes no straight line suffices.
          </li>
          <li>
            <strong className="text-slate-200">Overlapping classes:</strong> the
            uncertainty zone (where no class dominates) widens, and accuracy drops
            toward the Bayes error rate.
          </li>
          <li>
            <strong className="text-slate-200">Imbalanced classes:</strong> the model
            biases toward the majority class. The boundary shifts away from the majority.
          </li>
        </ul>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Implementation note:</strong> No TensorFlow.js
        is used. The entire model — softmax, cross-entropy, gradient computation,
        mini-batch loop — is pure TypeScript arithmetic. The probability-surface heatmap
        evaluates the model on a 60×60 grid every animation frame.
      </InfoBox>
    </div>
  )
}
