import { InfoBox, SectionHeading, Prose } from '../../components/Controls'

export function Insights() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Try These Experiments</SectionHeading>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          1. Overfitting: model degree &gg; true degree
        </p>
        <Prose>
          <p>
            In <em>Dataset</em>, set <strong>true degree = 2</strong>. Then in{' '}
            <em>Visualization</em>, train with <strong>model degree = 8</strong>.
            Watch the curve twist wildly between data points — it memorizes the
            noise. MSE will look low, but the fit is useless outside the training
            range.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          2. Underfitting: model degree &lt; true degree
        </p>
        <Prose>
          <p>
            Set <strong>true degree = 4</strong> and train with{' '}
            <strong>model degree = 1</strong>. The line can never capture the
            curve, and R² will plateau far below 1. Increasing the model degree
            to match the data will visibly improve the fit.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          3. Optimizer comparison
        </p>
        <Prose>
          <p>
            Reset and try the same settings with <strong>SGD</strong>,{' '}
            <strong>Momentum</strong>, and <strong>Adam</strong>. Notice that
            Adam typically converges fastest with less hand-tuning of the
            learning rate, while SGD may require a very small lr to avoid
            divergence at higher degrees.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          4. Learning rate sensitivity
        </p>
        <Prose>
          <p>
            With <strong>SGD</strong> and <strong>lr = 0.1</strong>, train a
            degree-5 model. If the loss explodes (nan or very large), reset and
            halve the lr. This is the classical "lr too large" failure mode.
            With Adam this is much less likely.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          5. High noise
        </p>
        <Prose>
          <p>
            Set <strong>noise = 1.0</strong> and a high model degree. Even a
            perfect model will have a high MSE because the data is inherently
            noisy. This is the irreducible (aleatoric) error — the floor you
            cannot beat regardless of model complexity.
          </p>
        </Prose>
      </InfoBox>
    </div>
  )
}
