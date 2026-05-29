import { Prose, InfoBox, SectionHeading } from '../../components/Controls'

export function Overview() {
  return (
    <div className="max-w-2xl space-y-6">
      <Prose>
        <p>
          <strong className="text-slate-200">Linear regression</strong> is the
          fundamental supervised learning algorithm for predicting a continuous
          output variable from one or more input features by fitting a linear
          (or polynomial) model to observed data.
        </p>
        <p>
          Given a set of (x, y) observations, we seek the parameters{' '}
          <em>w</em> and <em>b</em> such that the predicted values{' '}
          <span className="metric text-cyan-300">ŷ = wx + b</span> are as
          close as possible to the true labels <em>y</em> — where "close" is
          measured by <strong className="text-slate-200">Mean Squared Error (MSE)</strong>.
        </p>
      </Prose>

      <SectionHeading>When to use</SectionHeading>
      <Prose>
        <ul className="list-disc list-inside space-y-1">
          <li>The relationship between features and target is approximately linear.</li>
          <li>You need an interpretable model with fast training and prediction.</li>
          <li>As a baseline before trying more complex models.</li>
          <li>
            <strong className="text-slate-200">Polynomial regression</strong> extends
            this to any smooth curve — you simply add{' '}
            <span className="metric text-cyan-300">x², x³, …</span> as extra features
            and the algorithm stays linear in its parameters.
          </li>
        </ul>
      </Prose>

      <SectionHeading>Intuition</SectionHeading>
      <Prose>
        <p>
          Imagine stretching a rubber band across a scatter plot. Gradient
          descent "nudges" the line's slope and intercept toward whichever
          direction reduces the average squared distance to the data points.
          After enough steps, the band settles into the optimal position.
        </p>
        <p>
          For a polynomial of degree d, we work in a{' '}
          <em>lifted feature space</em> — the x-axis is replaced by{' '}
          <span className="metric text-cyan-300">[1, x, x², …, x^d]</span>.
          The model is still linear in those features, so the same gradient
          descent math applies exactly.
        </p>
      </Prose>

      <SectionHeading>Bias–Variance</SectionHeading>
      <Prose>
        <p>
          As you increase the model degree you buy{' '}
          <strong className="text-slate-200">lower bias</strong> (can fit more
          complex curves) at the cost of{' '}
          <strong className="text-slate-200">higher variance</strong> (sensitive
          to noise, overfits). The sweet spot is when the model degree matches
          the true underlying complexity of the data.
        </p>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Quick demo:</strong> Go to the{' '}
        <em>Dataset</em> tab, set <span className="metric">true degree = 3</span>,
        then in <em>Visualization</em> try <span className="metric">model degree = 1</span>{' '}
        (underfitting) and <span className="metric">model degree = 8</span>{' '}
        (overfitting) to see the trade-off.
      </InfoBox>

      <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Pros</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Fast, interpretable</li>
            <li>Closed-form solution exists</li>
            <li>Good starting baseline</li>
            <li>Works well when data is linear</li>
          </ul>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Cons</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Linear assumption may not hold</li>
            <li>High degree → overfitting</li>
            <li>Sensitive to outliers</li>
            <li>Requires feature scaling for GD</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
