import { Prose, InfoBox, SectionHeading } from '../../components/Controls'

export function Overview() {
  return (
    <div className="max-w-2xl space-y-6">
      <Prose>
        <p>
          <strong className="text-slate-200">Logistic Regression</strong> is the
          workhorse binary classification algorithm. Despite its name it is a{' '}
          <em>classifier</em>, not a regressor: it models the probability that a
          point belongs to a class, and predicts the class with the highest
          probability.
        </p>
        <p>
          <strong className="text-slate-200">Softmax Regression</strong> (also called
          multinomial logistic regression) generalises binary logistic regression to{' '}
          <em>K ≥ 2</em> classes. Both are <em>linear</em> classifiers — the decision
          boundary is always a straight line (or hyperplane in higher dimensions).
        </p>
      </Prose>

      <SectionHeading>How It Works</SectionHeading>
      <Prose>
        <p>
          For each class <em>k</em> the model computes a linear score (logit):
        </p>
        <p className="metric text-cyan-300 text-center py-1">
          z_k = W_k · x + b_k
        </p>
        <p>
          These raw scores are passed through the <strong className="text-slate-200">softmax</strong>{' '}
          function to turn them into a proper probability distribution that sums to 1.
          Training minimises <strong className="text-slate-200">cross-entropy loss</strong>,
          which penalises low probability assigned to the true class.
        </p>
      </Prose>

      <SectionHeading>Why 2-Class Softmax = Logistic Regression</SectionHeading>
      <Prose>
        <p>
          With <em>K = 2</em>, softmax over two scores{' '}
          <span className="metric text-cyan-300">[z₀, z₁]</span> simplifies exactly to the
          sigmoid function applied to the difference{' '}
          <span className="metric text-cyan-300">z₁ − z₀</span>. So binary logistic
          regression and 2-class softmax are mathematically equivalent — just different
          parameterisations of the same model.
        </p>
      </Prose>

      <SectionHeading>Linear Decision Boundaries</SectionHeading>
      <Prose>
        <p>
          The decision boundary between classes <em>i</em> and <em>j</em> is the set of
          points where <span className="metric text-cyan-300">z_i = z_j</span>, which is
          a linear equation in <em>x</em>. This means logistic/softmax regression can only
          separate classes that are{' '}
          <strong className="text-slate-200">linearly separable</strong>. For non-linearly
          separable data (spirals, circles, XOR), the boundary will be as good as a
          straight line can do — which is often not good enough.
        </p>
      </Prose>

      <SectionHeading>When It Works / Fails</SectionHeading>
      <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Works well when</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Classes are roughly linearly separable</li>
            <li>You need probability estimates (not just class)</li>
            <li>Fast training, few parameters</li>
            <li>Interpretable weights</li>
            <li>Good baseline before complex models</li>
          </ul>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Fails when</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Classes are not linearly separable</li>
            <li>Complex curved boundaries needed</li>
            <li>Strong feature interactions exist</li>
            <li>→ try SVM (kernel) or MLP instead</li>
          </ul>
        </div>
      </div>

      <InfoBox>
        <strong className="text-indigo-300">Quick demo:</strong> Select 2 classes with
        high separation — the model quickly learns a clean boundary. Then lower the
        separation until classes overlap. Watch the accuracy drop and the probability
        surface become uncertain (grey zone). Finally, try 3 or 4 classes to see how
        softmax partitions the plane with multiple linear boundaries.
      </InfoBox>
    </div>
  )
}
