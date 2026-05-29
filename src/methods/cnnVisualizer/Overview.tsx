import { Prose, InfoBox, SectionHeading } from '../../components/Controls'

export function Overview() {
  return (
    <div className="max-w-2xl space-y-6">
      <Prose>
        <p>
          <strong className="text-slate-200">Convolutional Neural Networks (CNNs)</strong> are
          the backbone of modern computer vision. Their key insight: instead of connecting every
          pixel to every neuron (expensive, brittle), share a small{' '}
          <strong className="text-slate-200">kernel (filter)</strong> that slides across the
          image and detects the same local pattern wherever it appears.
        </p>
        <p>
          This <em>translational invariance</em> — detecting an edge at position (3,5) or (10,2)
          with the same weights — is what makes CNNs so powerful and parameter-efficient.
        </p>
      </Prose>

      <SectionHeading>The Convolution Operation</SectionHeading>
      <Prose>
        <p>
          A <strong className="text-slate-200">convolution</strong> slides a small weight matrix
          (the <em>kernel</em>) over the input image. At each position, it multiplies each kernel
          weight by the overlapping input pixel and sums the results — producing one number in
          the <strong className="text-slate-200">output feature map</strong>.
        </p>
        <p>
          The rectangular region of the input that influences a single output value is called the{' '}
          <strong className="text-slate-200">receptive field</strong>. For a 3×3 kernel, the
          receptive field is a 3×3 patch of the input.
        </p>
      </Prose>

      <SectionHeading>Kernels / Filters</SectionHeading>
      <Prose>
        <p>
          Different kernels detect different features: Sobel kernels detect edges, Gaussian kernels
          blur (remove high-frequency noise), Laplacian kernels detect outlines, sharpen kernels
          amplify fine detail. In a trained CNN,{' '}
          <strong className="text-slate-200">the kernels are learned</strong> — backpropagation
          finds the weights that minimize the loss, automatically discovering edges, textures,
          and higher-level patterns.
        </p>
      </Prose>

      <SectionHeading>Stride &amp; Padding</SectionHeading>
      <Prose>
        <p>
          <strong className="text-slate-200">Stride</strong> controls how many pixels the kernel
          jumps between applications. Stride 1 produces a dense output; stride 2 halves each
          spatial dimension. <strong className="text-slate-200">Padding</strong> adds border
          pixels (usually zeros) around the input:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li><strong className="text-slate-200">Valid</strong> — no padding; output is smaller than input.</li>
          <li><strong className="text-slate-200">Same</strong> — pad so output matches input size (at stride 1).</li>
        </ul>
      </Prose>

      <SectionHeading>The Conv → ReLU → Pool Pipeline</SectionHeading>
      <Prose>
        <p>
          One layer of a CNN typically chains three operations:
        </p>
        <ol className="list-decimal list-inside space-y-1">
          <li>
            <strong className="text-slate-200">Convolution</strong> — extract features (can produce
            negative values for edge-like kernels).
          </li>
          <li>
            <strong className="text-slate-200">ReLU</strong> — clip negatives to zero, introducing
            non-linearity; negative conv outputs "don't fire."
          </li>
          <li>
            <strong className="text-slate-200">Max-Pooling</strong> — take the maximum in each
            non-overlapping 2×2 block, shrinking the feature map and providing some
            translation robustness.
          </li>
        </ol>
        <p>
          Stacking these layers lets a CNN build up progressively higher-level representations:
          edges → corners → textures → object parts → full objects.
        </p>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Try it live:</strong> Go to{' '}
        <em>Visualization</em>, hit <strong>Play</strong>, and watch the kernel sweep across the
        input. Switch from <span className="metric text-cyan-300">Sobel-X</span> to{' '}
        <span className="metric text-cyan-300">Sobel-Y</span> and see how vertical vs. horizontal
        edges light up differently in the feature map.
      </InfoBox>

      <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Pros</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Parameter sharing (very efficient)</li>
            <li>Translation invariant</li>
            <li>Learns hierarchical features</li>
            <li>State-of-the-art for vision tasks</li>
          </ul>
        </div>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700">
          <div className="text-slate-300 font-medium mb-1">Cons</div>
          <ul className="space-y-0.5 list-disc list-inside">
            <li>Needs substantial data to train</li>
            <li>Computationally heavy (depth/width)</li>
            <li>Hard to interpret deeper layers</li>
            <li>Not ideal for non-grid data</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
