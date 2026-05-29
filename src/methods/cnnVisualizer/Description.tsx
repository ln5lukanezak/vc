import { Prose, SectionHeading, InfoBox } from '../../components/Controls'

export function Description() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>How Convolution Really Works</SectionHeading>
      <Prose>
        <p>
          At output position (i, j), the convolution slides a K×K kernel over the input
          starting at row <em>i·S − P</em> and column <em>j·S − P</em> (where S is the
          stride and P is the zero-padding amount). It multiplies each of the K² input
          pixels by the corresponding kernel weight and sums them:{' '}
          <span className="metric text-cyan-300">(I * K)(i,j) = ΣΣ I(i·S + m − P, j·S + n − P)·K(m,n)</span>.
        </p>
        <p>
          This is the <strong className="text-slate-200">dot product</strong> of the kernel
          vector and the input patch vector — exactly what a single linear neuron computes,
          but applied at every spatial location with shared weights.
        </p>
      </Prose>

      <SectionHeading>Output Size Formula</SectionHeading>
      <Prose>
        <p>
          Given input width W, kernel size K, padding P (each side), and stride S, the
          output width is:
        </p>
        <div className="bg-slate-800 rounded-md p-3 border border-slate-700 my-2">
          <span className="metric text-cyan-300">
            out_W = ⌊(W − K + 2P) / S⌋ + 1
          </span>
        </div>
        <p>
          For a 16×16 input, 3×3 kernel, stride 1, valid padding (P=0):
          out = ⌊(16 − 3) / 1⌋ + 1 = 14. With same padding (P=1): out = 16. With stride 2,
          valid: ⌊13/2⌋ + 1 = 7.
        </p>
      </Prose>

      <SectionHeading>ReLU Activation</SectionHeading>
      <Prose>
        <p>
          After the convolution, each output value is passed through{' '}
          <strong className="text-slate-200">ReLU</strong> (Rectified Linear Unit):
          ReLU(x) = max(0, x). For edge-detecting kernels like Sobel, the raw output contains
          large <em>negative</em> values (edges of one polarity) and large <em>positive</em> values
          (edges of the other polarity). ReLU zeroes out the negatives, keeping only one
          polarity of edge responses. This is visible in the Conv→ReLU panel.
        </p>
      </Prose>

      <SectionHeading>Max-Pooling</SectionHeading>
      <Prose>
        <p>
          <strong className="text-slate-200">Max-pooling</strong> with a 2×2 window, stride 2,
          replaces each non-overlapping 2×2 block with its maximum value. This:
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Halves each spatial dimension (14×14 → 7×7 for valid-padded 16×16 input).</li>
          <li>Provides mild translation invariance — an edge 1px shifted still produces the same max.</li>
          <li>Discards exact position information, keeping only "there was a strong response here."</li>
        </ul>
      </Prose>

      <SectionHeading>Key Terms</SectionHeading>
      <Prose>
        <ul className="list-disc list-inside space-y-1">
          <li><strong className="text-slate-200">Feature map:</strong> the output of one conv+activation layer for one filter.</li>
          <li><strong className="text-slate-200">Filter / kernel:</strong> the learned (or hand-designed) weight matrix.</li>
          <li><strong className="text-slate-200">Receptive field:</strong> input region that influences one output value.</li>
          <li><strong className="text-slate-200">Channel:</strong> one "slice" of an activation volume; grayscale has 1 channel, RGB has 3.</li>
          <li><strong className="text-slate-200">Depth:</strong> number of filters in a layer = number of output feature maps.</li>
        </ul>
      </Prose>

      <SectionHeading>Assumptions &amp; Failure Modes</SectionHeading>
      <Prose>
        <p>
          CNNs assume the data has <strong className="text-slate-200">local structure</strong>
          (nearby pixels are correlated) and{' '}
          <strong className="text-slate-200">translational stationarity</strong> (same patterns
          can occur anywhere). They fail or need large datasets when those assumptions break.
          Very small inputs can "run out of space" before the pooling layers — a common practical
          mistake when building deep networks for tiny images.
        </p>
      </Prose>

      <InfoBox>
        <strong className="text-indigo-300">Learning the kernels:</strong> In a trained CNN,
        gradient descent adjusts every kernel weight to minimize the loss. The first layer learns
        Gabor-like edge detectors naturally, even though no one told it to — the same patterns
        that the hand-crafted Sobel/Gaussian kernels here were engineered to detect.
      </InfoBox>
    </div>
  )
}
