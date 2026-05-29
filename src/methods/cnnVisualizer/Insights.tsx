import { InfoBox, SectionHeading, Prose } from '../../components/Controls'

export function Insights() {
  return (
    <div className="max-w-2xl space-y-6">
      <SectionHeading>Try These Experiments</SectionHeading>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          1. Sobel-X vs Sobel-Y — directional edge detection
        </p>
        <Prose>
          <p>
            Select the <strong>Cross</strong> image, then compare{' '}
            <span className="metric text-cyan-300">Sobel-X</span> and{' '}
            <span className="metric text-cyan-300">Sobel-Y</span> kernels.
            Sobel-X detects <em>vertical edges</em> (left-right brightness change), so it
            strongly responds to the vertical sides of the cross bars and stays quiet at
            the horizontal top/bottom. Sobel-Y does the opposite. The output feature maps
            will look completely different — that's the kernel learning to be
            "directionally selective."
          </p>
          <p>
            Also try with <strong>Diagonal Edge</strong>: both Sobel kernels will produce
            weaker, more distributed responses because the edge is at 45°.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          2. Same padding — output stays the same size
        </p>
        <Prose>
          <p>
            Set padding to <span className="metric text-cyan-300">Same</span> and watch
            the "Out size" stat: it matches the input (16×16 → 16×16 with stride 1).
            With <span className="metric">Valid</span> padding and a 3×3 kernel you lose
            one pixel on each side (16×16 → 14×14). In deep networks this matters — many
            layers of Valid padding shrink the feature map to nothing.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          3. Increase stride — the sweep skips cells and the output shrinks
        </p>
        <Prose>
          <p>
            With stride 1 on a 16×16 input (valid), you get 14×14 = 196 steps.
            Switch to <span className="metric text-cyan-300">stride 2</span> and you get
            7×7 = 49 steps — the receptive-field box will visibly jump two pixels at a
            time. Pool size becomes 3×3. With stride 3: 5×5 output, 2×2 pool.
            Notice how the sweep finishes much faster and the feature map becomes
            coarser — you're trading spatial resolution for compute.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          4. Blur vs Sharpen vs Outline — frequency response
        </p>
        <Prose>
          <p>
            Try the <strong>Checkerboard</strong> image (highest frequency signal possible):
          </p>
          <ul className="list-disc list-inside space-y-1 mt-1">
            <li>
              <strong>Gaussian Blur</strong> — the feature map will be nearly flat (uniform
              gray) because blurring destroys high-frequency alternation.
            </li>
            <li>
              <strong>Sharpen</strong> — the feature map will be strongly amplified, nearly
              white everywhere.
            </li>
            <li>
              <strong>Outline (Laplacian)</strong> — very strong response everywhere on
              checkerboard (every cell is a local extremum).
            </li>
          </ul>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          5. ReLU kills negative responses
        </p>
        <Prose>
          <p>
            Run Sobel-X or Emboss and compare the "After Conv" and "After ReLU" panels.
            The conv output (diverging colormap) shows both blue (negative) and red
            (positive) responses — edges of two polarities. After ReLU, the blue regions
            all become black (zero). The network effectively "ignores" one polarity of
            edges per filter; two filters with opposite-sign kernels would cover both.
          </p>
        </Prose>
      </InfoBox>

      <InfoBox>
        <p className="font-semibold text-indigo-300 mb-1">
          6. MaxPool smooths and shrinks
        </p>
        <Prose>
          <p>
            Watch the 2×2 MaxPool panel fill in block by block as the sweep progresses
            (each 2×2 pool cell updates once all 4 contributing conv positions are
            computed). The pool output is always coarser than the ReLU output — strong
            responses spread slightly and weaker ones get dominated by the max. This
            is the mechanism that provides mild translation invariance.
          </p>
        </Prose>
      </InfoBox>
    </div>
  )
}
