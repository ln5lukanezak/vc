import { BlockMath, InlineMath } from 'react-katex'
import { SectionHeading } from '../../components/Controls'

interface FormulaRowProps {
  formula: string
  annotation: string
}

function FormulaRow({ formula, annotation }: FormulaRowProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center gap-2 py-3 border-b border-slate-700/50 last:border-0">
      <div className="flex-1 overflow-x-auto">
        <BlockMath math={formula} />
      </div>
      <div className="text-xs text-slate-500 sm:w-64 shrink-0 italic">
        {annotation}
      </div>
    </div>
  )
}

export function Formulas() {
  return (
    <div className="max-w-2xl space-y-8">
      <SectionHeading>2-D Discrete Convolution</SectionHeading>
      <div className="text-sm text-slate-400 mb-2">
        For input image <InlineMath math="I" />, kernel <InlineMath math="K" /> of size{' '}
        <InlineMath math="k \times k" />, stride <InlineMath math="S" />, padding <InlineMath math="P" />:
      </div>
      <FormulaRow
        formula="(I * K)(i,j) = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I(i \cdot S + m - P,\; j \cdot S + n - P)\; K(m,\, n)"
        annotation="Each output value is the dot product of the kernel with the overlapping input patch. Positions outside the input boundary are zero (zero-padding)."
      />

      <SectionHeading>Output Size</SectionHeading>
      <FormulaRow
        formula="\text{out} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1"
        annotation="Applies to both height and width. W = input size, K = kernel size, P = padding per side, S = stride. 'Same' padding sets P = ⌊K/2⌋ to preserve spatial size at S=1."
      />

      <SectionHeading>ReLU Activation</SectionHeading>
      <FormulaRow
        formula="\text{ReLU}(x) = \max(0,\, x)"
        annotation="Clips negative values to zero. Introduces non-linearity; without it, stacked conv layers collapse to one linear op. Negative edge responses (one polarity) are zeroed out."
      />

      <SectionHeading>2×2 Max-Pooling</SectionHeading>
      <FormulaRow
        formula="\text{Pool}(i,j) = \max_{\,r \in \{0,1\},\, c \in \{0,1\}} \text{feat}(2i + r,\; 2j + c)"
        annotation="Takes the maximum in each non-overlapping 2×2 block. Halves each spatial dimension. Provides mild translation invariance."
      />

      <SectionHeading>Learned Kernels (note)</SectionHeading>
      <div className="text-sm text-slate-300 leading-relaxed space-y-2">
        <p>
          In a trained CNN the kernel weights <InlineMath math="K(m,n)" /> are{' '}
          <strong className="text-slate-200">parameters learned by backpropagation</strong>:{' '}
        </p>
        <FormulaRow
          formula="K \leftarrow K - \eta \,\frac{\partial \mathcal{L}}{\partial K}, \quad \frac{\partial \mathcal{L}}{\partial K(m,n)} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial (I*K)(i,j)} \cdot I(i{\cdot}S{+}m{-}P,\; j{\cdot}S{+}n{-}P)"
          annotation="Gradient of loss w.r.t. a kernel weight = sum over all output positions of (upstream gradient × input patch value at that position). The kernel sees every spatial location — that's weight sharing."
        />
      </div>
    </div>
  )
}
