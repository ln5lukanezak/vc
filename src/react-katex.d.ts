declare module 'react-katex' {
  import type { FC } from 'react'

  interface KatexProps {
    math: string
    block?: boolean
    errorColor?: string
    renderError?: (error: Error | TypeError) => JSX.Element
    settings?: object
  }

  export const InlineMath: FC<KatexProps>
  export const BlockMath: FC<KatexProps>
}
