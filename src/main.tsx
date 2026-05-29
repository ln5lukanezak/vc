import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// KaTeX styles (required for react-katex to render correctly)
import 'katex/dist/katex.min.css'
// Method registrations — must be imported before App renders
import './methods/linearRegression/index'
import './methods/cnnVisualizer/index'
import './methods/mlpPlayground/index'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
