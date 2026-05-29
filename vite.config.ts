import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// TODO (Phase 2): lazy-import @tensorflow/tfjs so it only loads on the CNN / NLP routes (keep it out of the main bundle).
export default defineConfig({
  plugins: [react()],
})
