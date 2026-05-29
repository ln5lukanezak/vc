/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Indigo-to-cyan accent palette
        accent: {
          from: '#6366f1', // indigo-500
          to:   '#06b6d4', // cyan-500
          DEFAULT: '#818cf8', // indigo-400 – mid-point for single-color uses
        },
        surface: {
          DEFAULT: '#0f172a', // slate-900 – app background
          1: '#1e293b',       // slate-800 – card / sidebar
          2: '#334155',       // slate-700 – hover states, borders
        },
        muted: '#94a3b8',     // slate-400 – secondary text
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Avenir', 'Helvetica', 'Arial', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', '"Cascadia Code"', 'Consolas', 'monospace'],
      },
      backgroundImage: {
        'accent-gradient': 'linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)',
      },
    },
  },
  plugins: [],
}
