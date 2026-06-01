import { useHashRoute } from './hooks/useHashRoute'
import { getMethod } from './methods/registry'
import { Sidebar } from './components/Sidebar'
import { MethodView } from './components/MethodView'

// ─── Welcome panel ─────────────────────────────────────────────────────────────
function WelcomePanel() {
  return (
    <div className="flex flex-col items-center justify-center h-full px-8 text-center gap-6">
      {/* Accent gradient orb */}
      <div
        className="w-24 h-24 rounded-full blur-2xl opacity-40 pointer-events-none"
        style={{ background: 'linear-gradient(135deg, #6366f1, #06b6d4)' }}
        aria-hidden
      />

      <h2 className="text-3xl font-semibold text-slate-100 -mt-16">
        Welcome to{' '}
        <span className="text-accent-gradient font-bold">ML Explorer</span>
      </h2>

      <p className="max-w-lg text-slate-400 leading-relaxed">
        Select a method from the sidebar to start exploring. Each method gives
        you a conceptual overview, the governing mathematics, a tunable
        synthetic dataset, and a{' '}
        <span className="text-slate-200 font-medium">
          live, animated training visualization
        </span>{' '}
        running entirely in your browser.
      </p>

      {/* Quick-start cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4 w-full max-w-2xl">
        {[
          {
            icon: '↗',
            title: 'Linear Regression',
            desc: 'Watch gradient descent fit a polynomial curve in real time.',
            id: 'linear-regression',
          },
          {
            icon: '◉',
            title: 'CNN Visualizer',
            desc: 'Animate a kernel sweep, then classify your drawn digit.',
            id: 'cnn',
          },
        ].map((card) => (
          <div
            key={card.id}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4 text-left
                       hover:border-indigo-500/60 transition-colors duration-200 cursor-default"
          >
            <span className="text-2xl">{card.icon}</span>
            <h3 className="mt-2 text-sm font-semibold text-slate-200">
              {card.title}
            </h3>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed">
              {card.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Coming Soon panel ─────────────────────────────────────────────────────────
function ComingSoonPanel({ id }: { id: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-400">
      <div className="text-4xl opacity-30">⚙</div>
      <p className="text-lg font-medium text-slate-300 capitalize">
        {id.replace(/-/g, ' ')}
      </p>
      <p className="text-sm text-slate-500">Coming in a later phase.</p>
    </div>
  )
}

// ─── App root ──────────────────────────────────────────────────────────────────
export default function App() {
  // Hash-based routing: #/linear-regression, #/svm, etc.
  const [activeId, navigate] = useHashRoute('linear-regression')

  const method = getMethod(activeId)

  return (
    <div className="flex flex-col h-screen bg-[var(--surface-bg)] text-[var(--text-primary)]">
      {/* ── Header ── */}
      <header
        className="flex items-center gap-4 px-6 py-3 shrink-0
                   bg-slate-900 border-b border-slate-700/60
                   shadow-[0_1px_0_0_rgba(99,102,241,0.15)]"
      >
        {/* Logo + title */}
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-md flex items-center justify-center text-white font-bold text-sm select-none"
            style={{ background: 'linear-gradient(135deg, #6366f1, #06b6d4)' }}
            aria-hidden
          >
            ML
          </div>
          <span className="text-lg font-bold tracking-tight text-slate-100">
            ML Explorer
          </span>
        </div>

        {/* Divider */}
        <span className="hidden sm:block text-slate-600 select-none">|</span>

        {/* Tagline */}
        <p className="hidden sm:block text-sm text-slate-400 leading-tight">
          Interactive machine learning — learn, visualize, and watch models
          train
        </p>

        {/* Spacer */}
        <div className="flex-1" />
      </header>

      {/* ── Body: sidebar + content ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar is generated from the method registry */}
        <Sidebar activeId={activeId} onSelect={navigate} />

        <main className="flex-1 overflow-hidden bg-[var(--surface-bg)]">
          {method ? (
            <MethodView method={method} />
          ) : activeId ? (
            <ComingSoonPanel id={activeId} />
          ) : (
            <WelcomePanel />
          )}
        </main>
      </div>
    </div>
  )
}
