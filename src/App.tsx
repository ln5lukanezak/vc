import { useState } from 'react'
import { useHashRoute } from './hooks/useHashRoute'
import { getMethod } from './methods/registry'
import { Sidebar } from './components/Sidebar'
import { MethodView } from './components/MethodView'

// ─── Welcome panel ─────────────────────────────────────────────────────────────
function WelcomePanel({ onSelect }: { onSelect: (id: string) => void }) {
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
            desc: 'Watch a kernel sweep over an image, building the feature map cell by cell.',
            id: 'cnn-visualizer',
          },
        ].map((card) => (
          <button
            key={card.id}
            onClick={() => onSelect(card.id)}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4 text-left
                       hover:border-indigo-500/60 transition-colors duration-200 cursor-pointer"
          >
            <span className="text-2xl">{card.icon}</span>
            <h3 className="mt-2 text-sm font-semibold text-slate-200">
              {card.title}
            </h3>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed">
              {card.desc}
            </p>
          </button>
        ))}
      </div>
    </div>
  )
}

// ─── Not-found panel ───────────────────────────────────────────────────────────
function NotFoundPanel({ id }: { id: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 px-8 text-center text-slate-400">
      <div className="text-4xl opacity-30">∅</div>
      <p className="text-lg font-medium text-slate-300">Method not found</p>
      <p className="text-sm text-slate-500 max-w-sm">
        “{id.replace(/-/g, ' ')}” isn’t one of the available methods. Pick one
        from the sidebar to get started.
      </p>
    </div>
  )
}

// ─── App root ──────────────────────────────────────────────────────────────────
export default function App() {
  // Hash-based routing: #/linear-regression, #/svm, etc.
  // Empty default → land on the Welcome screen.
  const [activeId, navigate] = useHashRoute('')
  // Mobile nav drawer (sidebar collapses below the `lg` breakpoint).
  const [drawerOpen, setDrawerOpen] = useState(false)

  const method = getMethod(activeId)

  // Navigate + close the mobile drawer (no-op on desktop where it's always open).
  const handleSelect = (id: string) => {
    navigate(id)
    setDrawerOpen(false)
  }

  return (
    <div className="flex flex-col h-screen bg-[var(--surface-bg)] text-[var(--text-primary)]">
      {/* ── Header ── */}
      <header
        className="flex items-center gap-3 sm:gap-4 px-4 sm:px-6 py-3 shrink-0
                   bg-slate-900 border-b border-slate-700/60
                   shadow-[0_1px_0_0_rgba(99,102,241,0.15)]"
      >
        {/* Hamburger — toggles the nav drawer on small screens */}
        <button
          onClick={() => setDrawerOpen((o) => !o)}
          className="lg:hidden -ml-1 p-1.5 rounded-md text-slate-300 hover:bg-slate-800
                     focus:outline-none focus:ring-1 focus:ring-indigo-500"
          aria-label={drawerOpen ? 'Close navigation' : 'Open navigation'}
          aria-expanded={drawerOpen}
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
            <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
        </button>

        {/* Logo + title — click to return to the Welcome screen */}
        <button
          onClick={() => handleSelect('')}
          className="flex items-center gap-3 cursor-pointer"
          aria-label="ML Explorer — home"
        >
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
        </button>

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
      {/* `relative` anchors the mobile drawer + backdrop to the body (below the header). */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* Sidebar is generated from the method registry; it's a drawer below `lg`. */}
        <Sidebar
          activeId={activeId}
          onSelect={handleSelect}
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
        />

        <main className="flex-1 overflow-hidden bg-[var(--surface-bg)]">
          {method ? (
            <MethodView method={method} />
          ) : activeId ? (
            <NotFoundPanel id={activeId} />
          ) : (
            <WelcomePanel onSelect={handleSelect} />
          )}
        </main>
      </div>

      {/* ── Footer ── */}
      <footer
        className="shrink-0 px-4 sm:px-6 py-2 text-center text-xs text-slate-500
                   bg-slate-900 border-t border-slate-700/60"
      >
        Built by{' '}
        <span className="text-slate-400 font-medium">Luka Nežak</span> with
        Claude
      </footer>
    </div>
  )
}
