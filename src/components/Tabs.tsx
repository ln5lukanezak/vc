import React, { useState } from 'react'

// ─── Tab definitions ──────────────────────────────────────────────────────────

export interface TabItem {
  id: string
  label: string
  content: React.ReactNode
}

// The 6 standard section labels (§4 of spec)
export const SECTION_LABELS = [
  'Overview',
  'Description',
  'Formulas',
  'Dataset',
  'Visualization & Learning',
  'Insights',
] as const

export type SectionLabel = (typeof SECTION_LABELS)[number]

// ─── Tabs component ───────────────────────────────────────────────────────────

export interface TabsProps {
  tabs: TabItem[]
  defaultTab?: string
}

export function Tabs({ tabs, defaultTab }: TabsProps) {
  const [active, setActive] = useState(defaultTab ?? tabs[0]?.id ?? '')

  const activeTab = tabs.find((t) => t.id === active)

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div
        className="flex shrink-0 border-b border-slate-700/60 bg-slate-900 overflow-x-auto"
        role="tablist"
      >
        {tabs.map((tab) => {
          const isActive = tab.id === active
          return (
            <button
              key={tab.id}
              role="tab"
              aria-selected={isActive}
              onClick={() => setActive(tab.id)}
              className={[
                'px-4 py-2.5 text-sm font-medium whitespace-nowrap shrink-0',
                'border-b-2 transition-colors duration-150',
                isActive
                  ? 'border-indigo-500 text-indigo-300'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-500',
              ].join(' ')}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6" role="tabpanel">
        {activeTab?.content ?? null}
      </div>
    </div>
  )
}
