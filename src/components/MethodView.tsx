import { useState } from 'react'
import { Tabs } from './Tabs'
import type { MethodDef } from '../methods/registry'

interface MethodViewProps {
  method: MethodDef
}

// ─── MethodView ───────────────────────────────────────────────────────────────
// Renders a MethodDef through the 6-tab layout.
// Dataset and Visualization tabs share state through a simple key-based remount
// when data is regenerated.

export function MethodView({ method }: MethodViewProps) {
  const [dataKey, setDataKey] = useState(0)

  const tabs = [
    {
      id: 'overview',
      label: 'Overview',
      content: <method.Overview />,
    },
    {
      id: 'description',
      label: 'Description',
      content: <method.Description />,
    },
    {
      id: 'formulas',
      label: 'Formulas',
      content: <method.Formulas />,
    },
    {
      id: 'dataset',
      label: 'Dataset',
      content: <method.Dataset onDataChange={() => setDataKey((k) => k + 1)} />,
    },
    {
      id: 'visualization',
      label: 'Visualization & Learning',
      content: <method.Visualization key={dataKey} />,
    },
    {
      id: 'insights',
      label: 'Insights',
      content: <method.Insights />,
    },
  ]

  return (
    <div className="flex flex-col h-full">
      {/* Method title bar */}
      <div className="shrink-0 px-6 py-3 border-b border-slate-700/40 bg-slate-900/50">
        <h2 className="text-base font-semibold text-slate-100">{method.name}</h2>
        {method.blurb && (
          <p className="text-xs text-slate-500 mt-0.5">{method.blurb}</p>
        )}
      </div>
      <div className="flex-1 overflow-hidden">
        <Tabs tabs={tabs} defaultTab="overview" />
      </div>
    </div>
  )
}
