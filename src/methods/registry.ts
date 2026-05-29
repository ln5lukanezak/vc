import type { FC } from 'react'

// ─── Method definition ────────────────────────────────────────────────────────

export interface MethodDef {
  id: string
  name: string
  group: string
  blurb: string
  Overview: FC
  Description: FC
  Formulas: FC
  Dataset: FC<DatasetProps>
  Visualization: FC<VisualizationProps>
  Insights: FC
}

// Shared prop shapes that the framework passes to Dataset and Visualization
export interface DatasetProps {
  onDataChange?: () => void
}

export interface VisualizationProps {
  // reserved for cross-method data-sharing if needed
  _placeholder?: never
}

// ─── Group order ──────────────────────────────────────────────────────────────

const GROUP_ORDER = [
  'Regression',
  'Classification',
  'Unsupervised',
  'Neural Networks',
  'Deep Learning',
]

// ─── Registry (populated by each method's index file) ────────────────────────

const _methods: MethodDef[] = []

export function registerMethod(def: MethodDef): void {
  _methods.push(def)
}

export function getMethods(): MethodDef[] {
  return _methods
}

export function getMethod(id: string): MethodDef | undefined {
  return _methods.find((m) => m.id === id)
}

// ─── Grouped structure for the sidebar ───────────────────────────────────────

export interface NavGroup {
  label: string
  items: Array<{ id: string; name: string }>
}

export function getNavGroups(): NavGroup[] {
  const groupMap = new Map<string, Array<{ id: string; name: string }>>()
  for (const m of _methods) {
    if (!groupMap.has(m.group)) groupMap.set(m.group, [])
    groupMap.get(m.group)!.push({ id: m.id, name: m.name })
  }
  // Sort groups by GROUP_ORDER
  const groups: NavGroup[] = []
  for (const label of GROUP_ORDER) {
    const items = groupMap.get(label)
    if (items) groups.push({ label, items })
  }
  // Any groups not in ORDER go at the end
  for (const [label, items] of groupMap) {
    if (!GROUP_ORDER.includes(label)) groups.push({ label, items })
  }
  return groups
}
