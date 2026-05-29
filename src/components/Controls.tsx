import React from 'react'

// ─── Slider ──────────────────────────────────────────────────────────────────

export interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
  disabled?: boolean
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  disabled = false,
}: SliderProps) {
  const display = format ? format(value) : String(value)
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label className="text-xs text-slate-400">{label}</label>
        <span className="metric text-xs text-indigo-300 min-w-[3rem] text-right">
          {display}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="slider-track w-full h-1.5 appearance-none rounded-full bg-slate-700
                   accent-indigo-500 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
      />
    </div>
  )
}

// ─── Select ──────────────────────────────────────────────────────────────────

export interface SelectProps {
  label: string
  value: string
  options: Array<{ value: string; label: string }>
  onChange: (v: string) => void
  disabled?: boolean
}

export function Select({ label, value, options, onChange, disabled = false }: SelectProps) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-slate-400">{label}</label>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-2 py-1.5 text-sm rounded-md
                   bg-slate-800 border border-slate-600
                   text-slate-200 focus:outline-none focus:border-indigo-500
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

// ─── ButtonRow ───────────────────────────────────────────────────────────────

export interface ButtonDef {
  label: string
  onClick: () => void
  variant?: 'primary' | 'secondary' | 'danger'
  disabled?: boolean
}

export function ButtonRow({ buttons }: { buttons: ButtonDef[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {buttons.map((btn) => (
        <Button key={btn.label} {...btn} />
      ))}
    </div>
  )
}

export function Button({ label, onClick, variant = 'secondary', disabled = false }: ButtonDef) {
  const base =
    'px-3 py-1.5 text-sm rounded-md font-medium transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed'
  const variants: Record<string, string> = {
    primary:
      'bg-indigo-600 hover:bg-indigo-500 text-white border border-indigo-500',
    secondary:
      'bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600',
    danger:
      'bg-red-900/50 hover:bg-red-800/60 text-red-300 border border-red-700/60',
  }
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${variants[variant]}`}
    >
      {label}
    </button>
  )
}

// ─── Stat readout ─────────────────────────────────────────────────────────────

export interface StatProps {
  label: string
  value: string | number
  unit?: string
  color?: string
}

export function Stat({ label, value, unit, color }: StatProps) {
  const colorClass = color ?? 'text-cyan-300'
  return (
    <div className="flex flex-col items-start bg-slate-800 border border-slate-700 rounded-md px-3 py-2 min-w-[80px]">
      <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className={`metric text-base font-semibold ${colorClass}`}>
        {typeof value === 'number' ? fmtStat(value) : value}
        {unit && <span className="text-xs text-slate-500 ml-0.5">{unit}</span>}
      </span>
    </div>
  )
}

function fmtStat(v: number): string {
  if (!isFinite(v)) return '—'
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2)
  if (Math.abs(v) >= 10000) return v.toExponential(2)
  return v.toPrecision(4).replace(/\.?0+$/, '')
}

// ─── Section heading ─────────────────────────────────────────────────────────

export function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3 mt-5 first:mt-0">
      {children}
    </h3>
  )
}

// ─── Prose block ──────────────────────────────────────────────────────────────

export function Prose({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-sm text-slate-300 leading-relaxed space-y-3">
      {children}
    </div>
  )
}

// ─── Info box ─────────────────────────────────────────────────────────────────

export function InfoBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-indigo-500/30 bg-indigo-500/5 px-4 py-3 text-sm text-slate-300 leading-relaxed">
      {children}
    </div>
  )
}
