import { useState } from 'react'
import { Slider, Button, SectionHeading, InfoBox } from '../../components/Controls'
import { getDataConfig, setDataConfig, regenerate, subscribe } from './dataStore'
import type { DatasetProps } from '../registry'
import { useEffect } from 'react'

export function Dataset({ onDataChange }: DatasetProps) {
  const [cfg, setCfg] = useState(getDataConfig)

  // Keep in sync if another component regenerates
  useEffect(() => {
    return subscribe(() => setCfg(getDataConfig()))
  }, [])

  function update<K extends keyof typeof cfg>(key: K, value: (typeof cfg)[K]) {
    const next = { ...cfg, [key]: value }
    setCfg(next)
    setDataConfig({ [key]: value })
  }

  function handleRegenerate() {
    // New random seed each time
    const newSeed = Math.floor(Math.random() * 100000)
    const next = { ...cfg, seed: newSeed }
    setCfg(next)
    setDataConfig(next)
    regenerate()
    onDataChange?.()
  }

  return (
    <div className="max-w-md space-y-6">
      <SectionHeading>Data Generator</SectionHeading>

      <div className="space-y-4">
        <Slider
          label="Samples (n)"
          value={cfg.n}
          min={10}
          max={300}
          step={10}
          onChange={(v) => update('n', v)}
        />
        <Slider
          label="Noise (σ)"
          value={cfg.noise}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => update('noise', v)}
          format={(v) => v.toFixed(2)}
        />
        <Slider
          label="True polynomial degree"
          value={cfg.degree}
          min={1}
          max={6}
          step={1}
          onChange={(v) => update('degree', v)}
        />
      </div>

      <Button
        label="Regenerate Data"
        onClick={handleRegenerate}
        variant="primary"
      />

      <InfoBox>
        <strong className="text-indigo-300">Generating process:</strong> x values
        are drawn uniformly from [−3, 3]. The true function is a random polynomial
        of the selected degree with small coefficients. Gaussian noise with the
        chosen σ is added to y. All randomness is seeded for reproducibility —
        click <em>Regenerate</em> to get a new dataset.
      </InfoBox>

      <div className="text-xs text-slate-500 space-y-1">
        <div>Current seed: <span className="metric text-slate-400">{cfg.seed}</span></div>
        <div>Samples: <span className="metric text-slate-400">{cfg.n}</span></div>
        <div>True degree: <span className="metric text-slate-400">{cfg.degree}</span></div>
        <div>Noise σ: <span className="metric text-slate-400">{cfg.noise.toFixed(2)}</span></div>
      </div>
    </div>
  )
}
