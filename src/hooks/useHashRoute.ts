import { useState, useEffect } from 'react'

/**
 * Lightweight hash-based routing.
 * Hash format: `#/<method-id>` e.g. `#/linear-regression`
 *
 * Returns [currentId, navigate]
 * - currentId: the method id from the hash, or `defaultId` if none
 * - navigate: set the hash (and therefore the active method)
 */
export function useHashRoute(defaultId: string): [string, (id: string) => void] {
  const parseHash = (): string => {
    const hash = window.location.hash // e.g. "#/linear-regression" or ""
    if (hash.startsWith('#/')) return hash.slice(2)
    return ''
  }

  const [currentId, setCurrentId] = useState<string>(() => {
    const parsed = parseHash()
    return parsed || defaultId
  })

  useEffect(() => {
    const handler = () => {
      const parsed = parseHash()
      setCurrentId(parsed || defaultId)
    }
    window.addEventListener('hashchange', handler)
    // Sync in case the hash was set before mount
    handler()
    return () => window.removeEventListener('hashchange', handler)
  }, [defaultId])

  const navigate = (id: string) => {
    window.location.hash = `#/${id}`
    setCurrentId(id)
  }

  return [currentId, navigate]
}
