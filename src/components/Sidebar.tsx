import { getNavGroups } from '../methods/registry'

interface SidebarProps {
  activeId: string
  onSelect: (id: string) => void
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
// Generated from the method registry — no hardcoded nav lists.

export function Sidebar({ activeId, onSelect }: SidebarProps) {
  const groups = getNavGroups()

  return (
    <aside className="flex flex-col w-64 shrink-0 bg-slate-900 border-r border-slate-700/60 overflow-y-auto">
      {groups.map((group) => (
        <div key={group.label}>
          <p className="nav-group-label">{group.label}</p>
          <ul>
            {group.items.map((item) => (
              <li key={item.id}>
                <button
                  className={`nav-item w-full text-left${activeId === item.id ? ' active' : ''}`}
                  onClick={() => onSelect(item.id)}
                >
                  {item.name}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </aside>
  )
}
