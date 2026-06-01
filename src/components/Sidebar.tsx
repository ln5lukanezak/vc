import { getNavGroups } from '../methods/registry'

interface SidebarProps {
  activeId: string
  onSelect: (id: string) => void
  /** Drawer open state — only meaningful below the `lg` breakpoint. */
  open: boolean
  /** Dismiss the mobile drawer (backdrop tap). */
  onClose: () => void
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
// Generated from the method registry — no hardcoded nav lists.
// Desktop (lg+): static column. Mobile: slide-in drawer over the content.

export function Sidebar({ activeId, onSelect, open, onClose }: SidebarProps) {
  const groups = getNavGroups()

  return (
    <>
      {/* Backdrop — only on mobile while the drawer is open */}
      {open && (
        <div
          className="absolute inset-0 z-30 bg-black/50 lg:hidden"
          onClick={onClose}
          aria-hidden
        />
      )}

      <aside
        className={[
          'flex flex-col w-64 shrink-0 bg-slate-900 border-r border-slate-700/60 overflow-y-auto',
          // Mobile: absolute drawer anchored to the body, slides in/out.
          'absolute inset-y-0 left-0 z-40 transition-transform duration-200',
          open ? 'translate-x-0' : '-translate-x-full',
          // Desktop: part of the normal flex flow, always visible.
          'lg:static lg:z-auto lg:translate-x-0',
        ].join(' ')}
      >
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
    </>
  )
}
