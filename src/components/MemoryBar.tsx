import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { fmtBytes } from '@/lib/format'

export interface MemorySegment {
  key: string
  label: string
  bytes: number
}

// Categorical slots in fixed order (validated palette); text never wears these.
const SLOTS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)', 'var(--series-5)', 'var(--series-6)', 'var(--series-7)']

interface Props {
  segments: MemorySegment[]
  capacity: number
  /** Optional reserved headroom marker (e.g. gpu-memory-utilization). */
  usable?: number
  title?: string
}

/** Stacked horizontal bar of memory use against a GPU's capacity. */
export function MemoryBar({ segments, capacity, usable, title }: Props) {
  const used = segments.reduce((s, x) => s + Math.max(0, x.bytes), 0)
  const scale = Math.max(capacity, used)
  const over = used > capacity
  const visible = segments.filter((s) => s.bytes > 0)

  return (
    <figure className="flex flex-col gap-3">
      <figcaption className="flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium">{title}</span>
        <span className="font-mono tabular-nums text-muted-foreground">
          <span className={over ? 'font-semibold text-destructive' : 'text-foreground'}>{fmtBytes(used)}</span> / {fmtBytes(capacity, 0)}
        </span>
      </figcaption>
      <div className="relative h-6 w-full overflow-hidden rounded-md bg-muted" role="img" aria-label={`${fmtBytes(used)} of ${fmtBytes(capacity)}`}>
        <div className="flex h-full gap-[2px]">
          {visible.map((s) => {
            const i = segments.indexOf(s)
            return (
              <Tooltip key={s.key}>
                <TooltipTrigger asChild>
                  <div
                    className="h-full first:rounded-l-md last:rounded-r-md transition-[width] duration-300"
                    style={{ width: `${(s.bytes / scale) * 100}%`, background: SLOTS[i % SLOTS.length], minWidth: 3 }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <span className="font-medium">{s.label}</span> · {fmtBytes(s.bytes, 2)} ({((s.bytes / capacity) * 100).toFixed(1)}%)
                </TooltipContent>
              </Tooltip>
            )
          })}
        </div>
        {usable !== undefined && usable < capacity && (
          <div className="absolute inset-y-0 w-px bg-foreground/50" style={{ left: `${(usable / scale) * 100}%` }} title="Engine memory limit" />
        )}
        {over && <div className="absolute inset-y-0 w-0.5 bg-destructive" style={{ left: `${(capacity / scale) * 100}%` }} />}
      </div>
      <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
        {segments.map((s, i) => (
          <li key={s.key} className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <span className="size-2.5 shrink-0 rounded-sm" style={{ background: SLOTS[i % SLOTS.length] }} />
              {s.label}
            </span>
            <span className="font-mono tabular-nums">{fmtBytes(Math.max(0, s.bytes))}</span>
          </li>
        ))}
      </ul>
    </figure>
  )
}
