import { CircleHelp, type LucideIcon } from 'lucide-react'
import { useEffect, useId, useState, type ReactNode } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

export function Hint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="text-muted-foreground hover:text-foreground" aria-label="More info">
          <CircleHelp className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-72 text-xs leading-relaxed">{children}</TooltipContent>
    </Tooltip>
  )
}

interface FieldShellProps {
  label: string
  hint?: ReactNode
  htmlFor?: string
  className?: string
  children: ReactNode
}

function FieldShell({ label, hint, htmlFor, className, children }: FieldShellProps) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <div className="flex items-center gap-1.5">
        <Label htmlFor={htmlFor} className="text-xs font-medium text-muted-foreground">
          {label}
        </Label>
        {hint && <Hint>{hint}</Hint>}
      </div>
      {children}
    </div>
  )
}

interface NumberFieldProps {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  suffix?: string
  hint?: ReactNode
  /** Small helper line under the input (e.g. "≈ 1.2k words"). */
  help?: ReactNode
  className?: string
}

/** Numeric input that lets you type freely and commits valid numbers. */
export function NumberField({ label, value, onChange, min, max, step, suffix, hint, help, className }: NumberFieldProps) {
  const id = useId()
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])

  const commit = (raw: string) => {
    const v = Number(raw.replace(/[_,\s]/g, ''))
    if (!Number.isFinite(v)) return setText(String(value))
    const clamped = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v))
    onChange(clamped)
    setText(String(clamped))
  }

  return (
    <FieldShell label={label} hint={hint} htmlFor={id} className={className}>
      <div className="relative">
        <Input
          id={id}
          inputMode="decimal"
          value={text}
          step={step}
          onChange={(e) => {
            setText(e.target.value)
            const v = Number(e.target.value.replace(/[_,\s]/g, ''))
            if (e.target.value !== '' && Number.isFinite(v) && (min === undefined || v >= min) && (max === undefined || v <= max)) onChange(v)
          }}
          onBlur={(e) => commit(e.target.value)}
          className={cn('h-9 font-mono tabular-nums', suffix && 'pr-14')}
        />
        {suffix && (
          <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-muted-foreground">
            {suffix}
          </span>
        )}
      </div>
      {help && <span className="text-[11px] text-muted-foreground">{help}</span>}
    </FieldShell>
  )
}

interface SelectFieldProps<T extends string> {
  label: string
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string; detail?: string }[]
  hint?: ReactNode
  className?: string
}

export function SelectField<T extends string>({ label, value, onChange, options, hint, className }: SelectFieldProps<T>) {
  const id = useId()
  return (
    <FieldShell label={label} hint={hint} htmlFor={id} className={className}>
      <Select value={value} onValueChange={(v) => onChange(v as T)}>
        <SelectTrigger id={id} className="h-9 w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              <span>{o.label}</span>
              {o.detail && <span className="ml-2 text-xs text-muted-foreground">{o.detail}</span>}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </FieldShell>
  )
}

interface SwitchFieldProps {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  hint?: ReactNode
  className?: string
}

export function SwitchField({ label, checked, onChange, hint, className }: SwitchFieldProps) {
  const id = useId()
  return (
    <div className={cn('flex items-center justify-between gap-3 rounded-lg border px-3 py-2', className)}>
      <div className="flex items-center gap-1.5">
        <Label htmlFor={id} className="text-sm">
          {label}
        </Label>
        {hint && <Hint>{hint}</Hint>}
      </div>
      <Switch id={id} checked={checked} onCheckedChange={onChange} />
    </div>
  )
}

export function SectionTitle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{children}</h3>
      {action}
    </div>
  )
}

export interface ChoiceOption {
  id: string
  label: string
  detail?: string
  icon?: LucideIcon
}

/** Large, friendly single-choice cards (presets, quality levels). */
export function ChoiceCards({
  options,
  value,
  onChange,
  columns = 1,
}: {
  options: ChoiceOption[]
  value: string | null
  onChange: (id: string) => void
  columns?: 1 | 2 | 3
}) {
  return (
    <div className={cn('grid gap-2', columns === 2 && 'grid-cols-2', columns === 3 && 'grid-cols-3')} role="radiogroup">
      {options.map((o) => {
        const active = o.id === value
        return (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.id)}
            className={cn(
              'flex items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors hover:bg-muted/60',
              active && 'border-foreground/60 bg-muted ring-1 ring-foreground/20',
            )}
          >
            {o.icon && <o.icon className={cn('mt-0.5 size-4 shrink-0', active ? 'text-foreground' : 'text-muted-foreground')} aria-hidden />}
            <span className="flex min-w-0 flex-col">
              <span className="text-sm font-medium leading-5">{o.label}</span>
              {o.detail && <span className="text-xs leading-snug text-muted-foreground">{o.detail}</span>}
            </span>
          </button>
        )
      })}
    </div>
  )
}
