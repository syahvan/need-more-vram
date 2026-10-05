export function fmtBytes(bytes: number, digits = 1): string {
  const gb = bytes / 1e9
  if (Math.abs(gb) >= 1000) return `${(gb / 1000).toFixed(digits)} TB`
  if (Math.abs(gb) >= 1) return `${gb.toFixed(digits)} GB`
  return `${(bytes / 1e6).toFixed(0)} MB`
}

export function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 1 : 2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`
  return `${Math.round(n / 1e3)}K`
}

export function fmtCompact(n: number): string {
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(n)
}

export function fmtInt(n: number): string {
  return new Intl.NumberFormat('en').format(Math.round(n))
}

export function fmtUsd(n: number, compact = false): string {
  if (compact && n >= 10_000) return `$${fmtCompact(n)}`
  return new Intl.NumberFormat('en', { style: 'currency', currency: 'USD', maximumFractionDigits: n < 100 ? 2 : 0 }).format(n)
}

export function fmtMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`
}

export function fmtDuration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`
  const m = seconds / 60
  if (m < 90) return `${Math.round(m)} min`
  const h = m / 60
  if (h < 48) return `${h.toFixed(1)} h`
  return `${(h / 24).toFixed(1)} days`
}
