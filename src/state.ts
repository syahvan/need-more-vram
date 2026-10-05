import { useCallback, useEffect, useState } from 'react'
import type { PriceKind } from '@/core/hardware'
import { DEFAULT_ENGINE, type EngineAssumptions } from '@/core/inference'
import type { KvFormat, WeightFormat } from '@/core/precision'
import { DEFAULT_GRPO, DEFAULT_TRAIN, type GrpoConfig, type TrainConfig } from '@/core/training'

export interface InferenceInputs {
  /** 'auto' follows the checkpoint's native format. */
  weightFormat: WeightFormat | 'auto'
  kvFormat: KvFormat
  inputTokens: number
  outputTokens: number
  sharedPrefixTokens: number
  prefixCaching: boolean
  loadMode: 'concurrent' | 'rate'
  concurrentStreams: number
  peakRps: number
  avgRps: number
  callsPerRequest: number
  peakHoursPerDay: number
  ttftMs: number
  minTokPerSec: number
  minReplicas: number
}

export interface MarketInputs {
  providers: ('aws' | 'gcp')[]
  /** 'any' or a provider region id. */
  region: string
  priceKind: PriceKind
  /** Also rank GPUs you own / rent elsewhere at a flat $/GPU-hour. */
  includeOwned: boolean
  ownedHourlyPerGpu: number
  ownedGpuIds: string[]
}

export interface TrainingInputs extends Omit<TrainConfig, 'rl' | 'datasetTokens'> {
  /** Dataset size is entered as examples × average length. */
  numExamples: number
  avgTokensPerExample: number
  useRl: boolean
  rl: GrpoConfig
  /** GPU used for the memory breakdown / "will it fit" view. */
  localGpuId: string
  localGpuCount: number
}

export interface UiPrefs {
  /** Show every knob instead of the guided view. */
  advanced: boolean
}

export interface AppState {
  modelId: string
  ui: UiPrefs
  tab: 'inference' | 'training'
  inf: InferenceInputs
  train: TrainingInputs
  market: MarketInputs
  engine: EngineAssumptions
}

export const DEFAULT_STATE: AppState = {
  modelId: 'Qwen/Qwen3.5-9B',
  ui: { advanced: false },
  tab: 'inference',
  inf: {
    weightFormat: 'auto',
    kvFormat: 'bf16',
    inputTokens: 2048,
    outputTokens: 512,
    sharedPrefixTokens: 0,
    prefixCaching: true,
    loadMode: 'concurrent',
    concurrentStreams: 32,
    peakRps: 10,
    avgRps: 3,
    callsPerRequest: 1,
    peakHoursPerDay: 4,
    ttftMs: 2000,
    minTokPerSec: 20,
    minReplicas: 1,
  },
  train: {
    ...(({ datasetTokens: _d, ...rest }) => rest)(DEFAULT_TRAIN),
    numExamples: 20_000,
    avgTokensPerExample: 2_500,
    useRl: false,
    rl: DEFAULT_GRPO,
    localGpuId: 'rtx-4090-24gb',
    localGpuCount: 1,
  },
  market: {
    providers: ['aws', 'gcp'],
    region: 'any',
    priceKind: 'onDemand',
    includeOwned: false,
    ownedHourlyPerGpu: 1,
    ownedGpuIds: ['rtx-4090-24gb', 'rtx-5090-32gb', 'rtx-pro-6000-blackwell-96gb'],
  },
  engine: DEFAULT_ENGINE,
}

// ---- URL hash persistence: only values that differ from defaults are encoded.

type Json = Record<string, unknown>

function diff(cur: Json, base: Json): Json | undefined {
  const out: Json = {}
  for (const k of Object.keys(cur)) {
    const a = cur[k]
    const b = base[k]
    if (a && typeof a === 'object' && !Array.isArray(a) && b && typeof b === 'object') {
      const d = diff(a as Json, b as Json)
      if (d) out[k] = d
    } else if (JSON.stringify(a) !== JSON.stringify(b)) out[k] = a
  }
  return Object.keys(out).length ? out : undefined
}

function merge<T>(base: T, patch: unknown): T {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) return (patch ?? base) as T
  const out: Json = { ...(base as Json) }
  for (const [k, v] of Object.entries(patch as Json)) {
    if (!(k in out)) continue
    const b = out[k]
    out[k] = b && typeof b === 'object' && !Array.isArray(b) ? merge(b, v) : v
  }
  return out as T
}

function encode(s: AppState): string {
  const d = diff(s as unknown as Json, DEFAULT_STATE as unknown as Json)
  if (!d) return ''
  return btoa(unescape(encodeURIComponent(JSON.stringify(d)))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function decode(hash: string): AppState {
  const raw = hash.replace(/^#/, '')
  if (!raw) return DEFAULT_STATE
  try {
    const json = decodeURIComponent(escape(atob(raw.replace(/-/g, '+').replace(/_/g, '/'))))
    return merge(DEFAULT_STATE, JSON.parse(json))
  } catch {
    return DEFAULT_STATE
  }
}

export function useAppState() {
  const [state, setState] = useState<AppState>(() => decode(window.location.hash))

  useEffect(() => {
    const h = encode(state)
    const url = `${window.location.pathname}${window.location.search}${h ? `#${h}` : ''}`
    window.history.replaceState(null, '', url)
  }, [state])

  useEffect(() => {
    const onHash = () => setState(decode(window.location.hash))
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const update = useCallback(<K extends keyof AppState>(key: K, patch: Partial<AppState[K]> | AppState[K]) => {
    setState((s) => {
      const cur = s[key]
      const next = typeof cur === 'object' && !Array.isArray(cur) ? { ...cur, ...(patch as object) } : patch
      return { ...s, [key]: next }
    })
  }, [])

  const reset = useCallback(() => setState(DEFAULT_STATE), [])
  return { state, update, reset }
}
