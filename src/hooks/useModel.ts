import { useEffect, useState } from 'react'
import { parseConfig, type HFConfig } from '@/core/arch'
import { fetchModel, type ModelDetails } from '@/core/hf'

export const CUSTOM_PREFIX = 'custom:'

const cache = new Map<string, ModelDetails>()
const custom = new Map<string, ModelDetails>()

/** Registers a pasted config.json under a synthetic id ("custom:<name>"). */
export function registerCustomConfig(name: string, config: HFConfig): string {
  const id = `${CUSTOM_PREFIX}${name || 'my-model'}`
  custom.set(id, {
    id,
    arch: parseConfig(config),
    config,
    license: null,
    gated: false,
    pipelineTag: null,
    downloads: 0,
    lastModified: null,
    baseModel: null,
  })
  return id
}

type ModelState =
  | { status: 'loading'; id: string }
  | { status: 'error'; id: string; error: string; httpStatus?: number }
  | { status: 'ready'; id: string; model: ModelDetails }

export function useModel(id: string, reloadKey = 0): ModelState {
  const [state, setState] = useState<ModelState>(() => initial(id))

  useEffect(() => {
    const known = custom.get(id) ?? cache.get(id)
    if (known) {
      setState({ status: 'ready', id, model: known })
      return
    }
    if (id.startsWith(CUSTOM_PREFIX)) {
      setState({ status: 'error', id, error: 'Custom configs are kept in memory only — paste it again after a reload.' })
      return
    }
    const ctrl = new AbortController()
    setState({ status: 'loading', id })
    fetchModel(id, ctrl.signal)
      .then((m) => {
        cache.set(id, m)
        setState({ status: 'ready', id, model: m })
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return
        const err = e as Error & { status?: number }
        setState({ status: 'error', id, error: err.message, httpStatus: err.status })
      })
    return () => ctrl.abort()
  }, [id, reloadKey])

  return state.id === id ? state : initial(id)
}

function initial(id: string): ModelState {
  const known = custom.get(id) ?? cache.get(id)
  return known ? { status: 'ready', id, model: known } : { status: 'loading', id }
}

export function clearModelCache() {
  cache.clear()
}
