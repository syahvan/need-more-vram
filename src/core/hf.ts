// Browser-side Hugging Face Hub client. All requests go straight to
// huggingface.co (CORS-enabled); an optional token unlocks gated/private repos.

import { parseConfig, type HFConfig, type ModelArch } from './arch'

const HUB = 'https://huggingface.co'
const TOKEN_KEY = 'nmv:hf-token'

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    // Storage unavailable (private mode); the token just won't persist.
  }
}

function headers(): HeadersInit {
  const t = getToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

export class HubError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { headers: headers(), signal })
  if (!res.ok) {
    const reason =
      res.status === 401 || res.status === 403
        ? 'This repo is gated or private. Accept its license on huggingface.co and add a read token in Settings.'
        : res.status === 404
          ? 'Not found on the Hub (check the repo id, or it has no config.json).'
          : `Hub request failed (${res.status}).`
    throw new HubError(reason, res.status)
  }
  return res.json() as Promise<T>
}

export interface ModelSummary {
  id: string
  downloads: number
  likes: number
  pipelineTag: string | null
  params: number | null
  gated: boolean
}

interface ApiModel {
  id: string
  downloads?: number
  likes?: number
  pipeline_tag?: string
  safetensors?: { total?: number }
  gated?: boolean | string
  cardData?: { license?: string; license_name?: string; language?: string | string[]; base_model?: string | string[] }
  lastModified?: string
  tags?: string[]
}

const EXPAND = ['downloads', 'likes', 'pipeline_tag', 'safetensors', 'gated'].map((e) => `expand[]=${e}`).join('&')

export async function searchModels(query: string, signal?: AbortSignal): Promise<ModelSummary[]> {
  const q = encodeURIComponent(query.trim())
  const url = `${HUB}/api/models?search=${q}&sort=downloads&direction=-1&limit=20&${EXPAND}`
  const rows = await getJson<ApiModel[]>(url, signal)
  return rows.map((m) => ({
    id: m.id,
    downloads: m.downloads ?? 0,
    likes: m.likes ?? 0,
    pipelineTag: m.pipeline_tag ?? null,
    params: m.safetensors?.total ?? null,
    gated: !!m.gated,
  }))
}

export interface ModelDetails {
  id: string
  arch: ModelArch
  config: HFConfig
  license: string | null
  gated: boolean
  pipelineTag: string | null
  downloads: number
  lastModified: string | null
  baseModel: string | null
}

export async function fetchModel(id: string, signal?: AbortSignal): Promise<ModelDetails> {
  const info = await getJson<ApiModel>(
    `${HUB}/api/models/${id}?${EXPAND}&expand[]=cardData&expand[]=lastModified`,
    signal,
  )
  const config = await getJson<HFConfig>(`${HUB}/${id}/resolve/main/config.json`, signal)
  const card = info.cardData ?? {}
  const base = Array.isArray(card.base_model) ? card.base_model[0] : card.base_model
  return {
    id: info.id,
    arch: parseConfig(config, info.safetensors?.total),
    config,
    license: card.license === 'other' ? (card.license_name ?? 'other') : (card.license ?? null),
    gated: !!info.gated,
    pipelineTag: info.pipeline_tag ?? null,
    downloads: info.downloads ?? 0,
    lastModified: info.lastModified ?? null,
    baseModel: base ?? null,
  }
}

/** Accepts "org/name", or a full huggingface.co URL. */
export function normalizeRepoId(input: string): string | null {
  const s = input.trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//, '').replace(/\/(tree|blob|resolve)\/.*$/, '')
  return /^[\w.-]+\/[\w.-]+$/.test(s) ? s : null
}
