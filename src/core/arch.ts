// Normalizes a Hugging Face `config.json` into the handful of numbers that
// drive memory and compute estimates. Handles dense, MoE, MLA (DeepSeek),
// sliding-window (Gemma, gpt-oss) and hybrid linear-attention / SSM models
// (Qwen3-Next, Granite 4 H, Nemotron-H, LFM2, Jamba).

export type HFConfig = Record<string, unknown>

export interface MlaDims {
  kvLoraRank: number
  qkRopeHeadDim: number
  qkNopeHeadDim: number
  vHeadDim: number
  qLoraRank: number | null
}

export interface MoeDims {
  numExperts: number
  expertsPerToken: number
  expertIntermediate: number
  /** Total intermediate size of always-on shared experts (0 if none). */
  sharedIntermediate: number
  /** Number of layers whose MLP is a MoE block. */
  moeLayers: number
}

export interface LayerMix {
  /** Global (full-context) attention layers. */
  full: number
  /** Sliding-window attention layers (KV capped at `slidingWindow`). */
  sliding: number
  /** Linear-attention / SSM / conv layers with constant per-sequence state. */
  linear: number
}

export interface ModelArch {
  modelType: string
  architectures: string[]
  hiddenSize: number
  numLayers: number
  numHeads: number
  numKVHeads: number
  headDim: number
  intermediateSize: number
  vocabSize: number
  maxContext: number
  tiedEmbeddings: boolean
  gatedMlp: boolean
  layers: LayerMix
  /** Layers that store their own KV (Gemma 3n/4 share KV across the last layers). */
  kvLayers: LayerMix
  /** KV elements per token per layer, for global and sliding layers. */
  kvElems: { full: number; sliding: number }
  slidingWindow: number | null
  mla: MlaDims | null
  moe: MoeDims | null
  /** Per-sequence recurrent state of one linear/SSM layer, in elements. */
  linearStateElems: number
  /** Bytes per recurrent-state element (Qwen3.5 keeps SSM state in fp32). */
  stateBytes: number
  /** Native checkpoint quantization, if the repo ships pre-quantized weights. */
  quantization: string | null
  params: ParamCounts
}

export interface ParamCounts {
  total: number
  /** Parameters doing matmuls per token: excludes the input-embedding lookup
   * and inactive experts (matches how gpt-oss / Qwen report "active"). */
  active: number
  /** Input embedding + untied LM head. */
  embedding: number
  source: 'safetensors' | 'config'
}

const num = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined

function pick(cfg: HFConfig, ...keys: string[]): number | undefined {
  for (const k of keys) {
    const v = num(cfg[k])
    if (v !== undefined) return v
  }
  return undefined
}

/** Multimodal configs keep the language model under `text_config` / `llm_config`. */
export function textConfig(cfg: HFConfig): HFConfig {
  for (const k of ['text_config', 'llm_config', 'language_config']) {
    const sub = cfg[k]
    if (sub && typeof sub === 'object' && num((sub as HFConfig).hidden_size ?? (sub as HFConfig).d_model)) {
      return { ...(sub as HFConfig), quantization_config: cfg.quantization_config }
    }
  }
  return cfg
}

const NON_GATED_MLP = new Set([
  'gpt2', 'gpt_neox', 'gptj', 'opt', 'bloom', 'falcon', 'phi', 'gpt_bigcode', 'starcoder2', 'mpt', 'bert',
])

function layerMix(cfg: HFConfig, numLayers: number, modelType: string): LayerMix {
  const types = cfg.layer_types
  if (Array.isArray(types) && types.length > 0) {
    const mix: LayerMix = { full: 0, sliding: 0, linear: 0 }
    for (const t of types.map(String)) {
      if (t.includes('sliding') || t.includes('local')) mix.sliding++
      else if (t === 'attention' || t.includes('full') || t.includes('global')) mix.full++
      else mix.linear++ // mamba, linear_attention, conv, ...
    }
    return mix
  }
  const fullIdx = cfg.full_attn_idxs
  if (Array.isArray(fullIdx)) return { full: fullIdx.length, sliding: 0, linear: numLayers - fullIdx.length }

  const pattern = cfg.hybrid_override_pattern
  if (typeof pattern === 'string') {
    // Nemotron-H: M = mamba, * = attention, - = MLP-only (no sequence state).
    const full = [...pattern].filter((c) => c === '*').length
    const linear = [...pattern].filter((c) => c === 'M').length
    return { full, sliding: 0, linear }
  }
  const interval = pick(cfg, 'full_attention_interval')
  if (interval) {
    const full = Math.floor(numLayers / interval)
    return { full, sliding: 0, linear: numLayers - full }
  }
  const attnPeriod = pick(cfg, 'attn_layer_period')
  if (attnPeriod) {
    const full = Math.floor(numLayers / attnPeriod)
    return { full, sliding: 0, linear: numLayers - full }
  }
  const window = pick(cfg, 'sliding_window')
  if (window && cfg.use_sliding_window !== false) {
    const p = pick(cfg, 'sliding_window_pattern')
    if (p && p > 1) {
      const full = Math.floor(numLayers / p)
      return { full, sliding: numLayers - full, linear: 0 }
    }
    if (modelType === 'gemma2' || modelType === 'cohere2') {
      const full = Math.floor(numLayers / 2)
      return { full, sliding: numLayers - full, linear: 0 }
    }
    if (window < pick(cfg, 'max_position_embeddings')!) return { full: 0, sliding: numLayers, linear: 0 }
  }
  return { full: numLayers, sliding: 0, linear: 0 }
}

function scaleMix(m: LayerMix, f: number): LayerMix {
  return { full: Math.round(m.full * f), sliding: Math.round(m.sliding * f), linear: Math.round(m.linear * f) }
}

function moeDims(cfg: HFConfig, numLayers: number, hidden: number): MoeDims | null {
  const numExperts = pick(cfg, 'num_experts', 'num_local_experts', 'n_routed_experts', 'moe_num_experts')
  const k = pick(cfg, 'num_experts_per_tok', 'experts_per_token', 'moe_topk', 'top_k')
  if (!numExperts || numExperts <= 1 || !k) return null

  const expertIntermediate =
    pick(cfg, 'moe_intermediate_size', 'expert_intermediate_size') ?? pick(cfg, 'intermediate_size') ?? 4 * hidden
  const nShared = pick(cfg, 'n_shared_experts', 'num_shared_experts') ?? 0
  const sharedIntermediate =
    pick(cfg, 'shared_expert_intermediate_size', 'shared_intermediate_size') ?? nShared * expertIntermediate

  let moeLayers = numLayers
  const firstDense = pick(cfg, 'first_k_dense_replace')
  if (firstDense) moeLayers -= firstDense
  const onlyMlp = cfg.mlp_only_layers
  if (Array.isArray(onlyMlp)) moeLayers -= onlyMlp.length
  const step = pick(cfg, 'decoder_sparse_step', 'moe_layer_freq', 'expert_interval')
  if (step && step > 1) moeLayers = Math.floor(moeLayers / step)
  const pattern = cfg.hybrid_override_pattern
  if (typeof pattern === 'string' && pattern.includes('E')) moeLayers = [...pattern].filter((c) => c === 'E').length

  return { numExperts, expertsPerToken: k, expertIntermediate, sharedIntermediate, moeLayers }
}

function quantMethod(cfg: HFConfig): string | null {
  const q = cfg.quantization_config as HFConfig | undefined
  if (!q) return null
  const method = String(q.quant_method ?? q.quantization_method ?? 'unknown').toLowerCase()
  if (method === 'bitsandbytes') return q.load_in_4bit ? 'bnb-nf4' : 'int8'
  if (method === 'compressed-tensors') {
    const groups = JSON.stringify(q.config_groups ?? {})
    if (groups.includes('"num_bits":4')) return groups.includes('"type":"float"') ? 'nvfp4' : 'int4'
    if (groups.includes('"type":"float"')) return 'fp8'
    return 'int8'
  }
  return method
}

export function parseConfig(raw: HFConfig, safetensorsTotal?: number): ModelArch {
  const cfg = textConfig(raw)
  const modelType = String(cfg.model_type ?? raw.model_type ?? 'unknown')
  const hidden = pick(cfg, 'hidden_size', 'd_model', 'n_embd', 'block_dim')
  const numLayers = pick(cfg, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers')
  if (!hidden || !numLayers) throw new Error('config.json is missing hidden_size / num_hidden_layers')

  const numHeads = pick(cfg, 'num_attention_heads', 'n_head', 'num_heads') ?? hidden / 128
  const numKVHeads = pick(cfg, 'num_key_value_heads', 'n_kv_heads', 'num_kv_heads', 'multi_query_group_num') ??
    (cfg.multi_query === true ? 1 : numHeads)
  const headDim = pick(cfg, 'head_dim', 'attention_head_dim') ?? hidden / numHeads
  const vocabSize = pick(cfg, 'vocab_size', 'padded_vocab_size') ?? 32000
  const maxContext = pick(cfg, 'max_position_embeddings', 'max_sequence_length', 'seq_length', 'n_positions') ?? 4096
  // Most modern families untie; Gemma and small models default to tied.
  const tiedEmbeddings =
    typeof cfg.tie_word_embeddings === 'boolean'
      ? cfg.tie_word_embeddings
      : typeof raw.tie_word_embeddings === 'boolean'
        ? raw.tie_word_embeddings
        : /gemma|lfm2|granite/.test(modelType)

  let intermediateSize = pick(cfg, 'intermediate_size', 'ffn_hidden_size', 'n_inner', 'block_ff_dim') ?? 4 * hidden
  if (modelType === 'lfm2' && cfg.block_auto_adjust_ff_dim !== false) {
    // LFM2 shrinks block_ff_dim like Llama's SwiGLU sizing rule.
    let ff = Math.floor((2 * intermediateSize) / 3)
    ff = Math.floor((num(cfg.block_ffn_dim_multiplier) ?? 1) * ff)
    const multiple = num(cfg.block_multiple_of) ?? 256
    intermediateSize = multiple * Math.ceil(ff / multiple)
  }

  const kvLoraRank = pick(cfg, 'kv_lora_rank')
  const mla: MlaDims | null = kvLoraRank
    ? {
        kvLoraRank,
        qkRopeHeadDim: pick(cfg, 'qk_rope_head_dim') ?? 64,
        qkNopeHeadDim: pick(cfg, 'qk_nope_head_dim') ?? 128,
        vHeadDim: pick(cfg, 'v_head_dim') ?? 128,
        qLoraRank: pick(cfg, 'q_lora_rank') ?? null,
      }
    : null

  const layers = layerMix(cfg, numLayers, modelType)
  const kvShared = pick(cfg, 'num_kv_shared_layers') ?? 0
  const types = cfg.layer_types
  const kvLayers =
    kvShared > 0 && Array.isArray(types)
      ? layerMix({ ...cfg, layer_types: types.slice(0, numLayers - kvShared) }, numLayers - kvShared, modelType)
      : kvShared > 0
        ? scaleMix(layers, (numLayers - kvShared) / numLayers)
        : layers
  const globalHeadDim = pick(cfg, 'global_head_dim') ?? headDim
  const globalKVHeads = pick(cfg, 'num_global_key_value_heads') ?? numKVHeads
  const kvFactor = cfg.attention_k_eq_v === true ? 1 : 2
  const kvElems = mla
    ? { full: mla.kvLoraRank + mla.qkRopeHeadDim, sliding: mla.kvLoraRank + mla.qkRopeHeadDim }
    : { full: kvFactor * globalKVHeads * globalHeadDim, sliding: kvFactor * numKVHeads * headDim }
  const moe = moeDims(cfg, numLayers, hidden)

  const arch: ModelArch = {
    modelType,
    architectures: (raw.architectures as string[] | undefined) ?? [],
    hiddenSize: hidden,
    numLayers,
    numHeads,
    numKVHeads,
    headDim,
    intermediateSize,
    vocabSize,
    maxContext,
    tiedEmbeddings,
    gatedMlp: !NON_GATED_MLP.has(modelType),
    layers,
    kvLayers,
    kvElems,
    slidingWindow: layers.sliding > 0 ? (pick(cfg, 'sliding_window', 'attention_window_size') ?? null) : null,
    mla,
    moe,
    linearStateElems: linearStateElems(cfg, hidden),
    stateBytes: String(cfg.mamba_ssm_dtype ?? '').includes('32') ? 4 : 2,
    quantization: quantMethod(cfg),
    params: { total: 0, active: 0, embedding: 0, source: 'config' },
  }
  arch.params = countParams(arch, cfg, safetensorsTotal)
  return arch
}

/** Recurrent state per sequence per linear layer (SSM state + conv cache), in elements. */
function linearStateElems(cfg: HFConfig, hidden: number): number {
  const dState = pick(cfg, 'mamba_d_state', 'ssm_state_size', 'state_size')
  if (dState) {
    const nHeads = pick(cfg, 'mamba_n_heads', 'mamba_num_heads')
    const dHead = pick(cfg, 'mamba_d_head', 'mamba_head_dim')
    const dInner = nHeads && dHead ? nHeads * dHead : (pick(cfg, 'mamba_expand', 'expand') ?? 2) * hidden
    const groups = pick(cfg, 'mamba_n_groups', 'n_groups') ?? 1
    const conv = pick(cfg, 'mamba_d_conv', 'conv_kernel') ?? 4
    const ssm = nHeads && dHead ? nHeads * dHead * dState : dInner * dState
    return ssm + (dInner + 2 * groups * dState) * (conv - 1)
  }
  const nv = pick(cfg, 'linear_num_value_heads')
  if (nv) {
    // Gated DeltaNet (Qwen3-Next): per-head dk x dv state + short conv.
    const nk = pick(cfg, 'linear_num_key_heads') ?? nv
    const dk = pick(cfg, 'linear_key_head_dim') ?? 128
    const dv = pick(cfg, 'linear_value_head_dim') ?? 128
    const conv = pick(cfg, 'linear_conv_kernel_dim') ?? 4
    return nv * dk * dv + (2 * nk * dk + nv * dv) * (conv - 1)
  }
  const lCache = pick(cfg, 'conv_L_cache')
  if (lCache) return hidden * lCache // LFM2 short conv
  return 0
}

function attentionParams(a: ModelArch): number {
  const h = a.hiddenSize
  if (a.mla) {
    const m = a.mla
    const qk = m.qkNopeHeadDim + m.qkRopeHeadDim
    const q = m.qLoraRank ? h * m.qLoraRank + m.qLoraRank * a.numHeads * qk : h * a.numHeads * qk
    const kvA = h * (m.kvLoraRank + m.qkRopeHeadDim)
    const kvB = m.kvLoraRank * a.numHeads * (m.qkNopeHeadDim + m.vHeadDim)
    return q + kvA + kvB + a.numHeads * m.vHeadDim * h
  }
  return h * a.numHeads * a.headDim * 2 + h * a.numKVHeads * a.headDim * 2
}

function linearLayerParams(a: ModelArch, cfg: HFConfig): number {
  const h = a.hiddenSize
  const nHeads = pick(cfg, 'mamba_n_heads')
  const dHead = pick(cfg, 'mamba_d_head', 'mamba_head_dim')
  const dState = pick(cfg, 'mamba_d_state', 'ssm_state_size')
  if (dState) {
    const dInner = nHeads && dHead ? nHeads * dHead : (pick(cfg, 'mamba_expand') ?? 2) * h
    const groups = pick(cfg, 'mamba_n_groups', 'n_groups') ?? 1
    const heads = nHeads ?? dInner / 64
    return h * (2 * dInner + 2 * groups * dState + heads) + dInner * h
  }
  const nv = pick(cfg, 'linear_num_value_heads')
  if (nv) {
    const nk = pick(cfg, 'linear_num_key_heads') ?? nv
    const dk = pick(cfg, 'linear_key_head_dim') ?? 128
    const dv = pick(cfg, 'linear_value_head_dim') ?? 128
    return h * (2 * nk * dk + 2 * nv * dv) + h * 2 * nv + nv * dv * h
  }
  if (pick(cfg, 'conv_L_cache')) return 4 * h * h // LFM2: in_proj (3h) + out_proj (h)
  return attentionParams(a)
}

function countParams(a: ModelArch, cfg: HFConfig, safetensorsTotal?: number): ParamCounts {
  const h = a.hiddenSize
  const mlpMats = a.gatedMlp ? 3 : 2
  // Gemma 3n/4 per-layer embeddings: large lookup tables, no matmuls.
  const perLayerEmbed = (pick(cfg, 'vocab_size_per_layer_input') ?? 0) * (pick(cfg, 'hidden_size_per_layer_input') ?? 0) * a.numLayers
  const inputEmbedding = a.vocabSize * h + perLayerEmbed
  const embedding = inputEmbedding + (a.tiedEmbeddings ? 0 : a.vocabSize * h)
  const attnLayers = a.layers.full + a.layers.sliding
  const mixer = attnLayers * attentionParams(a) + a.layers.linear * linearLayerParams(a, cfg)

  let mlp = a.numLayers * mlpMats * h * a.intermediateSize
  let inactive = 0
  if (a.moe) {
    const m = a.moe
    const denseLayers = a.numLayers - m.moeLayers
    const expert = mlpMats * h * m.expertIntermediate
    const perMoeLayer = m.numExperts * expert + mlpMats * h * m.sharedIntermediate + h * m.numExperts
    mlp = denseLayers * mlpMats * h * a.intermediateSize + m.moeLayers * perMoeLayer
    inactive = m.moeLayers * (m.numExperts - m.expertsPerToken) * expert
  }
  const norms = a.numLayers * 2 * h + h
  const fromConfig = embedding + mixer + mlp + norms

  // Pre-quantized repos report packed tensor sizes, so trust the config there.
  const useSafetensors = !!safetensorsTotal && !a.quantization
  const total = useSafetensors ? safetensorsTotal : fromConfig
  const activeRatio = (fromConfig - inactive - inputEmbedding) / fromConfig
  return {
    total,
    active: Math.round(total * activeRatio),
    embedding,
    source: useSafetensors ? 'safetensors' : 'config',
  }
}

/** KV / state bytes held by one sequence of `tokens` length (all layers, unsharded). */
export function sequenceCacheBytes(a: ModelArch, tokens: number, kvBytes: number): number {
  const slidingTokens = a.slidingWindow ? Math.min(tokens, a.slidingWindow) : tokens
  return (
    kvBytes * (a.kvElems.full * a.kvLayers.full * tokens + a.kvElems.sliding * a.kvLayers.sliding * slidingTokens) +
    a.kvLayers.linear * a.linearStateElems * a.stateBytes
  )
}

/** KV bytes per token for full-attention layers only (what prefix caching can share). */
export function fullAttentionKvBytesPerToken(a: ModelArch, kvBytes: number): number {
  return a.kvElems.full * kvBytes * a.kvLayers.full
}

/** Constant per-sequence recurrent state (SSM / linear attention), in bytes. */
export function linearStateBytes(a: ModelArch): number {
  return a.kvLayers.linear * a.linearStateElems * a.stateBytes
}

export function attentionKind(a: ModelArch): string {
  if (a.mla) return 'MLA'
  if (a.numKVHeads === 1) return 'MQA'
  if (a.numKVHeads < a.numHeads) return `GQA (${a.numHeads / a.numKVHeads}:1)`
  return 'MHA'
}
