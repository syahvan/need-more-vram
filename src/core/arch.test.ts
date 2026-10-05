import { describe, expect, it } from 'vitest'
import { parseConfig, sequenceCacheBytes, type HFConfig } from './arch'

const fixtures = import.meta.glob<HFConfig>('./__fixtures__/*.json', { eager: true, import: 'default' })
const load = (name: string) => fixtures[`./__fixtures__/${name}.json`]

// Reference totals from the Hugging Face API (safetensors metadata).
const reference: Record<string, { total: number; active?: number }> = {
  'unsloth_Llama-3.1-8B': { total: 8_030_261_248 },
  'Qwen_Qwen3-8B': { total: 8_190_735_360 },
  'mistralai_Mistral-7B-Instruct-v0.3': { total: 7_248_023_552 },
  'Qwen_Qwen3-30B-A3B': { total: 30_532_122_624, active: 3.3e9 },
  // Qwen markets this as "A3B"; counting the LM head gives ~3.5B.
  'Qwen_Qwen3-Next-80B-A3B-Instruct': { total: 81_324_862_720, active: 3.5e9 },
  'openai_gpt-oss-20b': { total: 20_914_757_184, active: 3.6e9 },
  'ibm-granite_granite-4.0-h-small': { total: 32_207_337_984, active: 9e9 },
  'deepseek-ai_DeepSeek-V3': { total: 671e9, active: 37e9 },
  'LiquidAI_LFM2-1.2B': { total: 1_170_340_608 },
  // Includes the ~0.4B SigLIP vision tower we don't model.
  'unsloth_gemma-3-4b-it': { total: 3.88e9 },
}

describe('parameter counting from config.json', () => {
  for (const [name, ref] of Object.entries(reference)) {
    it(`${name} is within 3% of the published total`, () => {
      const a = parseConfig(load(name))
      expect(a.params.source).toBe('config')
      expect(Math.abs(a.params.total - ref.total) / ref.total).toBeLessThan(0.03)
      if (ref.active) expect(Math.abs(a.params.active - ref.active) / ref.active).toBeLessThan(0.1)
    })
  }

  it('prefers the safetensors total for unquantized repos and keeps the active ratio', () => {
    const a = parseConfig(load('Qwen_Qwen3-30B-A3B'), 30_532_122_624)
    expect(a.params.source).toBe('safetensors')
    expect(a.params.total).toBe(30_532_122_624)
    expect(a.params.active / a.params.total).toBeCloseTo(0.11, 1)
  })

  it('ignores safetensors totals for pre-quantized repos', () => {
    const a = parseConfig(load('openai_gpt-oss-20b'), 99)
    expect(a.quantization).toBe('mxfp4')
    expect(a.params.source).toBe('config')
  })
})

describe('layer mix and KV cache', () => {
  it('Llama 3.1 8B: 128 KiB of bf16 KV per token', () => {
    const a = parseConfig(load('unsloth_Llama-3.1-8B'))
    expect(a.layers).toEqual({ full: 32, sliding: 0, linear: 0 })
    expect(sequenceCacheBytes(a, 1, 2)).toBe(32 * 2 * 8 * 128 * 2)
  })

  it('Gemma 3: 5 of 6 layers are sliding (1024 window)', () => {
    const a = parseConfig(load('unsloth_gemma-3-4b-it'))
    expect(a.layers).toEqual({ full: 5, sliding: 29, linear: 0 })
    expect(a.slidingWindow).toBe(1024)
    const perTok = 2 * 4 * 256 * 2
    expect(sequenceCacheBytes(a, 32_768, 2)).toBe(perTok * (5 * 32_768 + 29 * 1024))
  })

  it('gpt-oss alternates sliding (128) and full layers', () => {
    const a = parseConfig(load('openai_gpt-oss-20b'))
    expect(a.layers).toEqual({ full: 12, sliding: 12, linear: 0 })
    expect(a.slidingWindow).toBe(128)
  })

  it('DeepSeek V3 uses MLA latent cache (576 per token per layer)', () => {
    const a = parseConfig(load('deepseek-ai_DeepSeek-V3'))
    expect(a.mla?.kvLoraRank).toBe(512)
    expect(sequenceCacheBytes(a, 1, 2)).toBe(61 * 576 * 2)
  })

  it('hybrids keep KV only on attention layers plus a constant state', () => {
    const next = parseConfig(load('Qwen_Qwen3-Next-80B-A3B-Instruct'))
    expect(next.layers).toEqual({ full: 12, sliding: 0, linear: 36 })
    const granite = parseConfig(load('ibm-granite_granite-4.0-h-small'))
    expect(granite.layers).toEqual({ full: 4, sliding: 0, linear: 36 })
    const lfm = parseConfig(load('LiquidAI_LFM2-1.2B'))
    expect(lfm.layers).toEqual({ full: 6, sliding: 0, linear: 10 })
    // State is length-independent: growing the context only grows attention KV.
    const d = sequenceCacheBytes(granite, 2000, 2) - sequenceCacheBytes(granite, 1000, 2)
    expect(d).toBe(1000 * 4 * 2 * 8 * 128 * 2)
  })

  it('Mistral v0.3 has no sliding window', () => {
    const a = parseConfig(load('mistralai_Mistral-7B-Instruct-v0.3'))
    expect(a.layers.sliding).toBe(0)
  })
})

describe('2026 architectures', () => {
  it('Qwen3.5-4B: 8 of 32 layers keep KV; DeltaNet state in fp32', () => {
    const a = parseConfig(load('Qwen_Qwen3.5-4B'), 4_659_865_088)
    expect(a.layers).toEqual({ full: 8, sliding: 0, linear: 24 })
    expect(a.stateBytes).toBe(4)
    // 8 layers × 2 × 4 KV heads × 256 dims × 2 bytes = 32 KiB per token.
    expect(sequenceCacheBytes(a, 2, 2) - sequenceCacheBytes(a, 1, 2)).toBe(32 * 1024)
  })

  it('Gemma 4 E4B: wider global heads, KV shared across the last 18 layers, per-layer embeddings', () => {
    const a = parseConfig(load('google_gemma-4-E4B-it'), 7_996_156_490)
    expect(a.layers.full + a.layers.sliding).toBe(42)
    expect(a.kvLayers.full + a.kvLayers.sliding).toBe(24)
    expect(a.kvElems.full).toBe(2 * 2 * 512)
    expect(a.kvElems.sliding).toBe(2 * 2 * 256)
    // Per-layer embedding tables are ~2.8B of the 8B total but do no matmuls.
    expect(a.params.active).toBeLessThan(4.5e9)
    expect(a.params.embedding).toBeGreaterThan(2.8e9)
  })
})
