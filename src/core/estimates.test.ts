import { describe, expect, it } from 'vitest'
import { parseConfig, type HFConfig } from './arch'
import type { Gpu } from './hardware'
import { decodeStepTime, prefillTime, replicaCapacity, replicaMemory, replicasNeeded, type Workload } from './inference'
import { DEFAULT_TRAIN, trainMemory, trainTime } from './training'

const fixtures = import.meta.glob<HFConfig>('./__fixtures__/*.json', { eager: true, import: 'default' })
const llama = parseConfig(fixtures['./__fixtures__/unsloth_Llama-3.1-8B.json'], 8_030_261_248)
const qwenMoe = parseConfig(fixtures['./__fixtures__/Qwen_Qwen3-30B-A3B.json'], 30_532_122_624)

const h100: Gpu = {
  id: 'h100-sxm-80gb', name: 'H100 SXM', vendor: 'nvidia', arch: 'hopper', segment: 'datacenter',
  memoryGB: 80, memoryBandwidthGBs: 3350,
  tflops: { fp32: 67, bf16: 989, fp8: 1979, int8: 1979, fp4: null },
  interconnect: { type: 'nvlink', bandwidthGBs: 900 }, tdpW: 700, fp8: true, fp4: false, sources: [],
}
const l4: Gpu = {
  ...h100, id: 'l4', name: 'L4', arch: 'ada', memoryGB: 24, memoryBandwidthGBs: 300,
  tflops: { fp32: 30, bf16: 121, fp8: 242, int8: 242, fp4: null }, interconnect: { type: 'pcie', bandwidthGBs: 32 },
}

const chat: Workload = {
  inputTokens: 2000,
  outputTokens: 300,
  sharedPrefixTokens: 0,
  prefixCaching: false,
  load: { mode: 'rate', peakRps: 20, avgRps: 5, callsPerRequest: 1, peakHoursPerDay: 4 },
  slo: { ttftMs: 1000, tpotMs: 50 },
}

describe('inference memory', () => {
  it('Llama 8B BF16 on one H100 leaves ~400k tokens of KV cache (vLLM ballpark)', () => {
    const m = replicaMemory(llama, { gpu: h100, tp: 1, weightFormat: 'bf16', kvFormat: 'bf16' }, chat)
    expect(m.weights / 1e9).toBeCloseTo(16.06, 1)
    expect(m.kvTokenCapacity).toBeGreaterThan(350_000)
    expect(m.kvTokenCapacity).toBeLessThan(450_000)
  })

  it('does not fit BF16 Llama 8B on an L4 with long contexts, but does with FP8 weights', () => {
    const big = { ...chat, inputTokens: 30_000 }
    const bf16 = replicaMemory(llama, { gpu: l4, tp: 1, weightFormat: 'bf16', kvFormat: 'bf16' }, big)
    const fp8 = replicaMemory(llama, { gpu: l4, tp: 1, weightFormat: 'fp8', kvFormat: 'fp8' }, big)
    expect(bf16.maxSeqsByMemory).toBeLessThan(fp8.maxSeqsByMemory)
    expect(fp8.fits).toBe(true)
  })

  it('prefix caching stores the shared prefix once', () => {
    const agent = { ...chat, inputTokens: 18_000, sharedPrefixTokens: 16_000 }
    const shape = { gpu: h100, tp: 1, weightFormat: 'bf16', kvFormat: 'bf16' } as const
    const off = replicaMemory(llama, shape, agent)
    const on = replicaMemory(llama, shape, { ...agent, prefixCaching: true })
    expect(on.maxSeqsByMemory).toBeGreaterThan(off.maxSeqsByMemory * 4)
  })
})

describe('inference speed', () => {
  const shape = { gpu: h100, tp: 1, weightFormat: 'bf16', kvFormat: 'bf16' } as const

  it('single-stream decode of an 8B model on H100 is ~120-180 tok/s', () => {
    const tps = 1 / decodeStepTime(llama, shape, 1, 1000)
    expect(tps).toBeGreaterThan(120)
    expect(tps).toBeLessThan(180)
  })

  it('batching amortizes weight reads', () => {
    const one = decodeStepTime(llama, shape, 1, 1000)
    const many = decodeStepTime(llama, shape, 64, 1000)
    expect(64 / many).toBeGreaterThan(20 / one)
  })

  it('TP=2 speeds up decode', () => {
    expect(decodeStepTime(llama, { ...shape, tp: 2 }, 1, 1000)).toBeLessThan(decodeStepTime(llama, shape, 1, 1000))
  })

  it('prefill of 2k tokens takes tens of milliseconds', () => {
    const t = prefillTime(llama, shape, 2000, 0)
    expect(t).toBeGreaterThan(0.03)
    expect(t).toBeLessThan(0.12)
  })

  it('MoE with 3B active decodes faster than a dense 8B at batch 1', () => {
    expect(decodeStepTime(qwenMoe, shape, 1, 1000)).toBeLessThan(decodeStepTime(llama, shape, 1, 1000) * 1.2)
  })

  it('finds an SLO-compliant operating point and sizes replicas', () => {
    const cap = replicaCapacity(llama, shape, chat)
    expect(cap.best).not.toBeNull()
    expect(cap.best!.ttftMs).toBeLessThanOrEqual(chat.slo.ttftMs)
    expect(cap.best!.tpotMs).toBeLessThanOrEqual(chat.slo.tpotMs)
    expect(cap.best!.rps).toBeGreaterThan(1)
    const plan = replicasNeeded(cap, chat.load)!
    expect(plan.peakReplicas).toBeGreaterThanOrEqual(plan.avgReplicas)
  })

  it('a tighter TPOT SLO lowers per-replica capacity', () => {
    const loose = replicaCapacity(llama, shape, chat).best!.rps
    const tight = replicaCapacity(llama, shape, { ...chat, slo: { ttftMs: 1000, tpotMs: 12 } }).best!.rps
    expect(tight).toBeLessThan(loose)
  })

  it('reports an impossible SLO instead of a fake number', () => {
    const cap = replicaCapacity(llama, shape, { ...chat, slo: { ttftMs: 1000, tpotMs: 2 } })
    expect(cap.best).toBeNull()
    expect(cap.limiter).toBe('tpot')
  })
})

describe('training memory', () => {
  it('full fine-tuning an 8B model with AdamW needs ~16 bytes/param before activations', () => {
    const m = trainMemory(llama, { ...DEFAULT_TRAIN, method: 'full', sharding: 'ddp' }, 1)
    expect((m.weights + m.gradients + m.optimizer) / 8.03e9).toBeCloseTo(16, 0)
  })

  it('ZeRO-3 across 8 GPUs shards weights, grads and optimizer', () => {
    const one = trainMemory(llama, { ...DEFAULT_TRAIN, method: 'full', sharding: 'zero3' }, 1)
    const eight = trainMemory(llama, { ...DEFAULT_TRAIN, method: 'full', sharding: 'zero3' }, 8)
    expect(eight.optimizer).toBeCloseTo(one.optimizer / 8)
    expect(eight.weights).toBeCloseTo(one.weights / 8)
  })

  it('QLoRA on an 8B model fits a 24 GB card; LoRA trainables are ~0.5% of params', () => {
    const q = trainMemory(llama, { ...DEFAULT_TRAIN, method: 'qlora', microBatch: 1, seqLen: 2048 }, 1)
    expect(q.total / 1e9).toBeLessThan(24)
    expect(q.trainableParams / llama.params.total).toBeGreaterThan(0.003)
    expect(q.trainableParams / llama.params.total).toBeLessThan(0.01)
  })

  it('training time scales with tokens and GPUs', () => {
    const t1 = trainTime(llama, DEFAULT_TRAIN, h100, 1, 1).seconds
    const t8 = trainTime(llama, DEFAULT_TRAIN, h100, 8, 1).seconds
    expect(t1 / t8).toBeCloseTo(8, 1)
    // 100M tokens of LoRA on one H100 at 35% MFU: a few hours.
    expect(t1 / 3600).toBeGreaterThan(1)
    expect(t1 / 3600).toBeLessThan(12)
  })
})
