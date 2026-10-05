// Training memory and time model for full fine-tuning, LoRA, QLoRA and GRPO,
// with DDP / ZeRO-1/2/3 (FSDP) sharding and CPU offload.

import type { ModelArch } from './arch'
import { sequenceCacheBytes } from './arch'
import { peakFlops, type Gpu } from './hardware'
import { decodeStepTime, DEFAULT_ENGINE, type Shape } from './inference'
import { weightBytes } from './precision'

export type TrainMethod = 'full' | 'lora' | 'qlora'
export type Optimizer = 'adamw' | 'adamw-8bit' | 'adafactor' | 'sgd'
export type Sharding = 'ddp' | 'zero1' | 'zero2' | 'zero3'
export type Checkpointing = 'none' | 'standard' | 'offload'
export type LoraTargets = 'attention' | 'all-linear'

export interface TrainConfig {
  method: TrainMethod
  optimizer: Optimizer
  sharding: Sharding
  /** Move optimizer states (and with ZeRO-3, params) to CPU RAM. */
  cpuOffload: boolean
  checkpointing: Checkpointing
  /** Fused / chunked cross-entropy (Liger, Unsloth) avoids materializing full logits. */
  chunkedLoss: boolean
  seqLen: number
  microBatch: number
  gradAccum: number
  loraRank: number
  loraTargets: LoraTargets
  /** Dataset tokens per epoch (prompt + completion, after packing). */
  datasetTokens: number
  epochs: number
  /** Model FLOPs utilization achieved by the trainer. */
  mfu: number
  rl: GrpoConfig | null
}

export interface GrpoConfig {
  /** Completions sampled per prompt (G). */
  numGenerations: number
  promptsPerStep: number
  promptTokens: number
  completionTokens: number
  totalPrompts: number
  /** Rollouts served by a colocated vLLM sharing the GPU. */
  colocateVllm: boolean
  /** Unsloth-style weight sharing between trainer and vLLM. */
  shareWeights: boolean
  /** KL penalty needs a frozen reference model (free for LoRA: disable adapter). */
  klBeta: number
}

export const DEFAULT_TRAIN: TrainConfig = {
  method: 'lora',
  optimizer: 'adamw',
  sharding: 'ddp',
  cpuOffload: false,
  checkpointing: 'standard',
  chunkedLoss: true,
  seqLen: 4096,
  microBatch: 2,
  gradAccum: 8,
  loraRank: 16,
  loraTargets: 'all-linear',
  datasetTokens: 50_000_000,
  epochs: 2,
  mfu: 0.35,
  rl: null,
}

export const DEFAULT_GRPO: GrpoConfig = {
  numGenerations: 8,
  promptsPerStep: 4,
  promptTokens: 2048,
  completionTokens: 512,
  totalPrompts: 2000,
  colocateVllm: true,
  shareWeights: true,
  klBeta: 0,
}

export interface TrainMemory {
  /** Bytes per GPU. */
  weights: number
  gradients: number
  optimizer: number
  activations: number
  logits: number
  rollout: number
  overhead: number
  total: number
  trainableParams: number
}

const OPT_BYTES: Record<Optimizer, number> = {
  // fp32 master copy (4) + optimizer states.
  adamw: 4 + 8,
  'adamw-8bit': 4 + 2,
  adafactor: 4 + 0.1,
  sgd: 4 + 4,
}

export function loraParams(a: ModelArch, rank: number, targets: LoraTargets): number {
  const h = a.hiddenSize
  const q = a.numHeads * a.headDim
  const kv = a.numKVHeads * a.headDim
  const attnLayers = a.layers.full + a.layers.sliding
  const attn = rank * ((h + q) * 2 + (h + kv) * 2) * attnLayers
  if (targets === 'attention') return attn
  const ffn = a.moe ? a.moe.numExperts * a.moe.expertIntermediate + a.moe.sharedIntermediate : a.intermediateSize
  const mlpLayers = a.moe ? a.moe.moeLayers : a.numLayers
  const mlp = rank * (h + ffn) * (a.gatedMlp ? 3 : 2) * mlpLayers
  return attn + mlp
}

/** Saved activation bytes per token per layer without checkpointing (bf16, flash attention). */
function activationPerTokenLayer(a: ModelArch): number {
  const ffn = a.moe ? a.moe.expertsPerToken * a.moe.expertIntermediate + a.moe.sharedIntermediate : a.intermediateSize
  return 2 * (8 * a.hiddenSize + 4 * ffn)
}

export function trainMemory(a: ModelArch, cfg: TrainConfig, gpusTotal: number): TrainMemory {
  const n = Math.max(1, gpusTotal)
  const shardParams = cfg.sharding === 'zero3' ? n : 1
  const shardGrads = cfg.sharding === 'zero2' || cfg.sharding === 'zero3' ? n : 1
  const shardOpt = cfg.sharding === 'ddp' ? 1 : n

  const full = cfg.method === 'full'
  const trainable = full ? a.params.total : loraParams(a, cfg.loraRank, cfg.loraTargets)
  const base =
    cfg.method === 'qlora' ? weightBytes(a.params.total, a.params.embedding, 'bnb-nf4') : a.params.total * 2
  const weightsOnGpu = cfg.cpuOffload && cfg.sharding === 'zero3' ? 0.1 * base : base
  const weights = weightsOnGpu / shardParams + (full ? 0 : (trainable * 2) / shardParams)
  const gradients = (trainable * 2) / shardGrads
  const optimizer = cfg.cpuOffload ? 0 : (trainable * OPT_BYTES[cfg.optimizer]) / shardOpt

  const rl = cfg.rl
  const seq = rl ? rl.promptTokens + rl.completionTokens : cfg.seqLen
  const tokens = cfg.microBatch * seq
  const perLayer = activationPerTokenLayer(a)
  const activations =
    cfg.checkpointing === 'none'
      ? tokens * perLayer * a.numLayers
      : cfg.checkpointing === 'standard'
        ? tokens * (a.hiddenSize * 2 * a.numLayers + perLayer)
        : tokens * perLayer * 1.5 // layer inputs offloaded to CPU RAM
  // bf16 logits + fp32 upcast for the loss; chunking keeps ~1/8 alive.
  const logits = tokens * a.vocabSize * 6 * (cfg.chunkedLoss ? 0.125 : 1)

  let rollout = 0
  if (rl) {
    if (rl.colocateVllm) {
      const seqs = rl.numGenerations * rl.promptsPerStep
      rollout += seqs * sequenceCacheBytes(a, rl.promptTokens + rl.completionTokens, 2) + 1.0e9
      if (!rl.shareWeights) rollout += a.params.total * 2
    }
    if (rl.klBeta > 0 && full) rollout += (a.params.total * 2) / shardParams
  }

  const overhead = 1.5e9
  const subtotal = weights + gradients + optimizer + activations + logits + rollout + overhead
  return {
    weights,
    gradients,
    optimizer,
    activations,
    logits,
    rollout,
    overhead,
    // ~8% allocator fragmentation.
    total: subtotal * 1.08,
    trainableParams: trainable,
  }
}

export interface TrainTime {
  /** Seconds for the whole run. */
  seconds: number
  tokensPerSecond: number
  steps: number
  /** Seconds spent generating rollouts (GRPO only). */
  rolloutSeconds: number
}

export function trainTime(a: ModelArch, cfg: TrainConfig, gpu: Gpu, gpusTotal: number, nodes: number): TrainTime {
  const flopsPeak = peakFlops(gpu, 'bf16') * cfg.mfu * gpusTotal * (nodes > 1 ? 0.9 : 1)
  const recompute = cfg.checkpointing === 'none' ? 0 : 2
  const matmul = cfg.method === 'full' ? 6 + recompute : 4 + recompute
  const slowdown = cfg.method === 'qlora' ? 1.3 : 1
  const attnContext = (cfg.rl ? cfg.rl.promptTokens + cfg.rl.completionTokens : cfg.seqLen) / 2
  const window = a.slidingWindow ?? attnContext
  const pairs = a.layers.full * attnContext + a.layers.sliding * Math.min(window, attnContext)
  const attn = 3 * 4 * a.numHeads * a.headDim * pairs
  const flopsPerToken = (matmul * a.params.active + attn) * slowdown

  const rl = cfg.rl
  if (!rl) {
    const tokens = cfg.datasetTokens * cfg.epochs
    const seconds = (tokens * flopsPerToken) / flopsPeak
    const steps = Math.ceil(tokens / (cfg.seqLen * cfg.microBatch * cfg.gradAccum * gpusTotal))
    return { seconds, tokensPerSecond: tokens / seconds, steps, rolloutSeconds: 0 }
  }

  const steps = Math.ceil((rl.totalPrompts * cfg.epochs) / rl.promptsPerStep)
  const seqs = rl.numGenerations * rl.promptsPerStep
  const tokensPerStep = seqs * (rl.promptTokens + rl.completionTokens)
  const trainSeconds = (tokensPerStep * flopsPerToken) / flopsPeak
  // Rollouts: one engine per GPU decodes its share of sequences in parallel.
  const shape: Shape = { gpu, tp: 1, weightFormat: 'bf16', kvFormat: 'bf16' }
  const perGpu = Math.max(1, Math.ceil(seqs / gpusTotal))
  const step = decodeStepTime(a, shape, perGpu, rl.promptTokens + rl.completionTokens / 2, DEFAULT_ENGINE)
  const genSeconds = rl.completionTokens * step + (perGpu * rl.promptTokens * 2 * a.params.active) / (peakFlops(gpu, 'bf16') * 0.5)
  const seconds = steps * (trainSeconds + genSeconds)
  return { seconds, tokensPerSecond: (steps * tokensPerStep) / seconds, steps, rolloutSeconds: steps * genSeconds }
}
