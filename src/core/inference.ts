// Roofline-style serving model for a continuous-batching engine (vLLM / SGLang).
// Decode is bounded by max(bytes moved / bandwidth, FLOPs / compute); prefill is
// compute-bound. Queueing is approximated so the result can be checked
// against latency SLOs. These are estimates — always confirm with a load test.

import { sequenceCacheBytes, fullAttentionKvBytesPerToken, type ModelArch } from './arch'
import { linkBandwidth, peakFlops, type Gpu } from './hardware'
import { computePrecision, KV_FORMATS, WEIGHT_FORMATS, weightBytes, type KvFormat, type WeightFormat } from './precision'

export interface EngineAssumptions {
  /** Fraction of GPU memory the engine may use (vLLM `--gpu-memory-utilization`). */
  gpuMemoryUtilization: number
  /** Achieved fraction of peak HBM bandwidth during decode. */
  bandwidthEfficiency: number
  /** Achieved fraction of peak FLOPs during prefill (MFU). */
  prefillMfu: number
  /** Achieved fraction of peak FLOPs in compute-bound decode steps. */
  decodeMfu: number
  /** Fixed per-step cost: scheduling, sampling, kernel launches (ms). */
  stepOverheadMs: number
  /** Token budget per forward pass (vLLM `--max-num-batched-tokens`). */
  maxBatchedTokens: number
  /** Max sequences per batch (vLLM `--max-num-seqs`). */
  maxNumSeqs: number
  /** CUDA context, CUDA graphs, NCCL buffers, allocator slack (GB per GPU). */
  runtimeOverheadGB: number
  /** Autoscaling target: run replicas at this fraction of their SLO-safe capacity. */
  targetUtilization: number
}

export const DEFAULT_ENGINE: EngineAssumptions = {
  gpuMemoryUtilization: 0.9,
  bandwidthEfficiency: 0.8,
  prefillMfu: 0.55,
  decodeMfu: 0.45,
  stepOverheadMs: 0.6,
  maxBatchedTokens: 8192,
  maxNumSeqs: 256,
  runtimeOverheadGB: 1.5,
  targetUtilization: 0.8,
}

export interface Slo {
  /** p50-ish time to first token target (ms). */
  ttftMs: number
  /** Time per output token target (ms); 1000 / min tokens-per-second per user. */
  tpotMs: number
}

export type LoadSpec =
  | { mode: 'concurrent'; concurrentStreams: number }
  | {
      mode: 'rate'
      /** Incoming user requests per second (peak and daily average). */
      peakRps: number
      avgRps: number
      /** LLM calls triggered per user request (agents often make several). */
      callsPerRequest: number
      /** Hours per day spent near peak, for autoscaled cost. */
      peakHoursPerDay: number
    }

export interface Workload {
  inputTokens: number
  outputTokens: number
  /** Tokens shared by every request (system prompt + tool schemas). */
  sharedPrefixTokens: number
  prefixCaching: boolean
  load: LoadSpec
  slo: Slo
}

export interface Shape {
  gpu: Gpu
  /** Tensor-parallel degree (GPUs per replica). */
  tp: number
  weightFormat: WeightFormat
  kvFormat: KvFormat
}

export interface ReplicaMemory {
  /** All values in bytes, per GPU. */
  weights: number
  activations: number
  overhead: number
  kvBudget: number
  usable: number
  total: number
  fits: boolean
  /** Sequences of the workload's length whose KV fits in one replica. */
  maxSeqsByMemory: number
  /** Total cacheable tokens across the replica (full-attention layers). */
  kvTokenCapacity: number
}

const GB = 1e9

function kvShard(a: ModelArch, tp: number): number {
  // MLA's latent cache is replicated across TP ranks; GQA shards by KV head.
  return a.mla ? 1 : Math.max(1, Math.min(tp, a.numKVHeads))
}

function ffnWidth(a: ModelArch): number {
  return a.moe ? a.moe.expertsPerToken * a.moe.expertIntermediate + a.moe.sharedIntermediate : a.intermediateSize
}

export function replicaMemory(a: ModelArch, s: Shape, w: Workload, e: EngineAssumptions = DEFAULT_ENGINE): ReplicaMemory {
  const kvB = KV_FORMATS[s.kvFormat].bytes
  const total = s.gpu.memoryGB * GB
  const usable = total * e.gpuMemoryUtilization
  const weights = weightBytes(a.params.total, a.params.embedding, s.weightFormat) / s.tp
  const activations =
    e.maxBatchedTokens * 2 * (4 * a.hiddenSize + (2 * ffnWidth(a)) / s.tp) + (e.maxNumSeqs * a.vocabSize * 4) / s.tp
  const overhead = e.runtimeOverheadGB * GB
  const kvBudget = usable - weights - activations - overhead

  const seqLen = w.inputTokens + w.outputTokens
  const shard = kvShard(a, s.tp)
  const shared = w.prefixCaching ? fullAttentionKvBytesPerToken(a, kvB) * Math.min(w.sharedPrefixTokens, w.inputTokens) : 0
  const perSeq = sequenceCacheBytes(a, seqLen, kvB) - shared
  const replicaKv = Math.max(0, kvBudget) * shard
  const maxSeqsByMemory = kvBudget > 0 ? Math.max(0, Math.floor((replicaKv - shared) / Math.max(perSeq, 1))) : 0
  const perTok = fullAttentionKvBytesPerToken(a, kvB) || sequenceCacheBytes(a, 1, kvB)
  return {
    weights,
    activations,
    overhead,
    kvBudget,
    usable,
    total,
    fits: kvBudget > 0 && maxSeqsByMemory >= 1,
    maxSeqsByMemory,
    kvTokenCapacity: Math.floor(replicaKv / perTok),
  }
}

function allReduceTime(s: Shape, tokens: number, a: ModelArch): number {
  if (s.tp <= 1) return 0
  const latency = s.gpu.interconnect.type === 'pcie' ? 25e-6 : 8e-6
  const bytes = tokens * a.hiddenSize * 2
  const per = latency + ((2 * (s.tp - 1)) / s.tp) * (bytes / linkBandwidth(s.gpu))
  return a.numLayers * 2 * per
}

function attentionFlopsPerToken(a: ModelArch, context: number): number {
  const window = a.slidingWindow ?? context
  const pairs = a.layers.full * context + a.layers.sliding * Math.min(context, window)
  const dims = a.mla ? 2 * a.mla.kvLoraRank + a.mla.qkRopeHeadDim : 2 * a.headDim
  return 2 * a.numHeads * dims * pairs
}

/** Bytes of weights streamed from HBM in one decode step of `batch` sequences (whole replica). */
function weightBytesPerStep(a: ModelArch, s: Shape, batch: number): number {
  const all = weightBytes(a.params.total, a.params.embedding, s.weightFormat)
  if (!a.moe) return all
  const m = a.moe
  const perExpert = (a.gatedMlp ? 3 : 2) * a.hiddenSize * m.expertIntermediate * WEIGHT_FORMATS[s.weightFormat].bytes
  const expertBytes = m.moeLayers * m.numExperts * perExpert
  const touched = 1 - Math.pow(1 - m.expertsPerToken / m.numExperts, batch)
  return all - expertBytes + expertBytes * touched
}

/** Seconds for one decode step with `batch` running sequences at average `context` tokens. */
export function decodeStepTime(a: ModelArch, s: Shape, batch: number, context: number, e: EngineAssumptions = DEFAULT_ENGINE): number {
  const kvB = KV_FORMATS[s.kvFormat].bytes
  const bw = s.gpu.memoryBandwidthGBs * GB * e.bandwidthEfficiency
  const bytes = weightBytesPerStep(a, s, batch) / s.tp + (batch * sequenceCacheBytes(a, context, kvB)) / kvShard(a, s.tp)
  const flops = (batch * (2 * a.params.active + attentionFlopsPerToken(a, context))) / s.tp
  const compute = flops / (peakFlops(s.gpu, computePrecision(s.weightFormat, s.gpu)) * e.decodeMfu)
  return Math.max(bytes / bw, compute) + allReduceTime(s, batch, a) + e.stepOverheadMs / 1000
}

/** Seconds to prefill `newTokens` on top of `cachedTokens` already in the KV cache. */
export function prefillTime(a: ModelArch, s: Shape, newTokens: number, cachedTokens: number, e: EngineAssumptions = DEFAULT_ENGINE): number {
  if (newTokens <= 0) return e.stepOverheadMs / 1000
  const midContext = cachedTokens + newTokens / 2
  const flops = (newTokens * (2 * a.params.active + attentionFlopsPerToken(a, midContext))) / s.tp
  const computeT = flops / (peakFlops(s.gpu, computePrecision(s.weightFormat, s.gpu)) * e.prefillMfu)
  const chunks = Math.ceil(newTokens / e.maxBatchedTokens)
  const bw = s.gpu.memoryBandwidthGBs * GB * e.bandwidthEfficiency
  const weightT = (chunks * weightBytes(a.params.total, a.params.embedding, s.weightFormat)) / s.tp / bw
  return Math.max(computeT, weightT) + allReduceTime(s, newTokens, a) + (chunks * e.stepOverheadMs) / 1000
}

export interface OperatingPoint {
  /** LLM calls per second this replica sustains. */
  rps: number
  concurrency: number
  ttftMs: number
  tpotMs: number
  utilization: number
  outputTokPerSec: number
}

export type Limiter = 'memory' | 'ttft' | 'tpot' | 'compute' | 'none'

export interface ReplicaCapacity {
  memory: ReplicaMemory
  /** Latency for a lone request on an idle replica. */
  idle: { ttftMs: number; tpotMs: number; tokPerSec: number }
  /** Highest load that still meets the SLO (null if even an idle replica misses it). */
  best: OperatingPoint | null
  limiter: Limiter
  /** Absolute max output tokens/s if latency didn't matter. */
  saturationTokPerSec: number
}

interface Ctx {
  a: ModelArch
  s: Shape
  w: Workload
  e: EngineAssumptions
  prefill: number
  /** Engine steps a request's prefill is split into (chunked prefill). */
  prefillChunks: number
  avgContext: number
  maxSeqs: number
}

/**
 * Steady state at arrival rate `r`. Decode runs continuously while any sequence is
 * active, so it is not a queue; prefills are. Each prefill takes its compute time
 * plus one decode step per chunk it rides along with, and new arrivals queue
 * behind in-flight prefills (M/D/1 waiting time). Decode steps are stretched by
 * the share of GPU time spent on prefill.
 */
function solveAtRate(c: Ctx, r: number): OperatingPoint | null {
  const prefillShare = r * c.prefill
  if (prefillShare >= 0.95) return null
  const point = (batch: number) => {
    const step = decodeStepTime(c.a, c.s, Math.max(1, batch), c.avgContext, c.e)
    const service = c.prefill + c.prefillChunks * step
    const u = r * service
    const q = Math.min(u, 0.98)
    const ttft = service * (1 + q / (2 * (1 - q)))
    const tpot = step / (1 - prefillShare)
    return { u, ttft, tpot, next: r * (ttft + c.w.outputTokens * tpot) }
  }
  let batch = Math.max(1, r * (c.prefill + c.w.outputTokens * decodeStepTime(c.a, c.s, 1, c.avgContext, c.e)))
  let p = point(batch)
  for (let i = 0; i < 100 && Math.abs(p.next - batch) > 0.01 * Math.max(1, batch); i++) {
    batch = Math.min(0.5 * batch + 0.5 * p.next, 1e6)
    p = point(batch)
  }
  if (p.u >= 0.98 || batch >= 1e6) return null
  return {
    rps: r,
    concurrency: batch,
    ttftMs: p.ttft * 1000,
    tpotMs: p.tpot * 1000,
    utilization: Math.max(p.u, prefillShare),
    outputTokPerSec: r * c.w.outputTokens,
  }
}

function meets(c: Ctx, p: OperatingPoint | null): Limiter | 'ok' {
  if (!p) return 'compute'
  if (p.concurrency > c.maxSeqs) return 'memory'
  if (p.ttftMs > c.w.slo.ttftMs) return 'ttft'
  if (p.tpotMs > c.w.slo.tpotMs) return 'tpot'
  return 'ok'
}

export function replicaCapacity(a: ModelArch, s: Shape, w: Workload, e: EngineAssumptions = DEFAULT_ENGINE): ReplicaCapacity {
  const memory = replicaMemory(a, s, w, e)
  const cached = w.prefixCaching ? Math.min(w.sharedPrefixTokens, w.inputTokens) : 0
  const prefill = prefillTime(a, s, w.inputTokens - cached, cached, e)
  const coldPrefill = prefillTime(a, s, w.inputTokens, 0, e)
  const avgContext = w.inputTokens + w.outputTokens / 2
  const maxSeqs = Math.min(memory.maxSeqsByMemory, e.maxNumSeqs)
  const idleStep = decodeStepTime(a, s, 1, avgContext, e)
  const idle = { ttftMs: (w.prefixCaching ? prefill : coldPrefill) * 1000, tpotMs: idleStep * 1000, tokPerSec: 1 / idleStep }

  const satBatch = Math.max(1, maxSeqs)
  const satStep = decodeStepTime(a, s, satBatch, avgContext, e)
  const saturationRps = 1 / (prefill + (w.outputTokens * satStep) / satBatch)
  const saturationTokPerSec = memory.fits ? saturationRps * w.outputTokens : 0

  if (!memory.fits) return { memory, idle, best: null, limiter: 'memory', saturationTokPerSec }
  const prefillChunks = Math.max(1, Math.ceil((w.inputTokens - cached) / e.maxBatchedTokens))
  const c: Ctx = { a, s, w, e, prefill, prefillChunks, avgContext, maxSeqs }
  const tiny = solveAtRate(c, saturationRps * 1e-4)
  const first = meets(c, tiny)
  if (first !== 'ok') return { memory, idle, best: null, limiter: first, saturationTokPerSec }

  let lo = saturationRps * 1e-4
  let hi = saturationRps
  let best = tiny!
  let limiter: Limiter = 'compute'
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2
    const p = solveAtRate(c, mid)
    const m = meets(c, p)
    if (m === 'ok') {
      lo = mid
      best = p!
    } else {
      hi = mid
      limiter = m
    }
  }
  return { memory, idle, best, limiter, saturationTokPerSec }
}

/** For closed-loop "N concurrent streams": how many streams one replica holds within SLO. */
export function streamsPerReplica(cap: ReplicaCapacity): number {
  return cap.best ? Math.floor(cap.best.concurrency) : 0
}

/** LLM calls per second implied by the load spec. */
export function callRates(load: LoadSpec): { peak: number; avg: number } | null {
  if (load.mode !== 'rate') return null
  return { peak: load.peakRps * load.callsPerRequest, avg: load.avgRps * load.callsPerRequest }
}

export interface ReplicaPlan {
  peakReplicas: number
  avgReplicas: number
}

export function replicasNeeded(cap: ReplicaCapacity, load: LoadSpec, e: EngineAssumptions = DEFAULT_ENGINE, minReplicas = 1): ReplicaPlan | null {
  if (!cap.best) return null
  if (load.mode === 'concurrent') {
    const per = Math.max(1, Math.floor(streamsPerReplica(cap) * e.targetUtilization))
    const n = Math.max(minReplicas, Math.ceil(load.concurrentStreams / per))
    return { peakReplicas: n, avgReplicas: n }
  }
  const rates = callRates(load)!
  const per = cap.best.rps * e.targetUtilization
  return {
    peakReplicas: Math.max(minReplicas, Math.ceil(rates.peak / per)),
    avgReplicas: Math.max(minReplicas, Math.ceil(rates.avg / per)),
  }
}
