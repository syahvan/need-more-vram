// Enumerates GPU / instance / parallelism combinations and ranks them by cost.

import type { ModelArch } from './arch'
import { offerHourly, type Gpu, type Offer, type PriceKind } from './hardware'
import { DEFAULT_ENGINE, replicaCapacity, replicasNeeded, type EngineAssumptions, type ReplicaCapacity, type Shape, type Workload } from './inference'
import type { KvFormat, WeightFormat } from './precision'
import { trainMemory, trainTime, type TrainConfig, type TrainMemory, type TrainTime } from './training'

export const HOURS_PER_MONTH = 730

export interface InferenceOption {
  offer: Offer
  gpu: Gpu
  tp: number
  replicasPerInstance: number
  capacity: ReplicaCapacity
  peakInstances: number
  avgInstances: number
  hourly: number
  /** Always provisioned for peak. */
  monthlyFixed: number
  /** Scaled down to the average outside peak hours. */
  monthlyAutoscaled: number
}

export interface InferenceSearch {
  arch: ModelArch
  workload: Workload
  weightFormat: WeightFormat
  kvFormat: KvFormat
  priceKind: PriceKind
  offers: Offer[]
  gpus: Map<string, Gpu>
  engine?: EngineAssumptions
  minReplicas?: number
}

export function tpOptions(gpuCount: number, a: ModelArch): number[] {
  const out: number[] = []
  for (let tp = 1; tp <= gpuCount; tp *= 2) {
    if (gpuCount % tp === 0 && a.numHeads % tp === 0) out.push(tp)
  }
  return out
}

/** Every (offer, TP) pair that meets the SLO, cheapest first. */
export function rankInference(q: InferenceSearch): InferenceOption[] {
  const engine = q.engine ?? DEFAULT_ENGINE
  const peakHours = q.workload.load.mode === 'rate' ? q.workload.load.peakHoursPerDay : 24
  const capCache = new Map<string, ReplicaCapacity>()
  const out: InferenceOption[] = []

  for (const offer of q.offers) {
    const gpu = q.gpus.get(offer.gpuId)
    const hourly = offerHourly(offer, q.priceKind)
    if (!gpu || hourly == null) continue
    for (const tp of tpOptions(offer.gpuCount, q.arch)) {
      const key = `${gpu.id}:${tp}`
      let cap = capCache.get(key)
      if (!cap) {
        const shape: Shape = { gpu, tp, weightFormat: q.weightFormat, kvFormat: q.kvFormat }
        cap = replicaCapacity(q.arch, shape, q.workload, engine)
        capCache.set(key, cap)
      }
      const plan = replicasNeeded(cap, q.workload.load, engine, q.minReplicas ?? 1)
      if (!plan) continue
      const perInstance = offer.gpuCount / tp
      const peakInstances = Math.ceil(plan.peakReplicas / perInstance)
      const avgInstances = Math.ceil(plan.avgReplicas / perInstance)
      const daily = peakInstances * peakHours + avgInstances * (24 - peakHours)
      out.push({
        offer,
        gpu,
        tp,
        replicasPerInstance: perInstance,
        capacity: cap,
        peakInstances,
        avgInstances,
        hourly,
        monthlyFixed: peakInstances * hourly * HOURS_PER_MONTH,
        monthlyAutoscaled: (daily / 24) * hourly * HOURS_PER_MONTH,
      })
    }
  }
  return out.sort((x, y) => x.monthlyAutoscaled - y.monthlyAutoscaled || x.monthlyFixed - y.monthlyFixed)
}

/** Keep the cheapest option per (provider, GPU type) for a readable shortlist. */
export function cheapestPerGpu<T extends { offer: Offer; gpu: Gpu }>(ranked: T[]): T[] {
  const seen = new Set<string>()
  return ranked.filter((o) => {
    const k = `${o.offer.provider}:${o.gpu.id}`
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}

export interface TrainingOption {
  offer: Offer
  gpu: Gpu
  nodes: number
  gpusTotal: number
  memory: TrainMemory
  time: TrainTime
  hourly: number
  cost: number
}

export function rankTraining(
  a: ModelArch,
  cfg: TrainConfig,
  offers: Offer[],
  gpus: Map<string, Gpu>,
  priceKind: PriceKind,
  maxNodes = 4,
): TrainingOption[] {
  const out: TrainingOption[] = []
  for (const offer of offers) {
    const gpu = gpus.get(offer.gpuId)
    const hourly = offerHourly(offer, priceKind)
    if (!gpu || hourly == null) continue
    for (let nodes = 1; nodes <= maxNodes; nodes *= 2) {
      if (nodes > 1 && offer.gpuCount < 8) break // multi-node only for full 8-GPU boxes
      const gpusTotal = offer.gpuCount * nodes
      const memory = trainMemory(a, cfg, gpusTotal)
      if (memory.total > gpu.memoryGB * 1e9 * 0.97) continue
      const time = trainTime(a, cfg, gpu, gpusTotal, nodes)
      out.push({ offer, gpu, nodes, gpusTotal, memory, time, hourly, cost: (time.seconds / 3600) * hourly * nodes })
      break // smallest node count that fits; more nodes only add cost for small models
    }
  }
  return out.sort((x, y) => x.cost - y.cost || x.time.seconds - y.time.seconds)
}
