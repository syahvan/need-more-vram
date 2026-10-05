export interface Gpu {
  id: string
  name: string
  vendor: 'nvidia' | 'amd' | 'intel' | 'google'
  arch: string
  segment: 'datacenter' | 'workstation' | 'consumer'
  memoryGB: number
  memoryBandwidthGBs: number
  /** Dense (non-sparse) peak throughput in TFLOPS / TOPS. */
  tflops: { fp32: number | null; bf16: number | null; fp8: number | null; int8: number | null; fp4: number | null }
  interconnect: { type: 'nvlink' | 'pcie' | 'infinity-fabric'; bandwidthGBs: number | null }
  tdpW: number | null
  fp8: boolean
  fp4: boolean
  sources: string[]
}

export type Provider = 'aws' | 'gcp' | 'custom'

export interface Offer {
  provider: Provider
  instanceType: string
  gpuId: string
  gpuCount: number
  vcpus: number | null
  ramGB: number | null
  region: string
  onDemandHourly: number | null
  spotHourly: number | null
  commit1yHourly: number | null
}

export interface Region {
  provider: Provider
  id: string
  name: string
}

export interface PricingSnapshot {
  updatedAt: string
  currency: string
  sources: { provider: string; url: string; note?: string }[]
  regions: Region[]
  offers: Offer[]
}

export type PriceKind = 'onDemand' | 'spot' | 'commit1y'

export function offerHourly(o: Offer, kind: PriceKind): number | null {
  if (kind === 'spot') return o.spotHourly
  if (kind === 'commit1y') return o.commit1yHourly
  return o.onDemandHourly
}

/** Peak dense matmul throughput (FLOP/s) for a given compute precision. */
export function peakFlops(gpu: Gpu, compute: 'bf16' | 'fp8' | 'fp4' | 'int8'): number {
  const t = gpu.tflops
  const bf16 = t.bf16 ?? (t.fp32 ?? 0) * 2
  const v = compute === 'fp4' ? (t.fp4 ?? t.fp8 ?? bf16) : compute === 'fp8' ? (t.fp8 ?? bf16) : compute === 'int8' ? (t.int8 ?? bf16) : bf16
  return v * 1e12
}

/** Effective per-direction GPU-to-GPU bandwidth in bytes/s used for TP all-reduce. */
export function linkBandwidth(gpu: Gpu): number {
  const fallback = gpu.interconnect.type === 'pcie' ? 32 : 300
  // NVLink figures are quoted bidirectional; PCIe figures per direction.
  const bw = gpu.interconnect.bandwidthGBs ?? fallback
  return (gpu.interconnect.type === 'pcie' ? bw : bw / 2) * 1e9
}
