import type { Gpu, Offer, PricingSnapshot } from '@/core/hardware'
import type { MarketInputs } from '@/state'
import gpuList from './gpus.json'
import pricingJson from './pricing.json'

export const GPUS = gpuList as Gpu[]
export const GPU_BY_ID = new Map(GPUS.map((g) => [g.id, g]))
export const PRICING = pricingJson as PricingSnapshot

export const PROVIDER_LABEL: Record<string, string> = { aws: 'AWS', gcp: 'Google Cloud', custom: 'Own / other' }

export function regionName(provider: string, id: string): string {
  return PRICING.regions.find((r) => r.provider === provider && r.id === id)?.name ?? id
}

/** Cloud offers filtered by the market controls, plus synthetic "owned GPU" offers. */
export function marketOffers(m: MarketInputs): Offer[] {
  const cloud = PRICING.offers.filter(
    (o) => m.providers.includes(o.provider as 'aws' | 'gcp') && (m.region === 'any' || o.region === m.region),
  )
  if (!m.includeOwned) return cloud
  const owned: Offer[] = []
  for (const gpuId of m.ownedGpuIds) {
    if (!GPU_BY_ID.has(gpuId)) continue
    for (const count of [1, 2, 4, 8]) {
      const hourly = m.ownedHourlyPerGpu * count
      owned.push({
        provider: 'custom',
        instanceType: `${count}× GPU server`,
        gpuId,
        gpuCount: count,
        vcpus: null,
        ramGB: null,
        region: 'on-prem',
        onDemandHourly: hourly,
        spotHourly: hourly,
        commit1yHourly: hourly,
      })
    }
  }
  return [...cloud, ...owned]
}
