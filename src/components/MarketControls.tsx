import { X } from 'lucide-react'
import { NumberField, SectionTitle, SelectField, SwitchField } from '@/components/fields'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { PriceKind } from '@/core/hardware'
import { GPU_BY_ID, GPUS, PRICING, PROVIDER_LABEL } from '@/data/catalog'
import type { MarketInputs } from '@/state'

export function MarketControls({ value, onChange }: { value: MarketInputs; onChange: (p: Partial<MarketInputs>) => void }) {
  const regions = PRICING.regions.filter((r) => value.providers.includes(r.provider as 'aws' | 'gcp'))
  return (
    <div className="flex flex-col gap-3">
      <SectionTitle>Where it runs</SectionTitle>
      <ToggleGroup
        type="multiple"
        variant="outline"
        value={value.providers}
        onValueChange={(v) => v.length && onChange({ providers: v as MarketInputs['providers'], region: 'any' })}
        className="w-full"
      >
        <ToggleGroupItem value="aws" className="flex-1">AWS</ToggleGroupItem>
        <ToggleGroupItem value="gcp" className="flex-1">Google Cloud</ToggleGroupItem>
      </ToggleGroup>
      <div className="grid grid-cols-2 gap-3">
        <SelectField
          label="Region"
          value={value.region}
          onChange={(region) => onChange({ region })}
          options={[
            { value: 'any', label: 'Any region' },
            ...regions.map((r) => ({ value: r.id, label: r.name, detail: `${PROVIDER_LABEL[r.provider]} · ${r.id}` })),
          ]}
        />
        <SelectField<PriceKind>
          label="Pricing"
          value={value.priceKind}
          onChange={(priceKind) => onChange({ priceKind })}
          options={[
            { value: 'onDemand', label: 'On-demand' },
            { value: 'spot', label: 'Spot / preemptible' },
            { value: 'commit1y', label: '1-year commitment' },
          ]}
          hint="Spot can be reclaimed at any time — fine for batch jobs and extra replicas above your baseline, risky as the only capacity."
        />
      </div>
      <SwitchField
        label="Include my own GPUs"
        checked={value.includeOwned}
        onChange={(includeOwned) => onChange({ includeOwned })}
        hint="Compare on-prem or other-cloud GPUs at a flat, amortized $/GPU-hour (hardware + power + hosting)."
      />
      {value.includeOwned && (
        <div className="flex flex-col gap-3 rounded-lg border border-dashed p-3">
          <NumberField
            label="Cost per GPU-hour"
            value={value.ownedHourlyPerGpu}
            onChange={(ownedHourlyPerGpu) => onChange({ ownedHourlyPerGpu })}
            min={0}
            suffix="USD"
          />
          <div className="flex flex-wrap gap-1.5">
            {value.ownedGpuIds.map((id) => (
              <Badge key={id} variant="secondary" className="gap-1 pr-1">
                {GPU_BY_ID.get(id)?.name ?? id}
                <button
                  type="button"
                  aria-label="Remove"
                  className="rounded-sm p-0.5 hover:bg-background"
                  onClick={() => onChange({ ownedGpuIds: value.ownedGpuIds.filter((g) => g !== id) })}
                >
                  <X className="size-3" />
                </button>
              </Badge>
            ))}
          </div>
          <Select value="" onValueChange={(id) => onChange({ ownedGpuIds: [...new Set([...value.ownedGpuIds, id])] })}>
            <SelectTrigger className="h-9 w-full">
              <SelectValue placeholder="Add a GPU…" />
            </SelectTrigger>
            <SelectContent>
              {GPUS.filter((g) => !value.ownedGpuIds.includes(g.id)).map((g) => (
                <SelectItem key={g.id} value={g.id}>
                  {g.name}
                  <span className="ml-2 text-xs text-muted-foreground">{g.memoryGB} GB</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Prices: snapshot of {PRICING.updatedAt}, refreshed weekly from public AWS/GCP price lists. Excludes storage, egress and taxes.
      </p>
    </div>
  )
}
