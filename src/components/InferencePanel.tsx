import { Activity, CheckCircle2, Gauge, Layers, Server, TrendingUp, TriangleAlert } from 'lucide-react'
import { useDeferredValue, useMemo, useState } from 'react'
import { MarketControls } from '@/components/MarketControls'
import { MemoryBar } from '@/components/MemoryBar'
import { ChoiceCards, Hint, NumberField, SectionTitle, SelectField, SwitchField } from '@/components/fields'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { ModelArch } from '@/core/arch'
import { replicaCapacity, type EngineAssumptions, type Limiter, type ReplicaCapacity, type Workload } from '@/core/inference'
import { formatFromCheckpoint, KV_FORMATS, WEIGHT_FORMATS, type KvFormat, type WeightFormat } from '@/core/precision'
import { cheapestPerGpu, rankInference, tpOptions, type InferenceOption } from '@/core/recommend'
import { GPU_BY_ID, GPUS, marketOffers, PROVIDER_LABEL, regionName } from '@/data/catalog'
import { fmtCompact, fmtInt, fmtMs, fmtUsd } from '@/lib/format'
import { INFERENCE_PRESETS, matchingPreset, QUALITY_LEVELS, tokensToWords } from '@/presets'
import type { AppState, InferenceInputs, MarketInputs } from '@/state'

const LIMITER_TEXT: Record<Limiter, string> = {
  memory: 'KV-cache memory: no room for more concurrent sequences.',
  ttft: 'Time-to-first-token SLO: prefill queueing grows past the target.',
  tpot: 'Per-user token speed SLO: bigger batches make each step slower.',
  compute: 'Saturated: the GPU is busy 100% of the time.',
  none: '—',
}

export function buildWorkload(i: InferenceInputs): Workload {
  return {
    inputTokens: i.inputTokens,
    outputTokens: i.outputTokens,
    sharedPrefixTokens: Math.min(i.sharedPrefixTokens, i.inputTokens),
    prefixCaching: i.prefixCaching,
    load:
      i.loadMode === 'concurrent'
        ? { mode: 'concurrent', concurrentStreams: i.concurrentStreams }
        : { mode: 'rate', peakRps: i.peakRps, avgRps: Math.min(i.avgRps, i.peakRps), callsPerRequest: i.callsPerRequest, peakHoursPerDay: i.peakHoursPerDay },
    slo: { ttftMs: i.ttftMs, tpotMs: 1000 / Math.max(0.1, i.minTokPerSec) },
  }
}

interface Props {
  arch: ModelArch
  state: AppState
  update: <K extends keyof AppState>(key: K, patch: Partial<AppState[K]>) => void
}

export function InferencePanel({ arch, state, update }: Props) {
  const inf = state.inf
  const setInf = (p: Partial<InferenceInputs>) => update('inf', p)
  const weightFormat: WeightFormat = inf.weightFormat === 'auto' ? formatFromCheckpoint(arch.quantization) : inf.weightFormat

  const deferred = useDeferredValue({ inf, market: state.market, engine: state.engine, weightFormat })
  const workload = useMemo(() => buildWorkload(deferred.inf), [deferred.inf])
  const ranked = useMemo(
    () =>
      rankInference({
        arch,
        workload,
        weightFormat: deferred.weightFormat,
        kvFormat: deferred.inf.kvFormat,
        priceKind: deferred.market.priceKind,
        offers: marketOffers(deferred.market),
        gpus: GPU_BY_ID,
        engine: deferred.engine,
        minReplicas: deferred.inf.minReplicas,
      }),
    [arch, workload, deferred],
  )
  const [showAll, setShowAll] = useState(false)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const list = showAll ? ranked.slice(0, 60) : cheapestPerGpu(ranked).slice(0, 10)
  const keyOf = (o: InferenceOption) => `${o.offer.provider}:${o.offer.instanceType}:${o.offer.region}:${o.tp}`
  const selected = ranked.find((o) => keyOf(o) === selectedKey) ?? ranked[0]
  const stale = deferred.inf !== inf || deferred.market !== state.market

  const advanced = state.ui.advanced
  const qualityOptions = [
    ...(arch.quantization ? [{ id: 'auto', label: 'As published', detail: `${WEIGHT_FORMATS[formatFromCheckpoint(arch.quantization)].label} — the checkpoint’s own format` }] : []),
    ...QUALITY_LEVELS.map((q) => ({ id: q.id, label: q.label, detail: q.detail })),
  ]
  const qualityId =
    inf.weightFormat === 'auto' && arch.quantization
      ? 'auto'
      : (QUALITY_LEVELS.find((q) => q.weights === weightFormat && q.kv === inf.kvFormat)?.id ?? null)
  const pickQuality = (id: string) => {
    if (id === 'auto') return setInf({ weightFormat: 'auto', kvFormat: 'bf16' })
    const q = QUALITY_LEVELS.find((x) => x.id === id)!
    setInf({ weightFormat: q.weights, kvFormat: q.kv })
  }
  const rateMode = workload.load.mode === 'rate'
  const contextTooLong = inf.inputTokens + inf.outputTokens > arch.maxContext

  return (
    <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      <aside className="flex flex-col gap-6 lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto lg:pr-1 [&>*]:shrink-0">
        <Card className="gap-4">
          <CardContent className="flex flex-col gap-3">
            <SectionTitle>What are you building?</SectionTitle>
            <ChoiceCards
              options={INFERENCE_PRESETS.map((p) => ({ id: p.id, label: p.label, detail: p.blurb, icon: p.icon }))}
              value={matchingPreset(inf)}
              onChange={(id) => setInf(INFERENCE_PRESETS.find((p) => p.id === id)!.values)}
            />
            <p className="text-[11px] text-muted-foreground">Presets fill in typical numbers — tweak anything below.</p>
          </CardContent>
        </Card>

        <Card className="gap-4">
          <CardContent className="flex flex-col gap-5">
            <div className="flex flex-col gap-3">
              <SectionTitle>Quality vs. memory</SectionTitle>
              {advanced ? (
                <div className="grid grid-cols-2 gap-3">
                  <SelectField<WeightFormat | 'auto'>
                    label="Weights"
                    value={inf.weightFormat}
                    onChange={(weightFormat) => setInf({ weightFormat })}
                    options={[
                      { value: 'auto', label: `Checkpoint (${WEIGHT_FORMATS[formatFromCheckpoint(arch.quantization)].label})` },
                      ...(Object.keys(WEIGHT_FORMATS) as WeightFormat[]).map((k) => ({ value: k, label: WEIGHT_FORMATS[k].label })),
                    ]}
                    hint={WEIGHT_FORMATS[weightFormat].note}
                  />
                  <SelectField<KvFormat>
                    label="KV cache"
                    value={inf.kvFormat}
                    onChange={(kvFormat) => setInf({ kvFormat })}
                    options={(Object.keys(KV_FORMATS) as KvFormat[]).map((k) => ({ value: k, label: KV_FORMATS[k].label }))}
                    hint="FP8 KV cache (vLLM --kv-cache-dtype fp8) halves KV memory, usually with negligible quality loss."
                  />
                </div>
              ) : (
                <ChoiceCards options={qualityOptions} value={qualityId} onChange={pickQuality} />
              )}
            </div>

            <div className="flex flex-col gap-3">
              <SectionTitle>Size of each request</SectionTitle>
              <div className="grid grid-cols-2 gap-3">
                <NumberField
                  label="Prompt tokens"
                  value={inf.inputTokens}
                  onChange={(inputTokens) => setInf({ inputTokens })}
                  min={1}
                  help={tokensToWords(inf.inputTokens)}
                  hint="Everything the model reads per call: system prompt, tool definitions, retrieved text, chat history and the user message."
                />
                <NumberField
                  label="Reply tokens"
                  value={inf.outputTokens}
                  onChange={(outputTokens) => setInf({ outputTokens })}
                  min={1}
                  help={tokensToWords(inf.outputTokens)}
                  hint="What the model writes per call. A tool call is often 20–80 tokens; a chat answer 200–500."
                />
              </div>
              {advanced && (
                <>
                  <NumberField
                    label="Shared prefix tokens"
                    value={inf.sharedPrefixTokens}
                    onChange={(sharedPrefixTokens) => setInf({ sharedPrefixTokens })}
                    min={0}
                    hint="Part of the prompt identical across requests (system prompt, tool definitions, few-shot examples). With prefix caching it is computed and stored once."
                  />
                  <SwitchField
                    label="Prefix caching"
                    checked={inf.prefixCaching}
                    onChange={(prefixCaching) => setInf({ prefixCaching })}
                    hint="vLLM automatic prefix caching / SGLang RadixAttention. On by default in both."
                  />
                </>
              )}
              {contextTooLong && (
                <p className="text-xs text-destructive">Prompt + reply is longer than the model’s max context ({fmtInt(arch.maxContext)} tokens).</p>
              )}
            </div>

            <div className="flex flex-col gap-3">
              <SectionTitle>Traffic</SectionTitle>
              <Tabs value={inf.loadMode} onValueChange={(v) => setInf({ loadMode: v as InferenceInputs['loadMode'] })}>
                <TabsList className="w-full">
                  <TabsTrigger value="concurrent">At the same time</TabsTrigger>
                  <TabsTrigger value="rate">Per second</TabsTrigger>
                </TabsList>
              </Tabs>
              {inf.loadMode === 'concurrent' ? (
                <NumberField
                  label="Requests being answered at once"
                  value={inf.concurrentStreams}
                  onChange={(concurrentStreams) => setInf({ concurrentStreams })}
                  min={1}
                  hint="Generations running at the same moment. Not the number of users online — people reading or typing don’t count. Rule of thumb: ~5–10% of active users."
                />
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField label="Busiest requests / s" value={inf.peakRps} onChange={(peakRps) => setInf({ peakRps })} min={0.001} hint="Use the busiest sustained minute, not the single busiest second." />
                    <NumberField label="Typical requests / s" value={inf.avgRps} onChange={(avgRps) => setInf({ avgRps })} min={0} hint="Daily average. Used for the autoscaled cost." />
                  </div>
                  <NumberField
                    label="Model calls per request"
                    value={inf.callsPerRequest}
                    onChange={(callsPerRequest) => setInf({ callsPerRequest })}
                    min={0.01}
                    hint="Agents often call the model several times per user message (route, call tools, answer, check)."
                  />
                  {advanced && (
                    <NumberField
                      label="Busy hours per day"
                      value={inf.peakHoursPerDay}
                      onChange={(peakHoursPerDay) => setInf({ peakHoursPerDay })}
                      min={0}
                      max={24}
                      hint="Used to estimate autoscaled cost: peak capacity for these hours, average capacity for the rest."
                    />
                  )}
                </>
              )}
            </div>

            <div className="flex flex-col gap-3">
              <SectionTitle>How fast should it feel?</SectionTitle>
              <div className="grid grid-cols-2 gap-3">
                <NumberField label="First word within" value={inf.ttftMs} onChange={(ttftMs) => setInf({ ttftMs })} min={10} suffix="ms" hint="Time to first token: how long until the reply starts appearing." />
                <NumberField
                  label="Typing speed"
                  value={inf.minTokPerSec}
                  onChange={(minTokPerSec) => setInf({ minTokPerSec })}
                  min={0.1}
                  suffix="tok/s"
                  help={inf.minTokPerSec >= 30 ? 'feels instant' : inf.minTokPerSec >= 10 ? 'faster than reading' : 'readable'}
                  hint="Minimum tokens per second each user sees. People read ~5–10 tok/s; 20+ feels instant."
                />
              </div>
              {advanced && (
                <NumberField
                  label="Minimum replicas"
                  value={inf.minReplicas}
                  onChange={(minReplicas) => setInf({ minReplicas })}
                  min={1}
                  hint="Set to 2+ for high availability (rolling updates, node failures)."
                />
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <MarketControls value={state.market} onChange={(p: Partial<MarketInputs>) => update('market', p)} />
          </CardContent>
        </Card>

        {advanced && <EngineAssumptionsCard engine={state.engine} onChange={(p) => update('engine', p)} />}
      </aside>

      <section className={`flex min-w-0 flex-col gap-6 transition-opacity ${stale ? 'opacity-70' : ''}`}>
        {selected ? (
          <>
            <Verdict option={selected} workload={workload} />
            <Headline option={selected} workload={workload} />
            <ReplicaDetails option={selected} />
          </>
        ) : (
          <NoFit
            canQuantize={weightFormat === 'bf16'}
            onQuantize={() => setInf({ weightFormat: 'fp8', kvFormat: 'fp8' })}
            onRelax={() => setInf({ ttftMs: Math.round(inf.ttftMs * 2), minTokPerSec: Math.max(1, Math.round(inf.minTokPerSec / 2)) })}
            onAnyRegion={state.market.region !== 'any' ? () => update('market', { region: 'any' }) : undefined}
          />
        )}

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="size-4" /> Cheapest deployments
            </CardTitle>
            <CardDescription>
              Every instance type × tensor-parallel layout that meets your latency targets, sized for peak{rateMode ? ' and ranked by autoscaled monthly cost (730 h/month)' : ' and ranked by monthly cost (730 h/month)'}. Click a row for details.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Instance</TableHead>
                    <TableHead>Layout</TableHead>
                    <TableHead className="text-right">Per replica</TableHead>
                    <TableHead className="text-right">Instances</TableHead>
                    <TableHead className="text-right">$/hour</TableHead>
                    {rateMode ? (
                      <>
                        <TableHead className="text-right">Fixed at peak</TableHead>
                        <TableHead className="text-right">Autoscaled</TableHead>
                      </>
                    ) : (
                      <TableHead className="text-right">Monthly</TableHead>
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {list.map((o) => {
                    const k = keyOf(o)
                    const active = selected && keyOf(selected) === k
                    return (
                      <TableRow key={k} data-state={active ? 'selected' : undefined} className="cursor-pointer" onClick={() => setSelectedKey(k)}>
                        <TableCell>
                          <div className="flex flex-col">
                            <span className="font-medium">
                              {o.offer.gpuCount}× {o.gpu.name.replace(/^NVIDIA |^AMD /, '')}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {PROVIDER_LABEL[o.offer.provider]} · <span className="font-mono">{o.offer.instanceType}</span> · {o.offer.region === 'on-prem' ? 'on-prem' : regionName(o.offer.provider, o.offer.region)}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="whitespace-nowrap text-xs">
                          TP={o.tp} · {o.replicasPerInstance} replica{o.replicasPerInstance > 1 ? 's' : ''}/box
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs tabular-nums">{perReplicaLabel(o.capacity, workload)}</TableCell>
                        <TableCell className="text-right font-mono tabular-nums">
                          {o.avgInstances === o.peakInstances ? o.peakInstances : `${o.avgInstances}–${o.peakInstances}`}
                        </TableCell>
                        <TableCell className="text-right font-mono tabular-nums">{fmtUsd(o.hourly)}</TableCell>
                        {rateMode && <TableCell className="text-right font-mono tabular-nums">{fmtUsd(o.monthlyFixed, true)}</TableCell>}
                        <TableCell className="text-right font-mono font-semibold tabular-nums">{fmtUsd(o.monthlyAutoscaled, true)}</TableCell>
                      </TableRow>
                    )
                  })}
                  {list.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={rateMode ? 7 : 6} className="py-8 text-center text-muted-foreground">
                        Nothing fits — see the per-GPU table for why.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
            {ranked.length > list.length || showAll ? (
              <Button variant="ghost" size="sm" className="self-start" onClick={() => setShowAll((v) => !v)}>
                {showAll ? 'Show cheapest per GPU type' : `Show all ${ranked.length} options`}
              </Button>
            ) : null}
          </CardContent>
        </Card>

        <GpuFitTable arch={arch} workload={workload} weightFormat={weightFormat} kvFormat={inf.kvFormat} engine={state.engine} />
      </section>
    </div>
  )
}

function perReplicaLabel(cap: ReplicaCapacity, w: Workload): string {
  if (!cap.best) return '—'
  return w.load.mode === 'concurrent' ? `${fmtInt(cap.best.concurrency)} streams` : `${cap.best.rps.toFixed(cap.best.rps < 10 ? 1 : 0)} calls/s`
}

const toneStyle = (tone: string) => ({
  borderColor: `color-mix(in oklab, ${tone} 40%, transparent)`,
  background: `color-mix(in oklab, ${tone} 8%, var(--card))`,
})

function Verdict({ option: o, workload }: { option: InferenceOption; workload: Workload }) {
  const b = o.capacity.best!
  const gpu = o.gpu.name.replace(/^NVIDIA |^AMD /, '')
  const where = o.offer.provider === 'custom' ? 'your own hardware' : `${PROVIDER_LABEL[o.offer.provider]} ${o.offer.region === 'on-prem' ? '' : regionName(o.offer.provider, o.offer.region)}`
  const count = o.avgInstances === o.peakInstances ? `${o.peakInstances}×` : `${o.avgInstances}–${o.peakInstances}×`
  return (
    <div className="flex items-start gap-3 rounded-xl border p-4" style={toneStyle('var(--status-good)')}>
      <CheckCircle2 className="mt-0.5 size-5 shrink-0" style={{ color: 'var(--status-good)' }} aria-label="Fits" />
      <div className="flex flex-col gap-1 text-sm leading-relaxed">
        <p>
          You need <b>{count} {o.offer.instanceType}</b> ({o.offer.gpuCount}× {gpu}, {where})
          {workload.load.mode === 'rate' ? ', scaling with traffic' : ''} — about{' '}
          <b>{fmtUsd(o.monthlyAutoscaled, true)}/month</b>.
        </p>
        <p className="text-muted-foreground">
          Replies start in ~{fmtMs(b.ttftMs)} and stream at {(1000 / b.tpotMs).toFixed(0)} tokens/s, even at your busiest.
          {o.tp > 1 && ` The model is split across ${o.tp} GPUs.`} Cheaper or faster alternatives are listed below.
        </p>
      </div>
    </div>
  )
}

function NoFit({ canQuantize, onQuantize, onRelax, onAnyRegion }: { canQuantize: boolean; onQuantize: () => void; onRelax: () => void; onAnyRegion?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border p-4" style={toneStyle('var(--status-warning)')}>
      <TriangleAlert className="mt-0.5 size-5 shrink-0" style={{ color: 'var(--status-warning)' }} aria-label="Warning" />
      <div className="flex flex-col gap-3 text-sm">
        <div>
          <p className="font-medium">Nothing meets these targets with the selected clouds.</p>
          <p className="text-muted-foreground">The model may not fit in memory, or even an idle GPU is slower than your speed targets. Try one of these:</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {canQuantize && (
            <Button size="sm" variant="outline" onClick={onQuantize}>
              Use Balanced quality (FP8)
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={onRelax}>
            Relax speed targets 2×
          </Button>
          {onAnyRegion && (
            <Button size="sm" variant="outline" onClick={onAnyRegion}>
              Search all regions
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

function Tile({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border bg-card p-4">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {icon} {label}
      </span>
      <span className="text-2xl font-semibold tracking-tight tabular-nums">{value}</span>
      <span className="text-xs text-muted-foreground">{sub}</span>
    </div>
  )
}

function Headline({ option: o, workload }: { option: InferenceOption; workload: Workload }) {
  const b = o.capacity.best!
  const replicasPeak = o.peakInstances * o.replicasPerInstance
  const replicasAvg = o.avgInstances * o.replicasPerInstance
  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <Tile
        icon={<Server className="size-3.5" />}
        label="Recommended"
        value={`${o.peakInstances}× ${o.offer.instanceType}`}
        sub={`${o.offer.gpuCount}× ${o.gpu.name.replace(/^NVIDIA /, '')} · ${PROVIDER_LABEL[o.offer.provider]}`}
      />
      <Tile
        icon={<Layers className="size-3.5" />}
        label="Replicas"
        value={replicasAvg === replicasPeak ? String(replicasPeak) : `${replicasAvg} → ${replicasPeak}`}
        sub={`TP=${o.tp} each · ${workload.load.mode === 'rate' ? 'average → peak' : 'for your concurrency'}`}
      />
      <Tile icon={<Gauge className="size-3.5" />} label="Latency at load" value={fmtMs(b.ttftMs)} sub={`first token · ${(1000 / b.tpotMs).toFixed(0)} tok/s per user`} />
      <Tile
        icon={<Activity className="size-3.5" />}
        label="Monthly cost"
        value={fmtUsd(o.monthlyAutoscaled, true)}
        sub={workload.load.mode === 'rate' ? `autoscaled · ${fmtUsd(o.monthlyFixed, true)} if fixed at peak` : `${fmtUsd(o.hourly)}/h per instance`}
      />
    </div>
  )
}

function ReplicaDetails({ option: o }: { option: InferenceOption }) {
  const c = o.capacity
  const m = c.memory
  const b = c.best!
  const free = Math.max(0, m.total - m.weights - m.activations - m.overhead - Math.max(0, m.kvBudget))
  return (
    <Card>
      <CardHeader>
        <CardTitle>One replica on {o.gpu.name}</CardTitle>
        <CardDescription>
          TP={o.tp} · memory per GPU and the operating point the SLO allows. Limited by: {LIMITER_TEXT[c.limiter]}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        <MemoryBar
          title="GPU memory"
          capacity={m.total}
          usable={m.usable}
          segments={[
            { key: 'w', label: 'Weights', bytes: m.weights },
            { key: 'kv', label: 'KV cache pool', bytes: Math.max(0, m.kvBudget) },
            { key: 'a', label: 'Activations', bytes: m.activations },
            { key: 'o', label: 'Runtime overhead', bytes: m.overhead },
            { key: 'f', label: 'Reserved headroom', bytes: free },
          ]}
        />
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
          <Metric label="KV capacity" value={`${fmtCompact(m.kvTokenCapacity)} tok`} />
          <Metric label="Max sequences (memory)" value={fmtInt(m.maxSeqsByMemory)} />
          <Metric label="Idle first token" value={fmtMs(c.idle.ttftMs)} />
          <Metric label="Idle speed" value={`${c.idle.tokPerSec.toFixed(0)} tok/s`} />
          <Metric label="Sustained calls/s" value={b.rps.toFixed(2)} />
          <Metric label="Concurrency at SLO" value={fmtInt(b.concurrency)} />
          <Metric label="Output tok/s (SLO)" value={fmtCompact(b.outputTokPerSec)} />
          <Metric label="Output tok/s (saturated)" value={fmtCompact(c.saturationTokPerSec)} />
        </dl>
        <ScalingPlan option={o} />
      </CardContent>
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-mono font-medium tabular-nums">{value}</dd>
    </div>
  )
}

function ScalingPlan({ option: o }: { option: InferenceOption }) {
  const b = o.capacity.best!
  const perReplicaConc = Math.max(1, Math.floor(b.concurrency))
  const loadSeconds = o.capacity.memory.weights * o.tp / 1e9 / 1.5 + 20
  return (
    <div className="flex flex-col gap-3 rounded-xl border bg-muted/40 p-4 text-sm">
      <SectionTitle>Scaling playbook</SectionTitle>
      <ul className="flex flex-col gap-2 leading-relaxed text-muted-foreground [&_b]:font-medium [&_b]:text-foreground">
        <li>
          Run <b>{o.avgInstances === o.peakInstances ? o.peakInstances : `${o.avgInstances}–${o.peakInstances}`}</b> × {o.offer.instanceType} with{' '}
          <b>{o.replicasPerInstance}</b> engine replica{o.replicasPerInstance > 1 ? 's' : ''} each (TP={o.tp}
          {o.tp > 1 ? `, needs ${o.gpu.interconnect.type === 'nvlink' ? 'NVLink' : 'PCIe — TP over PCIe is slow'}` : ', no inter-GPU traffic'}).
        </li>
        <li>
          Scale out when a replica holds more than <b>~{perReplicaConc} running requests</b>, or when <code>vllm:num_requests_waiting</code> stays above 0 /{' '}
          <code>vllm:kv_cache_usage_perc</code> above 0.9 for 30–60 s. Scale in slowly (5–10 min cool-down).
        </li>
        <li>
          New replicas need roughly <b>{Math.round(loadSeconds)} s</b> to load weights and capture CUDA graphs — keep one warm spare ahead of known peaks, and bake
          weights into the node image or a local cache.
        </li>
        <li>
          Route with a <b>prefix/KV-aware load balancer</b> (vLLM production-stack, llm-d, SGLang router, NVIDIA Dynamo) so requests sharing a system prompt land on
          replicas that already cached it.
        </li>
        {o.offer.provider !== 'custom' && (
          <li>
            Cost lever: keep the average fleet on 1-year commitment and burst the peak on on-demand or spot.
          </li>
        )}
      </ul>
    </div>
  )
}

function GpuFitTable({
  arch,
  workload,
  weightFormat,
  kvFormat,
  engine,
}: {
  arch: ModelArch
  workload: Workload
  weightFormat: WeightFormat
  kvFormat: KvFormat
  engine: EngineAssumptions
}) {
  const rows = useMemo(
    () =>
      GPUS.map((gpu) => {
        for (const tp of tpOptions(8, arch)) {
          const cap = replicaCapacity(arch, { gpu, tp, weightFormat, kvFormat }, workload, engine)
          if (cap.memory.fits) return { gpu, tp, cap }
        }
        return { gpu, tp: null as number | null, cap: null as ReplicaCapacity | null }
      }),
    [arch, workload, weightFormat, kvFormat, engine],
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle>Will it run on…</CardTitle>
        <CardDescription>Smallest tensor-parallel group of each GPU that holds the model plus at least one sequence of your request shape.</CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>GPU</TableHead>
              <TableHead className="text-right">GPUs</TableHead>
              <TableHead className="text-right">KV capacity</TableHead>
              <TableHead className="text-right">Idle speed</TableHead>
              <TableHead className="text-right">First token</TableHead>
              <TableHead className="text-right">
                <span className="inline-flex items-center gap-1">
                  Capacity at SLO <Hint>Per replica. “—” means even an idle replica misses your latency targets.</Hint>
                </span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map(({ gpu, tp, cap }) => (
              <TableRow key={gpu.id}>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{gpu.name.replace(/^NVIDIA /, '')}</span>
                    <span className="text-xs text-muted-foreground">{gpu.memoryGB} GB</span>
                    {gpu.segment !== 'datacenter' && (
                      <Badge variant="outline" className="text-[10px]">
                        {gpu.segment}
                      </Badge>
                    )}
                  </div>
                </TableCell>
                {cap && tp ? (
                  <>
                    <TableCell className="text-right font-mono tabular-nums">{tp}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{fmtCompact(cap.memory.kvTokenCapacity)} tok</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{cap.idle.tokPerSec.toFixed(0)} tok/s</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{fmtMs(cap.idle.ttftMs)}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{perReplicaLabel(cap, workload)}</TableCell>
                  </>
                ) : (
                  <TableCell colSpan={5} className="text-right text-xs text-muted-foreground">
                    Doesn’t fit on 8 GPUs at this precision
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

function EngineAssumptionsCard({ engine, onChange }: { engine: EngineAssumptions; onChange: (p: Partial<EngineAssumptions>) => void }) {
  return (
    <Card className="py-0">
      <Accordion type="single" collapsible>
        <AccordionItem value="a" className="border-none">
          <AccordionTrigger className="px-6 py-4 text-sm">Engine assumptions</AccordionTrigger>
          <AccordionContent className="grid grid-cols-2 gap-3 px-6 pb-6">
            <NumberField label="GPU memory utilization" value={engine.gpuMemoryUtilization} onChange={(gpuMemoryUtilization) => onChange({ gpuMemoryUtilization })} min={0.3} max={0.99} step={0.01} />
            <NumberField label="Bandwidth efficiency" value={engine.bandwidthEfficiency} onChange={(bandwidthEfficiency) => onChange({ bandwidthEfficiency })} min={0.1} max={1} step={0.05} />
            <NumberField label="Prefill MFU" value={engine.prefillMfu} onChange={(prefillMfu) => onChange({ prefillMfu })} min={0.05} max={1} step={0.05} />
            <NumberField label="Decode MFU" value={engine.decodeMfu} onChange={(decodeMfu) => onChange({ decodeMfu })} min={0.05} max={1} step={0.05} />
            <NumberField label="Step overhead" value={engine.stepOverheadMs} onChange={(stepOverheadMs) => onChange({ stepOverheadMs })} min={0} suffix="ms" />
            <NumberField label="Runtime overhead" value={engine.runtimeOverheadGB} onChange={(runtimeOverheadGB) => onChange({ runtimeOverheadGB })} min={0} suffix="GB" />
            <NumberField label="Max batched tokens" value={engine.maxBatchedTokens} onChange={(maxBatchedTokens) => onChange({ maxBatchedTokens })} min={256} />
            <NumberField label="Max sequences" value={engine.maxNumSeqs} onChange={(maxNumSeqs) => onChange({ maxNumSeqs })} min={1} />
            <NumberField
              label="Target utilization"
              value={engine.targetUtilization}
              onChange={(targetUtilization) => onChange({ targetUtilization })}
              min={0.1}
              max={1}
              step={0.05}
              hint="Headroom kept on every replica so bursts don’t blow the SLO before the autoscaler reacts."
              className="col-span-2"
            />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Card>
  )
}
