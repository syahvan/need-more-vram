import { CheckCircle2, Clock, Cpu, DollarSign, Sigma, XCircle } from 'lucide-react'
import { useDeferredValue, useMemo } from 'react'
import { MarketControls } from '@/components/MarketControls'
import { MemoryBar } from '@/components/MemoryBar'
import { ChoiceCards, NumberField, SectionTitle, SelectField, SwitchField } from '@/components/fields'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { ModelArch } from '@/core/arch'
import { cheapestPerGpu, rankTraining } from '@/core/recommend'
import { trainMemory, trainTime, type Checkpointing, type LoraTargets, type Optimizer, type Sharding, type TrainConfig, type TrainMethod } from '@/core/training'
import { GPU_BY_ID, GPUS, marketOffers, PROVIDER_LABEL, regionName } from '@/data/catalog'
import { fmtBytes, fmtCompact, fmtDuration, fmtParams, fmtUsd } from '@/lib/format'
import { matchingTrainingPreset, tokensToWords, TRAINING_PRESETS } from '@/presets'
import type { AppState, MarketInputs, TrainingInputs } from '@/state'

function toConfig(t: TrainingInputs): TrainConfig {
  const { useRl, rl, localGpuId: _g, localGpuCount: _c, numExamples, avgTokensPerExample, ...rest } = t
  return { ...rest, datasetTokens: numExamples * avgTokensPerExample, rl: useRl ? rl : null }
}

interface Props {
  arch: ModelArch
  state: AppState
  update: <K extends keyof AppState>(key: K, patch: Partial<AppState[K]>) => void
}

export function TrainingPanel({ arch, state, update }: Props) {
  const t = state.train
  const set = (p: Partial<TrainingInputs>) => update('train', p)
  const setRl = (p: Partial<TrainingInputs['rl']>) => update('train', { rl: { ...t.rl, ...p } })

  const deferred = useDeferredValue({ t, market: state.market })
  const cfg = useMemo(() => toConfig(deferred.t), [deferred.t])
  const localGpu = GPU_BY_ID.get(t.localGpuId) ?? GPUS[0]
  const localMem = useMemo(() => trainMemory(arch, cfg, t.localGpuCount), [arch, cfg, t.localGpuCount])
  const localTime = useMemo(() => trainTime(arch, cfg, localGpu, t.localGpuCount, 1), [arch, cfg, localGpu, t.localGpuCount])
  const localFits = localMem.total <= localGpu.memoryGB * 1e9 * 0.97

  const ranked = useMemo(
    () => rankTraining(arch, cfg, marketOffers(deferred.market), GPU_BY_ID, deferred.market.priceKind),
    [arch, cfg, deferred.market],
  )
  const shortlist = cheapestPerGpu(ranked).slice(0, 10)
  const fastest = [...ranked].sort((a, b) => a.time.seconds - b.time.seconds)[0]

  const minGpus = useMemo(
    () =>
      GPUS.map((gpu) => {
        for (const n of [1, 2, 4, 8, 16, 32]) {
          const m = trainMemory(arch, cfg, n)
          if (m.total <= gpu.memoryGB * 1e9 * 0.97) return { gpu, n, m }
        }
        return { gpu, n: null as number | null, m: null }
      }),
    [arch, cfg],
  )

  const advanced = state.ui.advanced
  const shardingMatters = t.localGpuCount > 1 || t.sharding !== 'ddp'
  const totalTokens = t.useRl ? t.rl.totalPrompts * t.epochs * t.rl.numGenerations * (t.rl.promptTokens + t.rl.completionTokens) : t.numExamples * t.avgTokensPerExample * t.epochs

  return (
    <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      <aside className="flex flex-col gap-6 lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto lg:pr-1 [&>*]:shrink-0">
        <Card>
          <CardContent className="flex flex-col gap-3">
            <SectionTitle>What do you want to do?</SectionTitle>
            <ChoiceCards
              options={TRAINING_PRESETS.map((p) => ({ id: p.id, label: p.label, detail: p.blurb, icon: p.icon }))}
              value={matchingTrainingPreset(t)}
              onChange={(id) => set(TRAINING_PRESETS.find((p) => p.id === id)!.values)}
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col gap-5">
            {advanced && (
              <div className="flex flex-col gap-3">
                <SectionTitle>Method</SectionTitle>
                <ToggleGroup type="single" variant="outline" value={t.method} onValueChange={(v) => v && set({ method: v as TrainMethod })} className="w-full">
                  <ToggleGroupItem value="full" className="flex-1">Full</ToggleGroupItem>
                  <ToggleGroupItem value="lora" className="flex-1">LoRA</ToggleGroupItem>
                  <ToggleGroupItem value="qlora" className="flex-1">QLoRA</ToggleGroupItem>
                </ToggleGroup>
                <SwitchField
                  label="Reinforcement learning (GRPO)"
                  checked={t.useRl}
                  onChange={(useRl) => set({ useRl })}
                  hint="Online RL with sampled rollouts (TRL GRPOTrainer, Unsloth, verl). Adds a generation engine and rollout KV cache to memory."
                />
                {t.method !== 'full' && (
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField label="LoRA rank" value={t.loraRank} onChange={(loraRank) => set({ loraRank })} min={1} max={1024} hint="Adapter size. 16–32 is typical; higher learns more but uses more memory." />
                    <SelectField<LoraTargets>
                      label="Target modules"
                      value={t.loraTargets}
                      onChange={(loraTargets) => set({ loraTargets })}
                      options={[
                        { value: 'all-linear', label: 'All linear' },
                        { value: 'attention', label: 'Attention only' },
                      ]}
                    />
                  </div>
                )}
              </div>
            )}

            {t.useRl ? (
              <div className="flex flex-col gap-3">
                <SectionTitle>Rollouts</SectionTitle>
                <div className="grid grid-cols-2 gap-3">
                  <NumberField label="Training prompts" value={t.rl.totalPrompts} onChange={(totalPrompts) => setRl({ totalPrompts })} min={1} hint="Distinct tasks the model practises on per epoch." />
                  <NumberField label="Attempts per prompt" value={t.rl.numGenerations} onChange={(numGenerations) => setRl({ numGenerations })} min={2} hint="Completions sampled per prompt (the GRPO group size, G). 8 is common." />
                  <NumberField label="Prompt tokens" value={t.rl.promptTokens} onChange={(promptTokens) => setRl({ promptTokens })} min={1} help={tokensToWords(t.rl.promptTokens)} />
                  <NumberField label="Answer tokens" value={t.rl.completionTokens} onChange={(completionTokens) => setRl({ completionTokens })} min={1} help={tokensToWords(t.rl.completionTokens)} />
                </div>
                {advanced && (
                  <>
                    <div className="grid grid-cols-2 gap-3">
                      <NumberField label="Prompts / step" value={t.rl.promptsPerStep} onChange={(promptsPerStep) => setRl({ promptsPerStep })} min={1} />
                      <NumberField label="KL β" value={t.rl.klBeta} onChange={(klBeta) => setRl({ klBeta })} min={0} step={0.01} hint="β > 0 with full fine-tuning keeps a frozen reference model in memory." />
                    </div>
                    <SwitchField label="Colocated vLLM rollouts" checked={t.rl.colocateVllm} onChange={(colocateVllm) => setRl({ colocateVllm })} />
                    {t.rl.colocateVllm && <SwitchField label="Share weights with trainer" checked={t.rl.shareWeights} onChange={(shareWeights) => setRl({ shareWeights })} hint="Unsloth's standby mode reuses the training weights for inference instead of a second copy." />}
                  </>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <SectionTitle>Your dataset</SectionTitle>
                <div className="grid grid-cols-2 gap-3">
                  <NumberField label="Examples" value={t.numExamples} onChange={(numExamples) => set({ numExamples })} min={1} hint="Number of training conversations / samples." />
                  <NumberField
                    label="Avg tokens / example"
                    value={t.avgTokensPerExample}
                    onChange={(avgTokensPerExample) => set({ avgTokensPerExample })}
                    min={1}
                    help={tokensToWords(t.avgTokensPerExample)}
                    hint="Including the system prompt and tool definitions. Multi-turn tool-calling samples are often 3k–10k tokens."
                  />
                  <NumberField label="Epochs" value={t.epochs} onChange={(epochs) => set({ epochs })} min={0.1} step={0.5} hint="Passes over the dataset. 1–3 is typical for fine-tuning." />
                  <NumberField
                    label="Max sequence length"
                    value={t.seqLen}
                    onChange={(seqLen) => set({ seqLen })}
                    min={128}
                    hint="Longest sample (or packed sequence) the trainer handles. Must be ≥ your longest example — trainers silently truncate otherwise."
                  />
                </div>
                <p className="text-[11px] text-muted-foreground">
                  {fmtCompact(t.numExamples * t.avgTokensPerExample)} tokens per epoch
                  {t.avgTokensPerExample > t.seqLen && <span className="text-destructive"> · examples are longer than the max sequence length</span>}
                </p>
                {advanced && (
                  <div className="grid grid-cols-2 gap-3">
                    <NumberField label="Micro-batch / GPU" value={t.microBatch} onChange={(microBatch) => set({ microBatch })} min={1} />
                    <NumberField label="Gradient accumulation" value={t.gradAccum} onChange={(gradAccum) => set({ gradAccum })} min={1} hint="Doesn't change memory; sets the effective batch with micro-batch × GPUs." />
                  </div>
                )}
              </div>
            )}

            {advanced && (
              <>
                <div className="flex flex-col gap-3">
                  <SectionTitle>Memory savers</SectionTitle>
                  <div className="grid grid-cols-2 gap-3">
                    <SelectField<Optimizer>
                      label="Optimizer"
                      value={t.optimizer}
                      onChange={(optimizer) => set({ optimizer })}
                      options={[
                        { value: 'adamw', label: 'AdamW' },
                        { value: 'adamw-8bit', label: 'AdamW 8-bit' },
                        { value: 'adafactor', label: 'Adafactor' },
                        { value: 'sgd', label: 'SGD + momentum' },
                      ]}
                    />
                    <SelectField<Checkpointing>
                      label="Grad checkpointing"
                      value={t.checkpointing}
                      onChange={(checkpointing) => set({ checkpointing })}
                      options={[
                        { value: 'none', label: 'Off' },
                        { value: 'standard', label: 'On' },
                        { value: 'offload', label: 'On + CPU offload' },
                      ]}
                      hint="Recompute activations in the backward pass (~30% slower, far less memory). Offload is Unsloth's 'unsloth' mode."
                    />
                  </div>
                  <SwitchField label="Chunked / fused loss" checked={t.chunkedLoss} onChange={(chunkedLoss) => set({ chunkedLoss })} hint="Liger / Unsloth cross-entropy never materializes the full logits — big win for 150k–260k vocabularies." />
                </div>

                <div className="flex flex-col gap-3">
                  <SectionTitle>Parallelism</SectionTitle>
                  <SelectField<Sharding>
                    label="Data-parallel sharding"
                    value={t.sharding}
                    onChange={(sharding) => set({ sharding })}
                    options={[
                      { value: 'ddp', label: 'DDP (replicate everything)' },
                      { value: 'zero1', label: 'ZeRO-1 (shard optimizer)' },
                      { value: 'zero2', label: 'ZeRO-2 (+ gradients)' },
                      { value: 'zero3', label: 'ZeRO-3 / FSDP (+ weights)' },
                    ]}
                    hint="DeepSpeed ZeRO stages or PyTorch FSDP. Sharding divides those states across all GPUs in the job."
                  />
                  <SwitchField label="CPU offload" checked={t.cpuOffload} onChange={(cpuOffload) => set({ cpuOffload })} hint="Optimizer states (and with ZeRO-3, parameters) live in CPU RAM. Slower, but fits much larger models." />
                  <NumberField label="Model FLOPs utilization" value={t.mfu} onChange={(mfu) => set({ mfu })} min={0.05} max={0.9} step={0.05} hint="Fraction of peak BF16 FLOPs the trainer achieves. 30–45% is typical for small-model fine-tuning." />
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col gap-3">
            <SectionTitle>Your hardware</SectionTitle>
            <div className="grid grid-cols-[1fr_88px] gap-3">
              <SelectField
                label="GPU"
                value={t.localGpuId}
                onChange={(localGpuId) => set({ localGpuId })}
                options={GPUS.map((g) => ({ value: g.id, label: g.name.replace(/^NVIDIA /, ''), detail: `${g.memoryGB} GB` }))}
              />
              <NumberField label="Count" value={t.localGpuCount} onChange={(localGpuCount) => set({ localGpuCount })} min={1} max={64} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <MarketControls value={state.market} onChange={(p: Partial<MarketInputs>) => update('market', p)} />
          </CardContent>
        </Card>
      </aside>

      <section className="flex min-w-0 flex-col gap-6">
        <TrainVerdict
          fits={localFits}
          gpuLabel={`${t.localGpuCount}× ${localGpu.name.replace(/^NVIDIA /, '')}`}
          needed={localMem.total}
          capacity={localGpu.memoryGB * 1e9}
          hours={localTime.seconds}
          cloud={ranked[0] ? { name: `${ranked[0].nodes > 1 ? `${ranked[0].nodes}× ` : ''}${ranked[0].offer.instanceType}`, gpu: ranked[0].gpu.name.replace(/^NVIDIA /, ''), cost: ranked[0].cost, seconds: ranked[0].time.seconds } : null}
          onSqueeze={t.method !== 'qlora' && !t.useRl ? () => set(TRAINING_PRESETS.find((p) => p.id === 'qlora')!.values) : undefined}
          onMoreGpus={() => set({ localGpuCount: t.localGpuCount * 2, sharding: 'zero3' })}
        />
        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <Tile icon={<Sigma className="size-3.5" />} label="Trainable parameters" value={fmtParams(localMem.trainableParams)} sub={`${((localMem.trainableParams / arch.params.total) * 100).toFixed(t.method === 'full' ? 0 : 2)}% of ${fmtParams(arch.params.total)}`} />
          <Tile icon={<Cpu className="size-3.5" />} label="Memory per GPU" value={fmtBytes(localMem.total)} sub={`${t.localGpuCount}× ${localGpu.name.replace(/^NVIDIA /, '')} · ${localFits ? 'fits' : 'does not fit'}`} tone={localFits ? 'good' : 'bad'} />
          <Tile icon={<Clock className="size-3.5" />} label="Time on your hardware" value={fmtDuration(localTime.seconds)} sub={`${fmtCompact(localTime.tokensPerSecond)} tok/s · ${fmtCompact(totalTokens)} tokens`} />
          <Tile
            icon={<DollarSign className="size-3.5" />}
            label="Cheapest cloud run"
            value={ranked[0] ? fmtUsd(ranked[0].cost) : '—'}
            sub={ranked[0] ? `${ranked[0].nodes > 1 ? `${ranked[0].nodes}× ` : ''}${ranked[0].offer.instanceType} · ${fmtDuration(ranked[0].time.seconds)}` : 'no instance fits'}
          />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Memory per GPU on {t.localGpuCount}× {localGpu.name}</CardTitle>
            <CardDescription>
              {shardingMatters ? `${t.sharding.toUpperCase()} across ${t.localGpuCount} GPU${t.localGpuCount > 1 ? 's' : ''}` : 'Single GPU'} · micro-batch{' '}
              {t.microBatch} × {t.useRl ? t.rl.promptTokens + t.rl.completionTokens : t.seqLen} tokens · ~8% allocator fragmentation included
              {t.useRl && localTime.rolloutSeconds > 0 && ` · ${Math.round((localTime.rolloutSeconds / localTime.seconds) * 100)}% of time spent generating rollouts`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <MemoryBar
              title={localFits ? 'Fits' : 'Out of memory — try QLoRA, gradient checkpointing, a smaller micro-batch, ZeRO-3 or more GPUs'}
              capacity={localGpu.memoryGB * 1e9}
              segments={[
                { key: 'w', label: t.method === 'qlora' ? 'Weights (4-bit base)' : 'Weights', bytes: localMem.weights },
                { key: 'g', label: 'Gradients', bytes: localMem.gradients },
                { key: 'o', label: 'Optimizer + master', bytes: localMem.optimizer },
                { key: 'a', label: 'Activations', bytes: localMem.activations },
                { key: 'l', label: 'Logits / loss', bytes: localMem.logits },
                { key: 'r', label: 'Rollouts / reference', bytes: localMem.rollout },
                { key: 'x', label: 'CUDA + fragmentation', bytes: localMem.overhead + localMem.total - localMem.total / 1.08 },
              ]}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cheapest cloud runs</CardTitle>
            <CardDescription>
              Smallest node count of each instance that fits, ranked by total cost of the run.
              {fastest && ` Fastest: ${fastest.nodes > 1 ? `${fastest.nodes}× ` : ''}${fastest.offer.instanceType} in ${fmtDuration(fastest.time.seconds)} for ${fmtUsd(fastest.cost)}.`}
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Instance</TableHead>
                  <TableHead className="text-right">GPUs</TableHead>
                  <TableHead className="text-right">Mem / GPU</TableHead>
                  <TableHead className="text-right">Duration</TableHead>
                  <TableHead className="text-right">$/hour</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {shortlist.map((o) => (
                  <TableRow key={`${o.offer.provider}:${o.offer.instanceType}:${o.offer.region}`}>
                    <TableCell>
                      <div className="flex flex-col">
                        <span className="font-medium">{o.gpu.name.replace(/^NVIDIA /, '')}</span>
                        <span className="text-xs text-muted-foreground">
                          {PROVIDER_LABEL[o.offer.provider]} · <span className="font-mono">{o.offer.instanceType}</span> ·{' '}
                          {o.offer.region === 'on-prem' ? 'on-prem' : regionName(o.offer.provider, o.offer.region)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {o.nodes > 1 ? `${o.nodes}×${o.offer.gpuCount}` : o.gpusTotal}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {fmtBytes(o.memory.total)} <span className="text-muted-foreground">/ {o.gpu.memoryGB}</span>
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{fmtDuration(o.time.seconds)}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{fmtUsd(o.hourly * o.nodes)}</TableCell>
                    <TableCell className="text-right font-mono font-semibold tabular-nums">{fmtUsd(o.cost)}</TableCell>
                  </TableRow>
                ))}
                {shortlist.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                      No instance fits this configuration.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Minimum GPUs to fit</CardTitle>
            <CardDescription>With the current method and sharding. Switch to ZeRO-3 to let more GPUs split the weights.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {minGpus.map(({ gpu, n, m }) => (
              <Badge key={gpu.id} variant={n ? (n === 1 ? 'secondary' : 'outline') : 'outline'} className={`gap-1.5 py-1 ${n ? '' : 'opacity-50'}`}>
                <span>{gpu.name.replace(/^NVIDIA |^AMD /, '')}</span>
                <span className="font-mono tabular-nums">{n ? `${n}×` : '✕'}</span>
                {m && <span className="font-mono text-muted-foreground tabular-nums">{fmtBytes(m.total, 0)}</span>}
              </Badge>
            ))}
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

function TrainVerdict({
  fits,
  gpuLabel,
  needed,
  capacity,
  hours,
  cloud,
  onSqueeze,
  onMoreGpus,
}: {
  fits: boolean
  gpuLabel: string
  needed: number
  capacity: number
  hours: number
  cloud: { name: string; gpu: string; cost: number; seconds: number } | null
  onSqueeze?: () => void
  onMoreGpus: () => void
}) {
  const tone = fits ? 'var(--status-good)' : 'var(--status-critical)'
  return (
    <div
      className="flex items-start gap-3 rounded-xl border p-4"
      style={{ borderColor: `color-mix(in oklab, ${tone} 40%, transparent)`, background: `color-mix(in oklab, ${tone} 8%, var(--card))` }}
    >
      {fits ? (
        <CheckCircle2 className="mt-0.5 size-5 shrink-0" style={{ color: tone }} aria-label="Fits" />
      ) : (
        <XCircle className="mt-0.5 size-5 shrink-0" style={{ color: tone }} aria-label="Does not fit" />
      )}
      <div className="flex flex-col gap-2 text-sm leading-relaxed">
        {fits ? (
          <p>
            Fits on your <b>{gpuLabel}</b> ({fmtBytes(needed)} of {fmtBytes(capacity, 0)}) and takes about <b>{fmtDuration(hours)}</b>.
          </p>
        ) : (
          <p>
            Doesn’t fit on your <b>{gpuLabel}</b> — needs <b>{fmtBytes(needed)}</b> per GPU, you have {fmtBytes(capacity, 0)}.
          </p>
        )}
        {cloud && (
          <p className="text-muted-foreground">
            Cheapest cloud option: {cloud.name} ({cloud.gpu}) — {fmtDuration(cloud.seconds)} for about <b className="text-foreground">{fmtUsd(cloud.cost)}</b>.
          </p>
        )}
        {!fits && (
          <div className="flex flex-wrap gap-2">
            {onSqueeze && (
              <Button size="sm" variant="outline" onClick={onSqueeze}>
                Try QLoRA (4-bit)
              </Button>
            )}
            <Button size="sm" variant="outline" onClick={onMoreGpus}>
              Use 2× the GPUs with ZeRO-3
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

function Tile({ icon, label, value, sub, tone }: { icon: React.ReactNode; label: string; value: string; sub: string; tone?: 'good' | 'bad' }) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border bg-card p-4">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {icon} {label}
      </span>
      <span className="text-2xl font-semibold tracking-tight tabular-nums">{value}</span>
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {tone && <span className="size-2 rounded-full" style={{ background: tone === 'good' ? 'var(--status-good)' : 'var(--status-critical)' }} />}
        {sub}
      </span>
    </div>
  )
}
