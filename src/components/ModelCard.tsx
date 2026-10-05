import { AlertTriangle, ExternalLink, FileJson, KeyRound, Loader2, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { attentionKind, linearStateBytes, sequenceCacheBytes, type HFConfig } from '@/core/arch'
import type { ModelDetails } from '@/core/hf'
import { registerCustomConfig, useModel } from '@/hooks/useModel'
import { fmtBytes, fmtCompact, fmtInt, fmtParams } from '@/lib/format'

const PERMISSIVE = new Set(['apache-2.0', 'mit', 'bsd-3-clause', 'bsd-2-clause', 'openrail', 'cc-by-4.0'])

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-sm font-semibold tabular-nums">{value}</span>
      {sub && <span className="text-[11px] text-muted-foreground">{sub}</span>}
    </div>
  )
}

export function ModelDetailsView({ model }: { model: ModelDetails }) {
  const a = model.arch
  const kvPerToken = sequenceCacheBytes(a, 1, 2) - linearStateBytes(a)
  const isCustom = model.id.startsWith('custom:')
  const layerNote = [
    a.layers.full && `${a.layers.full} global`,
    a.layers.sliding && `${a.layers.sliding} sliding (${fmtInt(a.slidingWindow ?? 0)})`,
    a.layers.linear && `${a.layers.linear} linear/SSM`,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant="secondary" className="font-mono">{a.modelType}</Badge>
        {a.moe && <Badge variant="secondary">MoE · {a.moe.expertsPerToken}/{a.moe.numExperts} experts</Badge>}
        <Badge variant="secondary">{attentionKind(a)}</Badge>
        {a.layers.linear > 0 && <Badge variant="secondary">Hybrid</Badge>}
        {a.quantization && <Badge variant="outline">Pre-quantized · {a.quantization}</Badge>}
        {model.license && (
          <Badge variant={PERMISSIVE.has(model.license.toLowerCase()) ? 'secondary' : 'outline'} title="License from the model card">
            {model.license}
          </Badge>
        )}
        {model.gated && <Badge variant="outline">Gated</Badge>}
        {!isCustom && (
          <a
            href={`https://huggingface.co/${model.id}`}
            target="_blank"
            rel="noreferrer"
            className="ml-auto flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            {fmtCompact(model.downloads)} downloads <ExternalLink className="size-3" />
          </a>
        )}
      </div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <Stat
          label="Parameters"
          value={fmtParams(a.params.total)}
          sub={a.moe ? `${fmtParams(a.params.active)} active` : a.params.source === 'safetensors' ? 'from safetensors' : 'from config'}
        />
        <Stat label="Layers" value={String(a.numLayers)} sub={layerNote} />
        <Stat label="Hidden / heads" value={`${fmtInt(a.hiddenSize)} / ${a.numHeads}`} sub={a.mla ? `MLA rank ${a.mla.kvLoraRank}` : `${a.numKVHeads} KV heads × ${a.headDim}`} />
        <Stat label="KV cache / token" value={kvPerToken >= 1e6 ? fmtBytes(kvPerToken, 2) : `${fmtInt(kvPerToken / 1024)} KiB`} sub={a.layers.sliding ? 'BF16 · sliding layers stop growing at the window' : a.layers.linear ? 'BF16 · plus constant SSM state' : 'BF16'} />
        <Stat label="Max context" value={fmtCompact(a.maxContext)} sub="max_position_embeddings" />
        <Stat label="Vocab" value={fmtCompact(a.vocabSize)} sub={a.tiedEmbeddings ? 'tied embeddings' : 'untied LM head'} />
      </div>
    </div>
  )
}

export function PasteConfigDialog({ onLoaded, trigger }: { onLoaded: (id: string) => void; trigger: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('my-model')
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)

  const submit = () => {
    try {
      const cfg = JSON.parse(text) as HFConfig
      onLoaded(registerCustomConfig(name.trim(), cfg))
      setOpen(false)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Use a custom config.json</DialogTitle>
          <DialogDescription>
            For private models, local checkpoints or architectures the Hub can’t serve. Paste the model’s <code>config.json</code>; nothing leaves your browser.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="cfg-name">Name</Label>
            <Input id="cfg-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="cfg-json">config.json</Label>
            <Textarea id="cfg-json" value={text} onChange={(e) => setText(e.target.value)} className="h-56 font-mono text-xs" placeholder='{ "model_type": "llama", "hidden_size": 4096, ... }' />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button onClick={submit} disabled={!text.trim()}>
            Load config
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface ModelPanelProps {
  modelId: string
  onModelChange: (id: string) => void
  onOpenSettings: () => void
  reloadKey: number
  onReload: () => void
  children: (model: ModelDetails) => React.ReactNode
}

/** Loads the selected model and renders its summary plus the calculator body. */
export function ModelPanel({ modelId, onModelChange, onOpenSettings, reloadKey, onReload, children }: ModelPanelProps) {
  const state = useModel(modelId, reloadKey)

  if (state.status === 'loading') {
    return (
      <Card>
        <CardContent className="flex items-center gap-2 py-10 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Loading <span className="font-mono">{modelId}</span> from Hugging Face…
        </CardContent>
      </Card>
    )
  }
  if (state.status === 'error') {
    const gated = state.httpStatus === 401 || state.httpStatus === 403
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>Couldn’t load {modelId}</AlertTitle>
        <AlertDescription className="flex flex-col gap-3">
          <p>{state.error}</p>
          <div className="flex flex-wrap gap-2">
            {gated && (
              <Button size="sm" variant="outline" onClick={onOpenSettings}>
                <KeyRound /> Add Hugging Face token
              </Button>
            )}
            <Button size="sm" variant="outline" onClick={onReload}>
              <RefreshCw /> Retry
            </Button>
            <PasteConfigDialog
              onLoaded={onModelChange}
              trigger={
                <Button size="sm" variant="outline">
                  <FileJson /> Paste config.json instead
                </Button>
              }
            />
          </div>
        </AlertDescription>
      </Alert>
    )
  }
  return <>{children(state.model)}</>
}
