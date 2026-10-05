import { BrainCircuit, FileJson, KeyRound, Link2, Moon, RotateCcw, Sun, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { InferencePanel } from '@/components/InferencePanel'
import { ModelDetailsView, ModelPanel, PasteConfigDialog } from '@/components/ModelCard'
import { ModelPicker } from '@/components/ModelPicker'
import { TrainingPanel } from '@/components/TrainingPanel'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { getToken, setToken } from '@/core/hf'
import { clearModelCache } from '@/hooks/useModel'
import { useAppState } from '@/state'

const REPO_URL = 'https://github.com/syahvan/need-more-vram'

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem('nmv:theme')
      if (saved) return saved === 'dark'
    } catch {
      // ignore
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
    try {
      localStorage.setItem('nmv:theme', dark ? 'dark' : 'light')
    } catch {
      // ignore
    }
  }, [dark])
  return [dark, setDark] as const
}

function TokenDialog({ open, onOpenChange, onSaved }: { open: boolean; onOpenChange: (v: boolean) => void; onSaved: () => void }) {
  const [value, setValue] = useState(getToken)
  useEffect(() => {
    if (open) setValue(getToken())
  }, [open])
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Hugging Face access token</DialogTitle>
          <DialogDescription>
            Needed only for gated (Llama, Gemma, …) or private repos. Create a <b>read</b> token at huggingface.co/settings/tokens and accept the model’s license on its
            page. The token is stored in this browser’s localStorage and sent only to huggingface.co.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="hf-token">Token</Label>
          <Input id="hf-token" type="password" autoComplete="off" placeholder="hf_…" value={value} onChange={(e) => setValue(e.target.value.trim())} />
        </div>
        <DialogFooter className="gap-2">
          {getToken() && (
            <Button
              variant="ghost"
              onClick={() => {
                setToken('')
                setValue('')
                onSaved()
                onOpenChange(false)
              }}
            >
              Remove token
            </Button>
          )}
          <Button
            onClick={() => {
              setToken(value)
              onSaved()
              onOpenChange(false)
            }}
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default function App() {
  const { state, update, reset } = useAppState()
  const [dark, setDark] = useTheme()
  const [tokenOpen, setTokenOpen] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const [copied, setCopied] = useState(false)

  const reload = () => {
    clearModelCache()
    setReloadKey((k) => k + 1)
  }

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard blocked; the URL bar already has the shareable link.
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-3 px-4 sm:px-6">
          <a href="./" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="flex size-7 items-center justify-center rounded-lg bg-foreground text-background">
              <Zap className="size-4" />
            </span>
            need-more-vram
          </a>
          <span className="hidden text-sm text-muted-foreground md:inline">Size your GPUs before they size your bill.</span>
          <div className="ml-auto flex items-center gap-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={copyLink} aria-label="Copy shareable link">
                  <Link2 />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{copied ? 'Copied!' : 'Copy link to this configuration'}</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={reset} aria-label="Reset all inputs">
                  <RotateCcw />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Reset all inputs</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={() => setTokenOpen(true)} aria-label="Hugging Face token">
                  <KeyRound />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Hugging Face token (for gated models)</TooltipContent>
            </Tooltip>
            <Button variant="ghost" size="icon" onClick={() => setDark(!dark)} aria-label="Toggle theme">
              {dark ? <Sun /> : <Moon />}
            </Button>
            <Button variant="ghost" size="icon" asChild>
              <a href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source on GitHub">
                <svg viewBox="0 0 24 24" fill="currentColor" className="size-4" aria-hidden>
                  <path d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2 0 1.9 1.2 1.9 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2 0-.4-.5-1.6.2-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0C17.3 4.7 18.3 5 18.3 5c.7 1.6.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .5Z" />
                </svg>
              </a>
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-4 py-6 sm:px-6">
        <div className="relative overflow-hidden rounded-2xl border bg-card">
          <div className="bg-grid pointer-events-none absolute inset-0 [mask-image:linear-gradient(to_bottom,black,transparent)]" />
          <div className="relative flex flex-col gap-5 p-5 sm:p-6">
            <div className="flex flex-col gap-1">
              <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">How much GPU does your model really need?</h1>
              <p className="text-sm text-muted-foreground">
                Pick a model, describe what you’re building, and get memory, speed, the number of GPUs and the cheapest place to run it — for serving or fine-tuning.
              </p>
            </div>
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
              <div className="flex flex-1 flex-col gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">1 · Pick a model</span>
                <ModelPicker value={state.modelId} onChange={(modelId) => update('modelId', modelId)} />
              </div>
              <div className="flex flex-wrap gap-2">
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-medium text-muted-foreground max-sm:hidden">&nbsp;</span>
                  <PasteConfigDialog
                    onLoaded={(modelId) => update('modelId', modelId)}
                    trigger={
                      <Button variant="outline" className="h-11" title="Use a config.json for a private or local model">
                        <FileJson /> Paste config
                      </Button>
                    }
                  />
                </div>
                <div className="flex flex-col gap-1.5 max-sm:w-full">
                  <span className="text-xs font-medium text-muted-foreground">2 · What for?</span>
                  <Tabs value={state.tab} onValueChange={(tab) => update('tab', tab as typeof state.tab)} className="max-sm:w-full">
                    <TabsList className="h-11 max-sm:w-full">
                      <TabsTrigger value="inference" className="px-4">
                        <Zap /> Serve it
                      </TabsTrigger>
                      <TabsTrigger value="training" className="px-4">
                        <BrainCircuit /> Fine-tune it
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-medium text-muted-foreground">Detail</span>
                  <div className="flex h-11 items-center gap-2 rounded-lg border bg-background px-3">
                    <Label htmlFor="advanced" className="text-sm">
                      Advanced
                    </Label>
                    <Switch id="advanced" checked={state.ui.advanced} onCheckedChange={(advanced) => update('ui', { advanced })} />
                  </div>
                </div>
              </div>
            </div>
            <ModelPanel
              modelId={state.modelId}
              onModelChange={(modelId) => update('modelId', modelId)}
              onOpenSettings={() => setTokenOpen(true)}
              reloadKey={reloadKey}
              onReload={reload}
            >
              {(model) => <ModelDetailsView model={model} />}
            </ModelPanel>
          </div>
        </div>

        <ModelPanel modelId={state.modelId} onModelChange={(modelId) => update('modelId', modelId)} onOpenSettings={() => setTokenOpen(true)} reloadKey={reloadKey} onReload={reload}>
          {(model) =>
            state.tab === 'inference' ? (
              <InferencePanel key={model.id} arch={model.arch} state={state} update={update} />
            ) : (
              <TrainingPanel key={model.id} arch={model.arch} state={state} update={update} />
            )
          }
        </ModelPanel>

        <Card className="bg-muted/30">
          <CardContent className="flex flex-col gap-2 text-xs leading-relaxed text-muted-foreground">
            <p>
              <b className="text-foreground">These are estimates.</b> Memory follows how vLLM / SGLang and PyTorch trainers allocate; speed is a roofline model (decode bound
              by memory bandwidth, prefill by FLOPs) with tunable efficiency factors under “Engine assumptions”. Real numbers depend on kernels, drivers, batch mix and
              prompt-cache hit rates — load-test before you buy.
            </p>
            <p>
              Model details are read live from the Hugging Face Hub. Open source under the MIT license —{' '}
              <a className="underline underline-offset-2 hover:text-foreground" href={REPO_URL} target="_blank" rel="noreferrer">
                contributions welcome
              </a>
              .
            </p>
          </CardContent>
        </Card>
      </main>
      <TokenDialog open={tokenOpen} onOpenChange={setTokenOpen} onSaved={reload} />
    </div>
  )
}
