import { ArrowDownToLine, Check, ChevronsUpDown, Heart, Loader2, Lock, Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { normalizeRepoId, searchModels, type ModelSummary } from '@/core/hf'
import { fmtCompact, fmtParams } from '@/lib/format'

const SUGGESTED = [
  'Qwen/Qwen3.5-4B',
  'Qwen/Qwen3.5-9B',
  'Qwen/Qwen3.6-35B-A3B',
  'google/gemma-4-E4B-it',
  'openai/gpt-oss-20b',
  'meta-llama/Llama-3.1-8B-Instruct',
  'ibm-granite/granite-4.0-h-small',
  'deepseek-ai/DeepSeek-V3',
]

export function ModelPicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([])
      setError(null)
      return
    }
    const ctrl = new AbortController()
    const t = setTimeout(() => {
      setLoading(true)
      searchModels(query, ctrl.signal)
        .then((r) => {
          setResults(r)
          setError(null)
        })
        .catch((e: Error) => !ctrl.signal.aborted && setError(e.message))
        .finally(() => !ctrl.signal.aborted && setLoading(false))
    }, 250)
    return () => {
      clearTimeout(t)
      ctrl.abort()
    }
  }, [query])

  const select = (id: string) => {
    onChange(id)
    setOpen(false)
    setQuery('')
  }
  const direct = normalizeRepoId(query)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} className="h-11 w-full justify-between px-3 text-left font-normal">
          <span className="flex min-w-0 items-center gap-2">
            <Search className="size-4 shrink-0 text-muted-foreground" />
            <span className="truncate font-mono text-sm">{value}</span>
          </span>
          <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] min-w-80 p-0" align="start">
        <Command shouldFilter={false}>
          <CommandInput placeholder="Search any Hugging Face model, or paste org/name…" value={query} onValueChange={setQuery} />
          <CommandList className="max-h-96">
            {loading && (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
                <Loader2 className="size-3 animate-spin" /> Searching the Hub…
              </div>
            )}
            {error && <div className="px-3 py-2 text-xs text-destructive">{error}</div>}
            {direct && !results.some((r) => r.id === direct) && (
              <CommandGroup heading="Open directly">
                <CommandItem value={`direct:${direct}`} onSelect={() => select(direct)}>
                  <span className="font-mono text-sm">{direct}</span>
                </CommandItem>
              </CommandGroup>
            )}
            {query.trim().length < 2 ? (
              <CommandGroup heading="Popular starting points">
                {SUGGESTED.map((id) => (
                  <CommandItem key={id} value={id} onSelect={() => select(id)}>
                    <Check className={`size-4 ${id === value ? 'opacity-100' : 'opacity-0'}`} />
                    <span className="font-mono text-sm">{id}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            ) : (
              <>
                {!loading && !error && results.length === 0 && <CommandEmpty>No models found.</CommandEmpty>}
                {results.length > 0 && (
                  <CommandGroup heading="Hugging Face Hub · by downloads">
                    {results.map((m) => (
                      <CommandItem key={m.id} value={m.id} onSelect={() => select(m.id)} className="flex items-center justify-between gap-3">
                        <span className="flex min-w-0 items-center gap-2">
                          {m.gated && <Lock className="size-3 shrink-0 text-muted-foreground" aria-label="Gated" />}
                          <span className="truncate font-mono text-sm">{m.id}</span>
                        </span>
                        <span className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground tabular-nums">
                          {m.params && <span className="rounded bg-muted px-1.5 py-0.5 font-medium text-foreground">{fmtParams(m.params)}</span>}
                          <span className="hidden items-center gap-1 sm:flex">
                            <ArrowDownToLine className="size-3" />
                            {fmtCompact(m.downloads)}
                          </span>
                          <span className="hidden items-center gap-1 md:flex">
                            <Heart className="size-3" />
                            {fmtCompact(m.likes)}
                          </span>
                        </span>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                )}
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
