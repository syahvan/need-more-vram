import { BookOpen, Code2, Dices, Dumbbell, FileText, MessageSquare, Puzzle, Target, Wrench, type LucideIcon } from 'lucide-react'
import type { WeightFormat, KvFormat } from '@/core/precision'
import type { InferenceInputs, TrainingInputs } from '@/state'

export interface InferencePreset {
  id: string
  label: string
  blurb: string
  icon: LucideIcon
  values: Partial<InferenceInputs>
}

export const INFERENCE_PRESETS: InferencePreset[] = [
  {
    id: 'chat',
    label: 'Chat assistant',
    icon: MessageSquare,
    blurb: 'Conversational replies, a short system prompt.',
    values: { inputTokens: 1500, outputTokens: 400, sharedPrefixTokens: 500, prefixCaching: true, callsPerRequest: 1, ttftMs: 1500, minTokPerSec: 15 },
  },
  {
    id: 'agent',
    label: 'AI agent with tools',
    icon: Wrench,
    blurb: 'Big system prompt + tool schemas, short tool-call outputs, several calls per message.',
    values: { inputTokens: 9000, outputTokens: 80, sharedPrefixTokens: 7000, prefixCaching: true, callsPerRequest: 3, ttftMs: 400, minTokPerSec: 50 },
  },
  {
    id: 'rag',
    label: 'Document Q&A (RAG)',
    icon: BookOpen,
    blurb: 'Retrieved passages in every prompt, medium answers.',
    values: { inputTokens: 6000, outputTokens: 300, sharedPrefixTokens: 800, prefixCaching: true, callsPerRequest: 1, ttftMs: 2000, minTokPerSec: 20 },
  },
  {
    id: 'summarize',
    label: 'Long-document summary',
    icon: FileText,
    blurb: 'Very long inputs, longer outputs, latency matters less.',
    values: { inputTokens: 30000, outputTokens: 800, sharedPrefixTokens: 300, prefixCaching: true, callsPerRequest: 1, ttftMs: 8000, minTokPerSec: 10 },
  },
  {
    id: 'code',
    label: 'Coding assistant',
    icon: Code2,
    blurb: 'Repository context in, code out, fast streaming.',
    values: { inputTokens: 8000, outputTokens: 600, sharedPrefixTokens: 2000, prefixCaching: true, callsPerRequest: 1, ttftMs: 1000, minTokPerSec: 40 },
  },
]

export function matchingPreset(i: InferenceInputs): string | null {
  const keys = ['inputTokens', 'outputTokens', 'sharedPrefixTokens', 'callsPerRequest', 'ttftMs', 'minTokPerSec'] as const
  return INFERENCE_PRESETS.find((p) => keys.every((k) => p.values[k] === undefined || p.values[k] === i[k]))?.id ?? null
}

/** Friendly precision choices for the simple view. */
export const QUALITY_LEVELS: { id: string; label: string; detail: string; weights: WeightFormat; kv: KvFormat }[] = [
  { id: 'full', label: 'Full quality', detail: 'BF16 — original weights', weights: 'bf16', kv: 'bf16' },
  { id: 'balanced', label: 'Balanced', detail: 'FP8 — ~half the memory, near-identical quality', weights: 'fp8', kv: 'fp8' },
  { id: 'small', label: 'Smallest', detail: '4-bit — fits small GPUs, some quality loss', weights: 'awq-int4', kv: 'fp8' },
]

export interface TrainingPreset {
  id: string
  label: string
  icon: LucideIcon
  blurb: string
  values: Partial<TrainingInputs>
}

export const TRAINING_PRESETS: TrainingPreset[] = [
  {
    id: 'lora',
    label: 'Teach a new skill',
    icon: Target,
    blurb: 'LoRA adapter — the usual choice for domain or tool-calling fine-tunes.',
    values: { method: 'lora', useRl: false, loraRank: 16, optimizer: 'adamw', checkpointing: 'standard', chunkedLoss: true, sharding: 'ddp', cpuOffload: false },
  },
  {
    id: 'qlora',
    label: 'Squeeze onto my GPU',
    icon: Puzzle,
    blurb: 'QLoRA — 4-bit base model so big models fit consumer cards.',
    values: { method: 'qlora', useRl: false, loraRank: 16, optimizer: 'adamw-8bit', checkpointing: 'offload', chunkedLoss: true, sharding: 'ddp', cpuOffload: false, microBatch: 1 },
  },
  {
    id: 'full',
    label: 'Full fine-tune',
    icon: Dumbbell,
    blurb: 'Update every weight — best ceiling, needs multi-GPU for most models.',
    values: { method: 'full', useRl: false, optimizer: 'adamw', checkpointing: 'standard', chunkedLoss: true, sharding: 'zero3', cpuOffload: false },
  },
  {
    id: 'rl',
    label: 'Reinforcement learning',
    icon: Dices,
    blurb: 'GRPO with sampled rollouts and a reward — after SFT, for agents and reasoning.',
    values: { method: 'lora', useRl: true, loraRank: 16, optimizer: 'adamw', checkpointing: 'standard', chunkedLoss: true, sharding: 'ddp' },
  },
]

export function matchingTrainingPreset(t: TrainingInputs): string | null {
  return (
    TRAINING_PRESETS.find((p) =>
      (Object.keys(p.values) as (keyof TrainingInputs)[]).every((k) => JSON.stringify(p.values[k]) === JSON.stringify(t[k])),
    )?.id ?? null
  )
}

/** ~0.75 English words per token; Indonesian and code are a bit lower. */
export function tokensToWords(tokens: number): string {
  const w = tokens * 0.75
  if (w >= 1000) return `≈ ${(w / 1000).toFixed(w >= 10_000 ? 0 : 1)}k words`
  return `≈ ${Math.round(w / 10) * 10} words`
}
