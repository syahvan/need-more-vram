import type { Gpu } from './hardware'

export type WeightFormat = 'bf16' | 'fp8' | 'int8' | 'awq-int4' | 'gptq-int4' | 'nvfp4' | 'mxfp4' | 'gguf-q8_0' | 'gguf-q4_k_m' | 'bnb-nf4'
export type KvFormat = 'bf16' | 'fp8'

interface FormatInfo {
  label: string
  /** Average bytes per quantized weight including scales / zero points. */
  bytes: number
  /** Precision the matmuls run in when the GPU supports it. */
  compute: 'bf16' | 'fp8' | 'fp4' | 'int8'
  /** Hardware feature the fast path needs; otherwise falls back to bf16 compute. */
  needs?: 'fp8' | 'fp4'
  note: string
}

export const WEIGHT_FORMATS: Record<WeightFormat, FormatInfo> = {
  bf16: { label: 'BF16 / FP16', bytes: 2, compute: 'bf16', note: 'Unquantized baseline.' },
  fp8: { label: 'FP8 (W8A8)', bytes: 1.004, compute: 'fp8', needs: 'fp8', note: 'Near-lossless; needs Ada/Hopper or newer for FP8 math.' },
  int8: { label: 'INT8 (W8A8)', bytes: 1.004, compute: 'int8', note: 'SmoothQuant-style W8A8.' },
  'awq-int4': { label: 'AWQ INT4 (W4A16)', bytes: 0.53, compute: 'bf16', note: 'Weight-only, group size 128. Great for memory-bound decode.' },
  'gptq-int4': { label: 'GPTQ INT4 (W4A16)', bytes: 0.53, compute: 'bf16', note: 'Weight-only, group size 128.' },
  nvfp4: { label: 'NVFP4 (W4A4)', bytes: 0.5625, compute: 'fp4', needs: 'fp4', note: 'Blackwell FP4 tensor cores (16-element FP8 scales).' },
  mxfp4: { label: 'MXFP4', bytes: 0.53, compute: 'bf16', note: 'OCP microscaling FP4 (gpt-oss native).' },
  'gguf-q8_0': { label: 'GGUF Q8_0', bytes: 1.0625, compute: 'bf16', note: 'llama.cpp / Ollama.' },
  'gguf-q4_k_m': { label: 'GGUF Q4_K_M', bytes: 0.606, compute: 'bf16', note: 'llama.cpp / Ollama, ~4.85 bits per weight.' },
  'bnb-nf4': { label: 'bitsandbytes NF4', bytes: 0.516, compute: 'bf16', note: 'QLoRA base format; slow for serving.' },
}

export const KV_FORMATS: Record<KvFormat, { label: string; bytes: number }> = {
  bf16: { label: 'BF16', bytes: 2 },
  fp8: { label: 'FP8', bytes: 1 },
}

/** Maps a checkpoint's `quantization_config.quant_method` to our format. */
export function formatFromCheckpoint(q: string | null): WeightFormat {
  switch (q) {
    case 'fp8':
    case 'fbgemm_fp8':
      return 'fp8'
    case 'awq':
      return 'awq-int4'
    case 'gptq':
    case 'int4':
      return 'gptq-int4'
    case 'nvfp4':
    case 'modelopt':
      return 'nvfp4'
    case 'mxfp4':
      return 'mxfp4'
    case 'bnb-nf4':
      return 'bnb-nf4'
    case 'int8':
      return 'int8'
    default:
      return 'bf16'
  }
}

/** Bytes of weights: embeddings stay in 16-bit; everything else uses the format. */
export function weightBytes(totalParams: number, embeddingParams: number, format: WeightFormat): number {
  const f = WEIGHT_FORMATS[format]
  if (format === 'bf16') return totalParams * 2
  return embeddingParams * 2 + (totalParams - embeddingParams) * f.bytes
}

export function computePrecision(format: WeightFormat, gpu: Gpu): 'bf16' | 'fp8' | 'fp4' | 'int8' {
  const f = WEIGHT_FORMATS[format]
  if (f.needs === 'fp8' && !gpu.fp8) return 'bf16'
  if (f.needs === 'fp4' && !gpu.fp4) return gpu.fp8 ? 'fp8' : 'bf16'
  return f.compute
}
