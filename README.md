# need-more-vram

**Size your GPUs before they size your bill.** Find out how much GPU an open LLM really needs to serve or fine-tune, and the cheapest place to run it.

Search for a model, describe your workload, and get:

- **Inference:** a per-GPU memory breakdown (weights, KV cache, activations, overhead); KV-cache capacity; first-token latency and tokens/s; how much load one replica handles within your latency targets; the replica count from average to peak; and the cheapest AWS / Google Cloud deployments, with tensor-parallel layout and both fixed and autoscaled monthly cost.
- **Training:** memory for full fine-tuning, LoRA, QLoRA and GRPO; DDP and ZeRO-1/2/3 (FSDP) sharding with CPU offload; gradient checkpointing and fused loss; estimated run time on your own GPUs; and the cheapest cloud instance for the run.
- **Scaling playbook:** when to scale out, which metrics to watch, cold-start time and routing advice for the configuration you pick.

## Features

- **Any Hub model, read live.** The app fetches `config.json` and safetensors metadata straight from huggingface.co in your browser, so there is no backend.
  - **Gated models** (Llama, Gemma, …): add a read token. It is stored only in your browser's localStorage and sent only to huggingface.co.
  - **Private or local models:** paste a `config.json`.
- **Architecture-aware:**
  - Dense and MoE models: the active-expert count drives both FLOPs and how many weights each step reads.
  - Attention variants: GQA, MQA and MLA (DeepSeek latent KV).
  - Sliding-window layers (Gemma, gpt-oss) and hybrid linear-attention / SSM models (Qwen3.5 / Qwen3-Next, Granite 4 H, Nemotron-H, LFM2, Jamba), whose state stays the same size whatever the context length.
  - KV sharing across layers and per-layer embeddings (Gemma 3n / 4).
  - Multimodal checkpoints, using their `text_config`.
- **Quantization:** BF16, FP8, INT8, AWQ/GPTQ INT4, NVFP4, MXFP4, GGUF, bitsandbytes NF4, and FP8 KV cache. Embeddings stay in 16-bit, and FP8/FP4 math is used only on GPUs that support it.
- **Serving model:**
  - Continuous batching with prefix caching: a shared system prompt or tool schema is stored and computed once.
  - Chunked-prefill interference, queueing delay, and Little's law. These turn requests/s, or "N requests in flight", into replicas that meet a time-to-first-token target and a minimum tokens/s per user.
  - A "calls per request" input for agents that make several LLM calls per user message.
- **GPU catalog:** 31 data-center, workstation and consumer GPUs (NVIDIA Blackwell / Hopper / Ada / Ampere, AMD MI300-series), with specs from vendor datasheets.
- **Cloud pricing:** a weekly-refreshed snapshot of AWS and Google Cloud on-demand, spot and 1-year-commitment prices for 8 regions, including Jakarta and Singapore. You can also add your own hardware at a flat $/GPU-hour.
- **Shareable links:** every input is encoded in the URL hash.

## How the estimates work

| Quantity | Model |
|---|---|
| Weights | parameters × bytes per weight (format-specific, including scales); embeddings in 16-bit |
| KV cache | `layers_with_kv × kv_elems × context × bytes`, with sliding layers capped at the window and linear / SSM layers as constant state |
| Decode step | `max(bytes read / (bandwidth × eff), FLOPs / (peak × MFU))` + TP all-reduce + fixed step overhead |
| Prefill | `(2 × active params + attention) × new tokens / (peak × MFU)`, minus any prefix-cache hits |
| Capacity under SLO | binary search on arrival rate; at each rate, a damped fixed point of Little's law with M/D/1-style queueing |
| Training memory | weights + gradients + optimizer (+ fp32 master) sharded per ZeRO stage, activations (with or without checkpointing), logits, GRPO rollout KV, about 8% fragmentation |
| Training time | `(6 or 4) × active params (+2 with recompute) × tokens / (GPUs × peak × MFU)`, plus rollout decode time for GRPO |

All efficiency factors (bandwidth efficiency, MFU, step overhead, memory utilization, target utilization) can be changed under **Engine assumptions**. The defaults are calibrated against public vLLM numbers. For example, Llama-3.1-8B BF16 on one H100 comes out at about 400k KV tokens and about 150 tok/s single-stream. These are planning estimates, so load-test your real traffic (for example with `vllm bench serve` or GuideLLM) before you commit to hardware.

## Development

```sh
npm install
npm run dev        # http://localhost:5173/need-more-vram/
npm test           # Vitest: architecture parsing + estimator sanity checks
npm run build      # static site in dist/
```

- `src/core/` holds the pure, framework-free estimation logic, unit-tested against real configs in `src/core/__fixtures__/`.
- `src/data/gpus.json` is the GPU spec catalog and `src/data/pricing.json` is the cloud offers snapshot; [`scripts/`](scripts/README.md) has the pricing updater.
- The UI uses React 19, Tailwind CSS v4 and shadcn/ui.

### Deployment

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on every push to `main`. To set it up, go to **Settings → Pages** and set the source to **GitHub Actions**.

`.github/workflows/update-pricing.yml` refreshes prices every Monday and opens a PR. It needs **Settings → Actions → General → Allow GitHub Actions to create and approve pull requests**. An optional `GCP_API_KEY` secret adds per-region GCP T4/V100 prices.

For other hosts, set `BASE_PATH=/` when building.

## Contributing

Corrections to GPU specs, prices or memory formulas are very welcome, ideally with a link to the primary source, such as a vendor datasheet, engine source code or a benchmark log. Please add a fixture and a test when you add support for a new architecture.

## License

[MIT](LICENSE)
