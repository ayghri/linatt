# LinAtt — 200M-class linear-attention LLM benchmark

Trains three FLA-family baselines from scratch on 10B FineWeb-Edu tokens and
evaluates them on 9 lm-eval-harness tasks (7 multiple-choice acc + 2 perplexity:
LAMBADA + Wikitext). Drop in your own architecture as a 4th yaml under
`conf/model/` and rerun.

**Baselines**

| arch          | layers | d_model | params  | block                                    |
| :------------ | :----: | :-----: | :-----: | :--------------------------------------- |
| Transformer   |  24    |   768   | ~195M   | full attn (RoPE + qk-norm) + SwiGLU MLP |
| GatedDeltaNet |  24    |   768   | ~209M   | gated delta + SwiGLU MLP                 |
| DeltaNet      |  24    |   768   | ~195M   | delta rule + SwiGLU MLP                  |
| Mamba2        |  48    |   768   | ~205M   | pure SSM (no MLP)                        |

Mamba2 has 2× depth so all four land at iso-params (Gu & Dao 2024 convention).
All use ctx=2048, vocab=32000 (LLaMA SP), bf16, tied embeddings.

The transformer uses **PyTorch's bundled FlashAttention-2** via
`torch.nn.functional.scaled_dot_product_attention` with the FLASH backend
forced (no `flash-attn` package install needed; shim lives in
`fla_patches.py`).

**Target deployment**: one arch per node, multiple nodes in parallel. Each
node has 4×H100. Three nodes → all three baselines train concurrently in
~6–8h wall time.

---

## 0. One-time setup (per node)

```bash
cd LinAtt
bash scripts/setup.sh                   # ~5 min. installs uv if missing, creates LinAtt/.venv
source .venv/bin/activate
wandb login                             # paste API key
```

`setup.sh` does NOT require mamba/conda on the node. It uses
[`uv`](https://github.com/astral-sh/uv) (single static binary; auto-installs
to `~/.local/bin` if missing) to create a python 3.11 venv at
`LinAtt/.venv`, then installs torch 2.10.0+cu128, fla (editable),
transformers <5, hydra, accelerate, wandb, lm-eval, plus prebuilt wheels for
causal-conv1d 1.6.1 and mamba-ssm 2.3.1 (Mamba2 fast kernels).

If `meta-llama/Llama-2-7b-hf` is preferred over the ungated Nous mirror:

```bash
export HF_TOKEN=...
# then override at preprocess time:
bash scripts/prepare.sh data.tokenizer=meta-llama/Llama-2-7b-hf
```

## 1. Pre-tokenize the corpus (one node, shared FS)

```bash
bash scripts/prepare.sh                 # ~30–60 min on 200 vCPU
```

Caches packed ctx=2048 sequences to
`data/HuggingFaceFW/fineweb-edu/sample-10BT/train`. Re-running is a no-op.
**Run on the node whose filesystem the other nodes can read** (or replicate
the cache). Each training node needs `data/...` in `LinAtt/`.

## 2. Smoke test (run before any real training)

Before kicking off a multi-hour run, validate that the full pipeline works
end-to-end on this node. `sanity.sh` tokenizes wikitext-2, runs 50 train
steps + inline lm-eval, saves a checkpoint, and re-evals from the saved
checkpoint. Default runs all 4 baselines sequentially.

```bash
bash scripts/sanity.sh                    # all 4 archs (~5-15 min on H100)
# or one at a time
bash scripts/sanity.sh transformer_200m
```

Pass criteria:
- "Sanity sweep PASSED for N arch(s)" at the end
- four W&B runs visible at https://wandb.ai/${WANDB_ENTITY}/kata
- `runs/sanity_<arch>/checkpoint-50/` directory exists for each arch

If anything fails here, **do not launch real training** — fix the
preflight/sanity error first.

## 3. Train one arch per node, in parallel

The primary entrypoint is `scripts/train.sh <arch>`. It runs `preflight.sh`
first (env, GPUs, W&B auth, tokenizer, data cache, model build) and aborts
on any failure. Then launches `accelerate` on the local GPUs.

### 4×H100 per node — parallel four-way

```bash
# node-A
NUM_GPUS=4 bash scripts/train.sh transformer_200m

# node-B
NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m

# node-C
NUM_GPUS=4 bash scripts/train.sh delta_net_200m

# node-D
NUM_GPUS=4 bash scripts/train.sh mamba2_200m
```

Each invocation is independent — DDP within the node, no inter-node comm.
All three log separately to W&B project `kata` with run name
`<arch>_fineweb_edu_10bt`. Inline lm-eval fires at 25/50/75/100% of training,
each eval point also writes a checkpoint to
`runs/<arch>_fineweb_edu_10bt/checkpoint-<step>/`.

### Throughput estimates (4×H100)

| | value |
|---|---|
| eff. batch | 32 micro × 4 GPUs × 2048 ctx = **262k tokens/step** |
| steps to 10B tokens | ~38k |
| per-arch wall time | ~6–8h |
| three nodes in parallel | ~6–8h end-to-end |

### Single-node sequential fallback

If you only have one node and want to run all three sequentially:

```bash
NUM_GPUS=4 bash scripts/run_all.sh        # ~18–24h on a 4xH100 box
```

## 4. Eval a saved checkpoint

```bash
bash scripts/eval.sh runs/gated_deltanet_200m_fineweb_edu_10bt
# or a specific intermediate ckpt:
bash scripts/eval.sh runs/gated_deltanet_200m_fineweb_edu_10bt/checkpoint-19073
```

Writes `lm_eval.json` next to the checkpoint and prints a summary. Also
useful for re-running eval with a larger `eval.batch_size` than the inline
callback uses (`eval.sh ... eval.batch_size=64`).

## 5. Adding your own architecture

1. Drop `conf/model/<your_arch>.yaml` mirroring the three baselines. The
   `model_type` must match a value registered with HF Auto* via `import fla`
   (or your own `AutoConfig.register(...)` call).
2. If your block needs a fla patch, add it in `LinAtt/fla_patches.py` (don't
   edit `fla/` directly — it's installed editable from the parent repo).
3. Run `NUM_GPUS=4 bash scripts/train.sh <your_arch>`.

The trainer, callbacks, eval, and W&B integration are arch-agnostic.

---

## Hydra overrides (cheat sheet)

Anything in `conf/` can be overridden on the cli. Examples:

```bash
# tune lr
NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m train.lr=8e-4

# bigger batch on H100 (try with bench_sweep.sh first)
NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m train.micro_batch_size=48

# fewer eval points
NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m eval.fractions=[0.5,1.0]

# different W&B project / entity
NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m wandb.project=foo wandb.entity=bar

# custom step count
NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m train.max_steps=20000
```

## Layout

```
LinAtt/
├── README.md
├── conf/
│   ├── config.yaml                  # composes model+data+train; eval/wandb/seed live here
│   ├── model/
│   │   ├── gated_deltanet_200m.yaml
│   │   ├── delta_net_200m.yaml
│   │   └── mamba2_200m.yaml
│   ├── data/
│   │   ├── fineweb_edu_10bt.yaml    # production
│   │   └── sanity.yaml              # wikitext-2, ctx=512 — dev only
│   └── train/
│       ├── default.yaml             # production recipe
│       └── sanity.yaml              # 50 steps, bs=2 — dev only
├── prepare.py                       # tokenize+pack
├── train.py                         # HF Trainer entry
├── eval.py                          # standalone lm-eval on a ckpt
├── lm_eval_callback.py              # DDP-safe inline eval + checkpoint save
├── fla_patches.py                   # runtime patches over installed fla
├── bench.py                         # synthetic-data throughput probe
└── scripts/
    ├── setup.sh                     # bootstrap env (one-time per node)
    ├── preflight.sh                 # env / GPU / wandb / tokenizer / data check
    ├── prepare.sh                   # tokenize FineWeb-Edu (one-time)
    ├── train.sh <arch>              # one arch on this node — primary entry
    ├── run_all.sh                   # single-node sequential fallback
    ├── eval.sh <ckpt>               # eval saved ckpt
    ├── sanity.sh                    # 4-min smoke test on 2 GPUs
    └── bench_sweep.sh <arch>        # max-batch-size probe
```

## Recipe (`conf/train/default.yaml`)

| | value |
|---|---|
| optim | AdamW fused, β=(0.9,0.95), wd=0.1 |
| lr | 6e-4 → 6e-5 (cosine_with_min_lr, min_lr_rate=0.1) |
| warmup | 2000 steps |
| grad clip | 1.0 |
| micro_batch | 32 / GPU |
| ctx | 2048 |
| steps | 19073 (assumes 8×H100 = 524k tok/step) |
| precision | bf16 weights + activations, fp32 grads/optim |
| dataloader | 16 workers, pin_memory, persistent |

> **Note**: `train.max_steps` in `conf/train/default.yaml` is sized for
> 8×H100 (524k tok/step → 19073 steps for 10B tokens). On 4×H100 the
> effective batch halves, so override:
> ```bash
> NUM_GPUS=4 bash scripts/train.sh gated_deltanet_200m \
>     train.max_steps=38146 train.warmup_steps=4000
> ```

## Eval suite

Inline callback at 25/50/75/100% of training; each eval point saves a
checkpoint and logs to W&B:

| task            | metric                                          |
| :-------------- | :---------------------------------------------- |
| piqa            | acc                                             |
| hellaswag       | acc                                             |
| winogrande      | acc                                             |
| arc_easy        | acc                                             |
| arc_challenge   | acc                                             |
| social_iqa      | acc                                             |
| boolq           | acc                                             |
| lambada_openai  | acc + perplexity                                |
| wikitext        | word_perplexity, byte_perplexity, bits_per_byte |

W&B keys are `eval/<task>/<metric>` (e.g. `eval/lambada_openai/perplexity,none`).

## Troubleshooting

- **Preflight fails**: read the [FAIL] line. Usually wandb auth (run
  `wandb login`), a missing dep (re-run `scripts/setup.sh`), or no GPUs
  visible. Make sure `.venv` is activated: `source .venv/bin/activate`.
- **OOM at training start**: reduce `train.micro_batch_size`. Run
  `bash scripts/bench_sweep.sh <arch>` to find the safe max for your GPU.
- **`AttributeError: 'list' object has no attribute 'keys'`**: transformers
  ≥5.0 changed the `_tied_weights_keys` contract. `setup.sh` pins
  `transformers<5` — make sure `.venv` is activated.
- **HF tokenizer download fails**: switch to `data.tokenizer=fla-hub/gla-1.3B-100B`
  (no auth, same 32k SP).
- **Inline lm-eval OOMs on rank 0**: drop `eval.batch_size`, or reduce
  `train.micro_batch_size` to leave more headroom on the eval rank.
- **Mamba2 prints "Falling back to Triton"**: mamba-ssm/causal-conv1d wheel
  install failed. Check `setup.sh` output. Triton fallback is correct but
  ~10–50× slower.
