# LinAtt — 200M-class linear-attention LLM benchmark

Trains three FLA-family baselines from scratch on 10B FineWeb-Edu tokens and
evaluates them on 8 lm-eval-harness tasks. Drop in your own architecture as a
4th yaml under `conf/model/` and rerun.

**Baselines**

| arch          | layers | d_model | params  | block                    |
| :------------ | :----: | :-----: | :-----: | :----------------------- |
| GatedDeltaNet |  24    |   768   | ~209M   | gated delta + SwiGLU MLP |
| DeltaNet      |  24    |   768   | ~205M   | delta rule + SwiGLU MLP  |
| Mamba2        |  48    |   768   | ~210M   | pure SSM (no MLP)        |

Mamba2 has 2× depth so all three land at iso-params (Gu & Dao 2024 convention).
All use ctx=2048, vocab=32000 (LLaMA SP), bf16, tied embeddings.

**Hardware target**: single node, 8×H100 (80GB), 200 vCPU, 640GB RAM.
**Wall time**: ~3–4h per arch, ~10–12h end-to-end.

---

## 0. One-time setup

```bash
cd LinAtt
bash scripts/setup.sh                          # creates mamba env "linatt"
mamba activate linatt
wandb login                                    # paste your W&B API key
```

If `meta-llama/Llama-2-7b-hf` is preferred over the ungated mirror, set
`HF_TOKEN` and override `data.tokenizer=meta-llama/Llama-2-7b-hf`.

## 1. Pre-tokenize the corpus (one-time, ~30–60 min on this node)

```bash
bash scripts/prepare.sh
```

Caches packed ctx=2048 sequences to `data/HuggingFaceFW/fineweb-edu/sample-10BT/train`.
Re-running is a no-op (HF datasets fingerprint cache).

## 2. Train one arch per node

`scripts/train.sh <arch>` is the primary entry. Run one arch per node so
multiple nodes can train in parallel:

```bash
# node-A
bash scripts/train.sh gated_deltanet_200m
# node-B
bash scripts/train.sh delta_net_200m
# node-C
bash scripts/train.sh mamba2_200m
```

Each invocation runs `preflight.sh` first (env, GPUs, W&B auth, tokenizer,
data cache, model build) and aborts on any failure. The runs are independent
and log separately to W&B project `kata` with run name `<arch>_fineweb_edu_10bt`.
lm-eval-harness on the 8-task suite fires inline at 25%, 50%, 75%, 100% of
training; results appear in W&B as `eval/<task>/<metric>`.

Single-node fallback (sequential, all three on one box, ~10–12h):

```bash
bash scripts/run_all.sh
```

## 3. Eval a saved checkpoint

```bash
bash scripts/eval.sh runs/gated_deltanet_200m_fineweb_edu_10bt
```

Writes `lm_eval.json` next to the checkpoint and prints a summary.

## 4. Adding your own architecture

1. Drop a `conf/model/<your_arch>.yaml` mirroring the three baselines. The
   `model_type` must match a value registered with HF Auto* via `import fla`
   (or your own `AutoConfig.register(...)` call).
2. Run `bash scripts/train.sh <your_arch>`.

The trainer, callbacks, data path, eval, and W&B integration are arch-agnostic.

---

## Layout

```
LinAtt/
├── conf/
│   ├── config.yaml                  # top-level: composes model+data+train
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
├── lm_eval_callback.py              # DDP-safe inline eval
├── bench.py                         # synthetic-data throughput probe
├── scripts/
│   ├── setup.sh                     # bootstrap env
│   ├── prepare.sh                   # tokenize FineWeb-Edu
│   ├── train.sh <arch>              # one arch
│   ├── run_all.sh                   # all baselines
│   ├── eval.sh <ckpt>               # eval saved ckpt
│   ├── sanity.sh                    # 4-min smoke test
│   └── bench_sweep.sh <arch>        # max-batch-size probe
└── runs/                            # per-run output_dir (HF Trainer ckpts)
```

## Recipe (`conf/train/default.yaml`)

| | value |
|---|---|
| optim | AdamW fused, β=(0.9,0.95), wd=0.1 |
| lr | 6e-4 → 6e-5 (cosine_with_min_lr, min_lr_rate=0.1) |
| warmup | 2000 steps |
| grad clip | 1.0 |
| batch | 32 micro × 8 GPUs × 2048 ctx = **524k tokens/step** |
| steps | 19073 (≈10B tokens) |
| precision | bf16 weights + activations, fp32 grads/optim |
| dataloader | 16 workers, pin_memory, persistent |

Override anything via Hydra cli. E.g. larger batch on H100:

```bash
bash scripts/train.sh gated_deltanet_200m \
    train.micro_batch_size=48 train.lr=8e-4
```

Lower the eval cadence:

```bash
bash scripts/train.sh gated_deltanet_200m \
    eval.fractions=[0.5,1.0]
```

## Smoke test (local 2x3090, 2 GPUs, ~4 min)

```bash
bash scripts/sanity.sh gated_deltanet_200m
```

50 train steps on wikitext-2 + piqa lm-eval + standalone eval. Confirms the
full pipeline runs without hitting the dataset.

## Troubleshooting

- **OOM at training start**: reduce `train.micro_batch_size`. Run
  `bash scripts/bench_sweep.sh <arch>` to find the safe max.
- **`AttributeError: 'list' object has no attribute 'keys'`**: transformers
  ≥5.0 changed the `_tied_weights_keys` contract. The setup script pins
  `transformers<5`.
- **HF tokenizer download fails**: switch to `data.tokenizer=fla-hub/gla-1.3B-100B`
  (no auth, same 32k SP).
- **Inline lm-eval OOMs on rank 0**: drop `eval.batch_size` or push
  `train.micro_batch_size` lower to leave more headroom.
