"""Manual DDP LLM pretrainer. fp32 master + bf16 compute (autocast) + bf16
residual via an embedding-output cast. Loads the per-doc packed shards written
by prepare_data.py and trains with wandb logging (loss, grad_norm, lr, tok/s).

Launch (CUDA_HOME needed for fla's tilelang GDN backward):
    CUDA_HOME=/media/misc/envs/cuda PATH=$CUDA_HOME/bin:$PATH \
    torchrun --nproc_per_node=8 train_llm.py \
        model=gated_deltanet_340m data=slimpajama_15bt train=manual
"""

import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import concatenate_datasets, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

import fla_patches  # noqa: F401  -- kata + SDPA shim
import fla  # noqa: F401  -- registers fla model_types with HF Auto*


def init_dist(backend=None):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    accelerator = torch.accelerator.current_accelerator()
    torch.accelerator.set_device_index(local_rank)
    if backend is None:  # default per-device (nccl on cuda); override e.g. "gloo"
        backend = dist.get_default_backend_for_device(accelerator)  # type: ignore
    dist.init_process_group(backend)
    rank = dist.get_rank()
    device = torch.device(accelerator.type, local_rank)  # type: ignore
    return device, local_rank, rank, dist.get_world_size()


def _grad_group(name):
    """Bucket a parameter name into a coarse module group for grad-norm logging."""
    n = name.lower()
    if "conv1d" in n:
        return "conv"
    if "q_proj" in n or "k_proj" in n or "v_proj" in n:
        return "qkv_proj"
    if "o_proj" in n:
        return "o_proj"
    if "q_norm" in n or "k_norm" in n:
        return "qk_norm"
    if (
        "a_proj" in n
        or "b_proj" in n
        or "g_proj" in n
        or "a_log" in n
        or "dt_bias" in n
    ):
        return "gdn_gate"  # GDN decay / beta / output gate
    if "embeddings" in n or "embed" in n:
        return "embed"
    if "lm_head" in n:
        return "head"
    if (
        "mlp" in n
        or "gate_proj" in n
        or "up_proj" in n
        or "down_proj" in n
        or "swiglu" in n
    ):
        return "mlp"
    if "norm" in n:
        return "norm"
    return "other"


def clip_grad_and_group_norms(
    named_parameters, max_norm, norm_type=2.0, return_groups=True
):
    """Drop-in for torch.nn.utils.clip_grad_norm_ that can ALSO return per-module-group
    grad norms -- computing each parameter's grad norm only ONCE (via the same
    torch._foreach_norm the clip uses) and reusing it for both the total-norm clip
    and the grouping.

    With return_groups=False this is exactly torch's clip (no grouping overhead), so
    the per-group breakdown can be computed only at a logging interval. Mutates grads
    in place (clips). DDP has already synced grads, so the values are global.
    """
    names, grads = [], []
    for name, p in named_parameters:
        if p.grad is not None:
            if return_groups:
                names.append(name)
            grads.append(p.grad)
    if not grads:
        return torch.tensor(0.0), {}
    # per-parameter norms in one fused pass -- exactly what clip_grad_norm_ does
    norms = torch._foreach_norm(grads, norm_type)
    norms_t = torch.stack(norms)
    total_norm = torch.linalg.vector_norm(norms_t, norm_type)
    # clip in place (launch before the CPU sync so the foreach_mul runs async)
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, clip_coef)
    if not return_groups:
        return total_norm, {}
    # group: ||group|| = sqrt(sum of per-param norm^2). One .tolist() sync, then pure python.
    sq = {}
    for name, nv in zip(names, norms_t.tolist()):
        g = _grad_group(name)
        sq[g] = sq.get(g, 0.0) + nv * nv
    return total_norm, {g: math.sqrt(v) for g, v in sq.items()}


def build_lr_param_groups(model, weight_decay, lr_mult):
    """AdamW param groups with per-pattern LR multipliers.

    lr_mult: dict {regex: multiplier} (or list of [regex, multiplier]); the FIRST
    pattern whose regex matches a parameter's name sets its multiplier, default 1.0.
    Each group carries `lr_mult`; the scheduler sets group lr = base_lr * lr_mult.
    e.g. {"q_proj|k_proj|v_proj|o_proj|a_proj|dt_proj|conv1d": 2.0} -> kata attn at 2x,
    MLP/embeddings stay at base.
    """
    import re

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not lr_mult:
        return [
            {
                "params": [p for _, p in params],
                "lr_mult": 1.0,
                "weight_decay": weight_decay,
                "name": "all",
            }
        ]
    rules = list(lr_mult.items() if isinstance(lr_mult, dict) else lr_mult)
    # one named group per pattern (+ a "base" group for unmatched); first match wins.
    buckets = {"base": (1.0, [])}
    for pat, m in rules:
        buckets[pat] = (float(m), [])
    for name, p in params:
        key = next((pat for pat, m in rules if re.search(pat, name)), "base")
        buckets[key][1].append(p)
    groups = [
        {"params": ps, "lr_mult": mult, "weight_decay": weight_decay, "name": key}
        for key, (mult, ps) in buckets.items()
        if ps
    ]
    print(
        "[lr_mult] groups: "
        + ", ".join(
            f"{g['name']}(x{g['lr_mult']:g}): {sum(p.numel() for p in g['params']) / 1e6:.1f}M"
            for g in groups
        )
    )
    return groups


class WarmedUpScheduler:
    """Linear warmup (min_lr -> peak_lr) then cosine decay back to min_lr,
    where min_lr = min_lr_rate * peak_lr (peak_lr is the configured `lr`)."""

    def __init__(self, optimizer, max_steps, warmup_steps, peak_lr, min_lr_rate):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_rate
        self.update(0)  # prime lr for step 0 so update() runs AFTER optimizer.step()

    def get_lr(self, t):
        if t < self.warmup_steps:
            return self.min_lr + (self.peak_lr - self.min_lr) * t / max(
                1, self.warmup_steps
            )
        p = (t - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return (
            self.min_lr + (1 + math.cos(math.pi * p)) * (self.peak_lr - self.min_lr) / 2
        )

    def update(self, t):
        lr = self.get_lr(t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)  # per-group LR multiplier
        return lr


class DDPLLMPretrainer:
    def __init__(self, model, dataset, cfg):
        self.cfg = cfg
        tr, data = cfg.train, cfg.data
        self.device, self.local_rank, self.rank, self.world_size = init_dist(
            tr.get("backend")
        )

        # batch_size/GPU is derived from the token budget, not configured directly.
        self.tokens_per_step = int(tr.tokens_per_step)
        self.grad_accum = int(tr.grad_accum)
        self.seq_len = int(data.seq_len)
        self.batch_size = (
            self.tokens_per_step // self.world_size // self.grad_accum // self.seq_len
        )
        assert self.batch_size > 0, (
            "tokens_per_step too small for world*grad_accum*seq_len"
        )

        self.max_steps = int(data.target_tokens) // self.tokens_per_step
        self.warmup_steps = int(tr.warmup_tokens) // self.tokens_per_step
        self.progress_interval = int(tr.progress_interval)
        # checkpoint every checkpoint_rate * max_steps steps (+ step 0 and last).
        self.checkpoint_interval = max(
            1, round(float(cfg.checkpoint_rate) * self.max_steps)
        )
        self.max_grad_norm = float(tr.max_grad_norm)
        self.save_dir = cfg.output_dir
        self.keep_last_k = int(cfg.get("keep_last_k", 5))  # checkpoints to retain

        self.sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.model = DistributedDataParallel(
            model.to(self.device), device_ids=[self.local_rank]
        )
        # per-layer-type LR multipliers: train.lr_mult overrides the model config's
        # own default (e.g. kata_quadratic_m1_340m -> {attn: 5/3}); as a plain dict.
        lr_mult = tr.get("lr_mult") or cfg.model.get("lr_mult")
        lr_mult = OmegaConf.to_container(lr_mult, resolve=True) if lr_mult else None
        param_groups = build_lr_param_groups(
            self.model, float(tr.weight_decay), lr_mult
        )
        self.optimizer = AdamW(
            param_groups,
            lr=float(tr.lr),  # peak; overwritten each step by scheduler (x lr_mult)
            betas=tuple(tr.betas),
        )
        self.scheduler = WarmedUpScheduler(
            self.optimizer,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            peak_lr=float(tr.lr),
            min_lr_rate=float(tr.min_lr_rate),
        )

        # ---- resume (optional): null | "latest" | path/to/step_N.pt ----
        self.start_step = 0
        resume = tr.get("resume")
        if resume:
            if resume == "latest":
                resume = os.path.join(self.save_dir, "latest.pt")
            if os.path.exists(resume):
                self.load_checkpoint(resume)  # sets self.start_step
            elif self.is_main():
                print(f"resume='{resume}' not found -- starting fresh.")

        if self.is_main():
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                mode=cfg.wandb.mode,
                name=cfg.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print(
                f"batch_size/GPU={self.batch_size}  max_steps={self.max_steps}  "
                f"warmup_steps={self.warmup_steps}  tokens/step={self.tokens_per_step}"
            )

    def is_main(self):
        return self.rank == 0

    def save_checkpoint(self, step):
        if not self.is_main():
            return
        os.makedirs(self.save_dir, exist_ok=True)
        ckpt = {
            "step": step,
            # model architecture config, so resume rebuilds the EXACT model from the
            # checkpoint -- never from the (possibly later-changed) yaml/code defaults.
            "config": self.model.module.config.to_dict(),
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "tokens_seen": step * self.tokens_per_step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        dst = os.path.join(self.save_dir, f"step_{step}.pt")
        tmp = dst + ".tmp"
        # ATOMIC save: write to a temp file, fsync, then rename. A crash or a full
        # disk mid-write leaves the PREVIOUS checkpoints intact -- never a partial dst.
        try:
            with open(tmp, "wb") as f:
                torch.save(ckpt, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)  # atomic
        except OSError as e:  # disk full / io error -> do NOT corrupt or abort
            try:
                os.remove(tmp)
            except OSError:
                pass
            print(
                f"[checkpoint] FAILED to save step {step}: {e}. "
                f"Previous checkpoints are intact; continuing training.",
                flush=True,
            )
            return
        # latest.pt -> a symlink to the new step file (atomic; avoids a 2nd full-size
        # copy that was doubling disk usage and filling the volume).
        link = os.path.join(self.save_dir, "latest.pt")
        try:
            ltmp = link + ".tmp"
            if os.path.lexists(ltmp):
                os.remove(ltmp)
            os.symlink(os.path.basename(dst), ltmp)  # relative target
            os.replace(ltmp, link)
        except OSError:
            pass
        try:  # human-readable arch copy (atomic)
            ctmp = os.path.join(self.save_dir, "config.json.tmp")
            self.model.module.config.to_json_file(ctmp)
            os.replace(ctmp, os.path.join(self.save_dir, "config.json"))
        except OSError:
            pass
        # Prune old step checkpoints to bound disk usage (keep the newest K).
        try:
            steps = sorted(
                (
                    p
                    for p in os.listdir(self.save_dir)
                    if p.startswith("step_") and p.endswith(".pt")
                ),
                key=lambda p: int(p[5:-3]),
            )
            for old in steps[: -self.keep_last_k]:
                os.remove(os.path.join(self.save_dir, old))
        except (OSError, ValueError):
            pass

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.module.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        self.start_step = int(ckpt["step"])
        self.scheduler.update(self.start_step)  # restore lr for the resume step
        if self.is_main():
            print(
                f"resumed from {path} at step {self.start_step} "
                f"({ckpt.get('tokens_seen', 0) / 1e9:.2f}B tokens)"
            )

    def _next_input_ids(self):
        """Next batch's input_ids on device, cycling the loader at epoch end."""
        try:
            batch = next(self._loader)
        except StopIteration:
            self._loader = iter(self.dataloader)
            batch = next(self._loader)
        return batch["input_ids"].to(self.device, non_blocking=True)

    def train(self):
        self.model.train()
        assert self.max_steps <= len(self.dataloader) // self.grad_accum, (
            f"need {self.max_steps} steps but only "
            f"{len(self.dataloader) // self.grad_accum} available"
        )

        # Single pass with a fixed epoch -> deterministic order, so skipping
        # start_step*grad_accum microbatches resumes at the exact data position.
        self.sampler.set_epoch(0)
        self._loader = iter(self.dataloader)
        if self.start_step > 0:
            for _ in tqdm(
                range(self.start_step * self.grad_accum),
                desc="skip seen data",
                disable=not self.is_main(),
            ):
                next(self._loader)

        pbar = tqdm(
            range(self.start_step, self.max_steps),
            initial=self.start_step,
            total=self.max_steps,
            disable=not self.is_main(),
        )

        if self.start_step == 0:
            self.save_checkpoint(0)  # sanity: checkpoint the random init at step 0

        t_last = time.perf_counter()  # start the throughput clock AFTER the init save

        for step in pbar:
            self.optimizer.zero_grad()
            step_loss = torch.zeros((), device=self.device)
            for g in range(self.grad_accum):
                sync = g == self.grad_accum - 1
                input_ids = self._next_input_ids()
                ctx = nullcontext() if sync else self.model.no_sync()
                with ctx:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16):
                        loss = (
                            self.model(input_ids=input_ids, labels=input_ids).loss
                            / self.grad_accum
                        )
                    loss.backward()
                step_loss += loss.detach()

            # Clip EVERY step; compute the per-group grad-norm breakdown only at the
            # loss_reduced cadence (return_groups=False -> exactly torch's clip cost).
            log_reduced = (step + 1) % self.progress_interval == 0
            grad_norm, group_norms = clip_grad_and_group_norms(
                self.model.module.named_parameters(),
                self.max_grad_norm,
                return_groups=log_reduced,
            )
            self.optimizer.step()
            lr = self.scheduler.update(
                step + 1
            )  # step 0 primed at init; advance to next

            # Cross-worker reduced loss only every progress_interval -- all_reduce is
            # a collective, so ALL ranks must call it (outside the is_main guard).
            if log_reduced:
                reduced = step_loss.detach().clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.AVG)

            if self.is_main():
                now = time.perf_counter()
                tok_per_sec = self.tokens_per_step / (now - t_last)
                t_last = now
                metrics = {
                    "loss": step_loss.item(),  # worker-0 local loss, EVERY step
                    "grad_norm": grad_norm.item(),
                    "lr": lr,  # base scheduled lr (warmup+cosine); per-group = lr * lr_mult
                    "tokens_per_sec": tok_per_sec,
                    "tokens_seen": (step + 1) * self.tokens_per_step,
                }
                # actual per-group LRs (base x lr_mult) so the REAL rates are visible
                for grp in self.optimizer.param_groups:
                    metrics[f"lr/{grp.get('name', 'all')}"] = grp["lr"]
                if log_reduced:
                    metrics["loss_reduced"] = reduced.item()  # mean across workers
                    metrics.update({f"gradnorm/{g}": v for g, v in group_norms.items()})
                wandb.log(metrics, step=step + 1)
                pbar.set_postfix(
                    {
                        "loss": f"{step_loss.item():.4f}",
                        "tok/s": f"{tok_per_sec / 1e3:.0f}k",
                    }
                )

            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step + 1)
                t_last = time.perf_counter()  # exclude save time from next tok/s window

        self.save_checkpoint(self.max_steps)
        if self.is_main():
            wandb.finish()
        dist.destroy_process_group()


def load_sharded_dataset(save_dir):
    """Concatenate the per-worker shards prepare_data.py wrote to {save_dir}/shard_*."""
    shard_paths = sorted(glob.glob(os.path.join(save_dir, "shard_*")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards at {save_dir}/shard_*. Run: python prepare_data.py"
        )
    ds = concatenate_datasets([load_from_disk(p) for p in shard_paths])
    return ds.with_format("torch", columns=["input_ids"])


def build_model(cfg):
    """Construct the model architecture.

    Base architecture comes from the NAMED config (cfg.model.hf_kwargs). On resume,
    if the checkpoint path has a saved model config, its fields OVERRIDE the base --
    so a resumed run uses the exact architecture it was trained with, while later
    yaml/code-default changes (e.g. a flipped `rope_group` default) only affect
    fields the checkpoint never recorded. Backward compatible: old checkpoints with
    no saved config simply use the yaml.
    """
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    resume = cfg.train.get("resume")
    if resume == "latest":
        resume = os.path.join(cfg.output_dir, "latest.pt")
    saved = None
    if resume and os.path.exists(resume):
        cfg_json = os.path.join(os.path.dirname(resume) or ".", "config.json")
        if os.path.exists(cfg_json):
            with open(cfg_json) as f:
                saved = json.load(f)
        else:  # older checkpoints embed the config in the .pt instead of config.json
            saved = torch.load(resume, map_location="cpu", weights_only=False).get(
                "config"
            )
    if saved:
        hf_kwargs = {
            **hf_kwargs,
            **saved,
        }  # checkpoint config overrides the named config
        print(
            f"[resume] arch = cfg.model.hf_kwargs overridden by saved checkpoint config "
            f"({resume})"
        )
    # Hybrid: attn_freq=F -> every F-th layer is full softmax attention, rest linear.
    # Model-agnostic (kata + fla GDN both read config.attn = {layers, num_heads}).
    freq = hf_kwargs.pop("attn_freq", None)
    if freq and not hf_kwargs.get("attn"):
        n_layers = hf_kwargs["num_hidden_layers"]
        attn_layers = [i for i in range(n_layers) if (i + 1) % freq == 0]
        hf_kwargs["attn"] = {
            "layers": attn_layers,
            "num_heads": hf_kwargs.get("attn_num_heads", hf_kwargs["num_heads"]),
        }
        print(f"[hybrid] attn_freq={freq} -> full-attention layers {attn_layers}")
    hf_kwargs.pop("attn_num_heads", None)
    return AutoModelForCausalLM.from_config(AutoConfig.for_model(**hf_kwargs))


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    model = build_model(cfg)
    # fp32 MASTER weights + fp32 AdamW state; bf16 ONLY for the forward/backward compute.
    # We do NOT cast the model to bf16 -- that would make AdamW's moments bf16 and drop
    # weight updates smaller than the weight's bf16 ULP (~0.4%), stalling training.
    # Instead: keep weights fp32, run the forward under torch.autocast(bf16) (see .train),
    # and cast the token-embedding OUTPUT to bf16 so the RESIDUAL STREAM is bf16 -- else
    # hidden states stay fp32 and the whole forward runs in fp32, losing the bf16 speedup.
    model.model.embeddings.register_forward_hook(
        lambda mod, inp, out: out.to(torch.bfloat16)
    )
    dataset = load_sharded_dataset(cfg.data.save_dir)

    trainer = DDPLLMPretrainer(model, dataset, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
