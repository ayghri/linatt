"""Manual DDP LLM pretrainer. fp32 master + bf16 compute (autocast) + bf16
residual via an embedding-output cast. Loads the per-doc packed shards written
by prepare_data.py and trains with wandb logging (loss, grad_norm, lr, tok/s).

Launch (CUDA_HOME needed for fla's tilelang GDN backward):
    CUDA_HOME=/media/misc/envs/cuda PATH=$CUDA_HOME/bin:$PATH \
    torchrun --nproc_per_node=8 train_llm.py \
        model=gated_deltanet_340m data=slimpajama_15bt train=manual
"""

import glob
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
            g["lr"] = lr
        return lr


class DDPLLMPretrainer:

    def __init__(self, model, dataset, cfg):
        self.cfg = cfg
        tr, data = cfg.train, cfg.data
        self.device, self.local_rank, self.rank, self.world_size = init_dist(
            tr.get("backend", None)
        )

        # batch_size/GPU is derived from the token budget, not configured directly.
        self.tokens_per_step = int(tr.tokens_per_step)
        self.grad_accum = int(tr.grad_accum)
        self.seq_len = int(data.seq_len)
        self.batch_size = (
            self.tokens_per_step // self.world_size // self.grad_accum // self.seq_len
        )
        assert (
            self.batch_size > 0
        ), "tokens_per_step too small for world*grad_accum*seq_len"

        self.max_steps = int(data.target_tokens) // self.tokens_per_step
        self.warmup_steps = int(tr.warmup_tokens) // self.tokens_per_step
        self.progress_interval = int(tr.progress_interval)
        # checkpoint every checkpoint_rate * max_steps steps (+ step 0 and last).
        self.checkpoint_interval = max(
            1, round(float(cfg.checkpoint_rate) * self.max_steps)
        )
        self.max_grad_norm = float(tr.max_grad_norm)
        self.save_dir = cfg.output_dir

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
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(tr.lr),  # peak; overwritten each step by scheduler
            betas=tuple(tr.betas),
            weight_decay=float(tr.weight_decay),
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
        resume = tr.get("resume", None)
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
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "tokens_seen": step * self.tokens_per_step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        torch.save(ckpt, os.path.join(self.save_dir, f"step_{step}.pt"))
        # stable name so `resume=latest` always finds the newest state.
        torch.save(ckpt, os.path.join(self.save_dir, "latest.pt"))

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
                f"({ckpt.get('tokens_seen', 0)/1e9:.2f}B tokens)"
            )

    def train(self):
        self.model.train()
        assert self.max_steps <= len(self.dataloader) // self.grad_accum, (
            f"need {self.max_steps} steps but only "
            f"{len(self.dataloader) // self.grad_accum} available"
        )

        # Single pass with a fixed epoch -> deterministic order, so skipping
        # start_step*grad_accum microbatches resumes at the exact data position.
        self.sampler.set_epoch(0)
        loader = iter(self.dataloader)
        if self.start_step > 0:
            for _ in tqdm(
                range(self.start_step * self.grad_accum),
                desc="skip seen data",
                disable=not self.is_main(),
            ):
                next(loader)

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
                try:
                    input_ids = next(loader)["input_ids"].to(
                        self.device, non_blocking=True
                    )
                except StopIteration:
                    loader = iter(self.dataloader)
                    input_ids = next(loader)["input_ids"].to(
                        self.device, non_blocking=True
                    )
                ctx = nullcontext() if sync else self.model.no_sync()
                with ctx:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16):
                        loss = (
                            self.model(input_ids=input_ids, labels=input_ids).loss
                            / self.grad_accum
                        )
                    loss.backward()
                step_loss += loss.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            lr = self.scheduler.update(
                step + 1
            )  # step 0 primed at init; advance to next

            # Cross-worker reduced loss only every progress_interval -- all_reduce is
            # a collective, so ALL ranks must call it (outside the is_main guard).
            log_reduced = (step + 1) % self.progress_interval == 0
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
                    "lr": lr,
                    "tokens_per_sec": tok_per_sec,
                    "tokens_seen": (step + 1) * self.tokens_per_step,
                }
                if log_reduced:
                    metrics["loss_reduced"] = reduced.item()  # mean across workers
                wandb.log(metrics, step=step + 1)
                pbar.set_postfix(
                    {
                        "loss": f"{step_loss.item():.4f}",
                        "tok/s": f"{tok_per_sec/1e3:.0f}k",
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


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    # ---- model: fp32 MASTER weights; bf16 compute via autocast in the loop. ----
    # fla's fused triton kernels (norm/swiglu/(linear-)CE + chunk-scan mixer) are
    # the optimization; we run them eager (no torch.compile -- Inductor can't
    # trace them and only fuses un-fused ops, of which fla leaves none).
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**hf_kwargs))
    if cfg.train.get("pure_bf16", False):
        model = model.to(torch.bfloat16)  # A/B control: bf16 master (defective)
    else:
        # bf16 residual via embedding-output cast; weights stay fp32 (fp32 master + grad).
        model.model.embeddings.register_forward_hook(
            lambda mod, inp, out: out.to(torch.bfloat16)
        )

    dataset = load_sharded_dataset(cfg.data.save_dir)

    trainer = DDPLLMPretrainer(model, dataset, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
