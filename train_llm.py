"""Entry point: build an fla model + load tokenized dataset, train via the manual
DDP trainer (train_llm.DDPLLMPretrainer).
include CUDA HOME for tilelang

Launch (see launch_llm.sh):
    CUDA_HOME=/media/misc/envs/cuda PATH=$CUDA_HOME/bin:$PATH \
    torchrun --nproc_per_node=4 train_llm.py conf_llm/gated_deltanet_340m_slim15b.yaml
"""

import math
import os
import sys
from contextlib import nullcontext

import torch
import torch.distributed as dist
from datasets import load_from_disk
from omegaconf import OmegaConf
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

    def __init__(self, optimizer, max_steps, warmup_steps, peak_lr, min_lr):

        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.update(0)

    def update(self, t):
        for p_group in self.optimizer.param_groups:
            if t < self.warmup_steps:
                p_group["lr"] = self.min_lr + (self.peak_lr - self.min_lr) * t / max(
                    1, self.warmup_steps
                )
            else:
                p = (t - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                p_group["lr"] = (
                    self.min_lr
                    + (1 + math.cos(math.pi * p)) * (self.peak_lr - self.min_lr) / 2
                )


class DDPLLMPretrainer:

    def __init__(self, model, dataset, config):

        self.config = config
        self.device, self.local_rank, self.rank, self.world_size = init_dist(
            config.get("backend", None)
        )
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
        )
        model = model.to(self.device)

        self.tokens_per_step_worker = (
            self.config.train.tokens_per_step / self.world_size / config.grad_accum
        )
        self.batch_size = (
            self.tokens_per_step_worker + config.data.seq_len - 1
        ) // config.data.seq_len

        self.wamup_steps = (
            config.train.warmup_tokens / self.config.train.tokens_per_step
        )
        self.max_steps = (
            self.config.data.total_tokens // self.config.train.tokens_per_step
        )

        self.model = DistributedDataParallel(model, device_ids=[self.local_rank])
        self.optimizer = AdamW(
            self.model.parameters(),
            **OmegaConf.to_container(config.optimizer, resolve=True),
        )
        self.scheduler = WarmedUpScheduler(
            self.optimizer,
            max_steps=self.max_steps,
            warmup_steps=self.wamup_steps,
            peak_lr=config.train.peak_lr,
            min_lr=config.train.min_lr,
        )

    def is_main(self):
        return self.local_rank == 0

    def save_checkpoint(self, step):
        if not self.is_main():
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"step_{step}.pt")
        torch.save(
            {
                "step": step,
                "model": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def train(self):

        self.model.train()

        assert self.max_steps <= len(self.dataloader) // self.config.grad_accum

        pbar = tqdm(range(self.max_steps), disable=not self.is_main())
        loader = iter(self.dataloader)

        for step in pbar:
            self.optimizer.zero_grad()
            step_loss = torch.zeros((), device=self.device)
            for grad_step in range(self.config.grad_accum):
                sync = grad_step == self.config.grad_accum - 1
                try:
                    input_ids = next(loader)["input_ids"].to(
                        self.device, non_blocking=True
                    )
                except StopIteration:
                    if self.is_main():
                        print(
                            "No more batches to use bro! check your token counts! rescuing this by re-using the data"
                        )
                    loader = iter(self.dataloader)
                    input_ids = next(loader)["input_ids"].to(
                        self.device, non_blocking=True
                    )
                ctx = nullcontext()
                if not sync:
                    ctx = self.model.no_sync()
                with ctx:
                    # bf16 compute (matmuls/kv), fp32 master+grads. Residual is
                    # bf16 via the embedding-output cast hook set in run_llm.py.
                    with torch.autocast(self.device.type, dtype=torch.bfloat16):
                        loss = (
                            self.model(input_ids=input_ids, labels=input_ids).loss
                            / self.config.grad_accum
                        )
                    loss.backward()
                step_loss += loss.detach()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.update(step + 1)  # step 0 lr primed at init; advance to next

            if (step + 1) % self.config.progress_interval == 0:
                dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
                if self.is_main():
                    pbar.set_postfix({"loss": f"{step_loss.item():.4f}"})

            if (step + 1) % self.config.save_interval == 0:
                self.save_checkpoint(step + 1)

        self.save_checkpoint(self.max_steps)
        dist.destroy_process_group()


def main():
    cfg_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("conf_llm/gated_deltanet_340m_slim15b.yaml")
    )
    cfg = OmegaConf.load(cfg_path)

    # ---- model (random init, from_config) ----
    # fp32 MASTER weights; bf16 compute via autocast in the train loop. The
    # embedding-output cast below makes the residual stream bf16 (low activation
    # memory, nanochat-style) WITHOUT downcasting any weight -- so every param
    # keeps an fp32 master + fp32 grad, including the tied embed/lm_head.
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    hf_cfg = AutoConfig.for_model(**hf_kwargs)
    model = AutoModelForCausalLM.from_config(hf_cfg)  # fp32 master
    if cfg.train.get("pure_bf16", False):
        # A/B control: bf16 MASTER weights (the defective recipe). Demonstrates
        # the low-LR frozen-param effect. autocast in the loop is then ~no-op.
        model = model.to(torch.bfloat16)
    else:
        # fp32 master + bf16 residual via embedding-output cast (no weight downcast).
        model.model.embeddings.register_forward_hook(
            lambda mod, inp, out: out.to(torch.bfloat16)
        )

    # ---- dataset (pre-tokenized, packed seq_len blocks) ----
    if not os.path.exists(cfg.data.cache_path):
        raise FileNotFoundError(
            f"Tokenized dataset not found at {cfg.data.cache_path}. "
            f"Run: python prepare_slimpajama.py data=slimpajama_15bt"
        )
    ds = load_from_disk(cfg.data.cache_path)
    ds = ds.with_format("torch", columns=["input_ids"])  # rows -> {"input_ids": tensor}

    # ---- train (DDP init happens inside the trainer) ----
    trainer = DDPLLMPretrainer(model, ds, cfg.train)
    trainer.train()


if __name__ == "__main__":
    main()
