"""
Throughput probe. Synthetic random token batches, full fwd+bwd+optim loop,
DDP-aware. Reports tokens/sec and wall-clock projections to 10B tokens.

Usage:
    accelerate launch --num_processes=2 --mixed_precision=bf16 bench.py \
        model=gated_deltanet_200m train.micro_batch_size=8

    # sweep batch sizes
    for bs in 4 8 12 16 20 24 32 40; do
        accelerate launch --num_processes=2 --mixed_precision=bf16 bench.py \
            model=gated_deltanet_200m train.micro_batch_size=$bs train.max_steps=20 \
            || break
    done
"""

from __future__ import annotations

import logging
import time

import fla  # noqa: F401
import fla_patches  # noqa: F401
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

WARMUP = 5
SEQ_LEN = 2048
TOKENS_TARGET = 10_000_000_000


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # init distributed
    if 'LOCAL_RANK' in __import__('os').environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(__import__('os').environ['LOCAL_RANK'])
        world = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = 0
        world = 1
        rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    bs = cfg.train.micro_batch_size
    n_steps = cfg.train.max_steps if cfg.train.max_steps > 0 else 30
    vocab = cfg.model.hf_kwargs.vocab_size

    if rank == 0:
        logger.info(f'world={world}  bs/gpu={bs}  ctx={SEQ_LEN}  '
                    f'tokens/step={world * bs * SEQ_LEN}')

    # build model
    hf_kwargs = OmegaConf.to_container(cfg.model.hf_kwargs, resolve=True)
    hf_cfg = AutoConfig.for_model(**hf_kwargs)
    model = AutoModelForCausalLM.from_config(hf_cfg, dtype=torch.bfloat16).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank])

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f'params={n_params / 1e6:.1f}M')

    optim = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95),
                              weight_decay=0.1, fused=True)

    # synthetic batch on device
    g = torch.Generator(device=device).manual_seed(42 + rank)

    def gen_batch():
        ids = torch.randint(0, vocab, (bs, SEQ_LEN), device=device, generator=g)
        return ids

    # warmup
    model.train()
    for _ in range(WARMUP):
        ids = gen_batch()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()

    # timed steps
    t0 = time.perf_counter()
    peak_mem = 0
    for _ in range(n_steps):
        ids = gen_batch()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        tokens = world * bs * SEQ_LEN * n_steps
        tps = tokens / elapsed
        sec_for_10b = TOKENS_TARGET / tps
        logger.info(
            f'\n========= BENCH =========\n'
            f'  steps_timed   : {n_steps}\n'
            f'  elapsed       : {elapsed:.2f} s\n'
            f'  step_time     : {elapsed / n_steps * 1000:.0f} ms\n'
            f'  tokens/sec    : {tps / 1e3:.1f} k\n'
            f'  peak_mem/gpu  : {peak_mem:.2f} GB\n'
            f'  10B-token ETA : {sec_for_10b / 3600:.1f} h on this config\n'
            f'==========================\n'
        )

    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
