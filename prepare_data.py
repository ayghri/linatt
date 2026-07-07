"""Per-document packing, parallel. Each block comes from ONE document.

Usage:
    python prepare_perdoc.py data=slimpajama_15bt \
        data.dataset=gmongaras/SlimPajama-627B_Reupload \
        data.target_tokens=15_000_000_000 \
        data.save_dir=/buckets/workspace/linatt/data/slimpajama_15bt/train
"""

import glob
import math
import multiprocessing as mp
import os

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, logging as hf_logging


class ShardedDatasetStream:

    def __init__(
        self,
        dataset_name,
        tokenizer_name,
        num_workers,
        worker_id,
        target_blocks,
        seq_len,
        split="train",
    ):
        stream = load_dataset(dataset_name, split=split, streaming=True)
        self.stream = stream.shard(num_shards=num_workers, index=worker_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            add_bos_token=True,
            add_eos_token=False,
        )
        self.target_blocks = target_blocks
        self.seq_len = seq_len

        self.produced_blocks = 0
        self.token_buffer = []
        self.text_buffer: list[str] = []

    def flush_tokens(self):

        for ids in self.tokenizer(self.text_buffer)["input_ids"]:
            self.token_buffer.extend(ids)
        self.text_buffer.clear()
        n = len(self.token_buffer) // self.seq_len
        for i in range(n):
            if self.produced_blocks >= self.target_blocks:
                return
            yield {
                "input_ids": self.token_buffer[
                    i * self.seq_len : (i + 1) * self.seq_len
                ]
            }
            self.produced_blocks += 1
        # Trim the consumed whole blocks; keep the remainder for the next flush
        # (continuous packing -> no duplicates, bounded buffer).
        del self.token_buffer[: n * self.seq_len]

    def generate_samples(self):
        for example in self.stream:
            text = example.get("text")
            if not text:
                continue
            self.text_buffer.append(text)
            if len(self.text_buffer) >= 1000:  # batch tokenization for speed
                yield from self.flush_tokens()
                if self.produced_blocks >= self.target_blocks:
                    return
        if self.text_buffer:
            yield from self.flush_tokens()


def _run_worker(args):
    (
        worker_id,
        num_workers,
        target_blocks,
        dataset,
        split,
        seq_len,
        tok_name,
        shard_dir,
    ) = args

    hf_logging.set_verbosity_error()

    sharded_stream = ShardedDatasetStream(
        dataset_name=dataset,
        tokenizer_name=tok_name,
        num_workers=num_workers,
        worker_id=worker_id,
        target_blocks=target_blocks,
        seq_len=seq_len,
        split=split,
    )
    ds = Dataset.from_generator(
        sharded_stream.generate_samples,
    )
    out = os.path.join(shard_dir, f"shard_{worker_id}")
    ds.save_to_disk(out)
    return worker_id, len(ds)


@hydra.main(version_base=None, config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    print(OmegaConf.to_yaml(data_cfg))

    shard_dir = data_cfg.save_dir
    if glob.glob(os.path.join(shard_dir, "shard_*")):
        print(f"Shards already exist in {shard_dir} -- nothing to do.")
        return

    num_workers = data_cfg.num_workers
    total_blocks = math.ceil(int(data_cfg.target_tokens) / data_cfg.seq_len)
    per_worker = math.ceil(total_blocks / num_workers)
    os.makedirs(shard_dir, exist_ok=True)

    print(
        f"{num_workers} workers x {per_worker:,} blocks = up to "
        f"{num_workers * per_worker * data_cfg.seq_len / 1e6:.4f}M tokens (per-doc packing)"
    )

    args = [
        (
            i,
            num_workers,
            per_worker,
            data_cfg.dataset,
            data_cfg.split,
            data_cfg.seq_len,
            data_cfg.tokenizer,
            shard_dir,
        )
        for i in range(num_workers)
    ]
    with mp.get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(_run_worker, args)
    for wid, n in sorted(results):
        print(f"  worker {wid}: {n:,} blocks")


if __name__ == "__main__":
    main()
