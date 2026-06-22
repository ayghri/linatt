"""
Pre-tokenize FineWeb-Edu sample-10BT into packed ctx=2048 sequences.

Usage:
    python prepare.py
    python prepare.py data.tokenizer=meta-llama/Llama-2-7b-hf
    python prepare.py data=fineweb_edu_10bt data.num_proc=128

Caches to data.cache_path. Re-running with the same args is a no-op
(datasets caches by fingerprint).
"""

from itertools import chain

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def tokenize(examples, tokenizer, seq_len):
    text = examples["text"]
    input_ids = tokenizer(text)["input_ids"]
    lens = torch.tensor([len(seq) for seq in input_ids]).cumsum(0)
    total_len = (lens[-1] // seq_len) * seq_len
    flat = list(chain(*input_ids))
    chunks = [flat[i : i + seq_len] for i in range(0, total_len, seq_len)]
    return {"input_ids": chunks}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.set_verbosity_info()
    d = cfg.data
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    logger.info(f"Loading tokenizer {d.tokenizer}")
    tok = AutoTokenizer.from_pretrained(d.tokenizer, trust_remote_code=True)

    logger.info(f"Loading dataset {d.dataset} ({d.subset}, {d.split})")
    ds = load_dataset(d.dataset, name=d.subset, split=d.split)
    ds = ds.shuffle(seed=d.seed)
    remove_cols = list(next(iter(ds)).keys())

    logger.info(f"Tokenizing with {d.num_proc} workers, packing to ctx={d.seq_len}")
    ds = ds.map(
        lambda ex: tokenize(ex, tok, d.seq_len),
        batched=True,
        batch_size=d.batch_size,
        remove_columns=remove_cols,
        num_proc=d.num_proc,
        desc="tokenize+pack",
    )

    logger.info(f"Saving to {d.cache_path}")
    ds.save_to_disk(d.cache_path, num_proc=d.num_proc)
    n_tok = len(ds) * d.seq_len
    logger.info(f"Done. {len(ds)} sequences x {d.seq_len} = {n_tok / 1e9:.2f}B tokens.")


if __name__ == "__main__":
    main()
