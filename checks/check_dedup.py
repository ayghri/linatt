"""Check tokenized shards for DUPLICATE blocks.

Catches the un-trimmed token_buffer bug in prepare_data.py (every flush
re-emitting earlier blocks -> heavy duplication -> memorization -> fake-low
loss). Hashes every block's full input_ids across all shards and reports the
duplicate rate. Exits non-zero if any duplicates are found.

Usage:
    python checks/check_dedup.py --save_dir /buckets/datasets/pajama_shards
    python checks/check_dedup.py --save_dir <dir> --limit 200000   # cap blocks (fast)
"""
import argparse
import glob
import os
import sys

from datasets import load_from_disk


def parse_args():
    p = argparse.ArgumentParser(description="Check tokenized shards for duplicate blocks.")
    p.add_argument("--save_dir", required=True,
                   help="directory containing shard_* subfolders")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many blocks (default: scan all)")
    p.add_argument("--batch_size", type=int, default=2000,
                   help="rows loaded per iteration (default: 2000)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=== check_dedup args ===")
    for k, v in vars(args).items():
        print(f"  {k} = {v}")
    print("========================")

    shard_paths = sorted(glob.glob(os.path.join(args.save_dir, "shard_*")))
    if not shard_paths:
        print(f"ERROR: no shards at {args.save_dir}/shard_*")
        sys.exit(2)
    print(f"found {len(shard_paths)} shards")

    seen = set()
    total = dups = 0
    seq_len = None
    dup_examples = []

    for p in shard_paths:
        ds = load_from_disk(p)
        for batch in ds.iter(batch_size=args.batch_size):
            for ids in batch["input_ids"]:
                if seq_len is None:
                    seq_len = len(ids)
                total += 1
                key = hash(tuple(ids))  # full-block hash (exact)
                if key in seen:
                    dups += 1
                    if len(dup_examples) < 5:
                        dup_examples.append(list(ids[:16]))
                else:
                    seen.add(key)
                if args.limit and total >= args.limit:
                    break
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break

    dup_rate = dups / total if total else 0.0
    tokens = total * (seq_len or 0)
    print(f"\nshards={len(shard_paths)}  blocks={total:,}  unique={len(seen):,}  "
          f"duplicates={dups:,}  dup_rate={dup_rate:.4f}  "
          f"tokens={tokens/1e9:.2f}B (seq_len={seq_len})")
    if args.limit:
        print(f"(scanned a capped {total:,} blocks; rerun without --limit for a full check)")

    if dups:
        print("FAIL: duplicate blocks found "
              "(prepare_data token_buffer not trimmed? short dataset re-looped?)")
        for ex in dup_examples:
            print("  dup block prefix:", ex)
        sys.exit(1)
    print("OK: no duplicate blocks")


if __name__ == "__main__":
    main()
