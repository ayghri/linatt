"""Check tokenized shards for DUPLICATE blocks.

Catches the un-trimmed token_buffer bug in prepare_data.py (every flush
re-emitting earlier blocks -> heavy duplication -> memorization -> fake-low
loss). Hashes every block's full input_ids across all shards and reports the
duplicate rate. Exits non-zero if any duplicates are found.

Usage:
    python checks/check_dedup.py <save_dir>            # globs <save_dir>/shard_*
    python checks/check_dedup.py <save_dir> --limit 200000   # cap blocks (fast)
"""
import glob
import os
import sys

from datasets import load_from_disk


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save_dir = args[0] if args else "/buckets/datasets/pajama_shards"
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    shard_paths = sorted(glob.glob(os.path.join(save_dir, "shard_*")))
    if not shard_paths:
        print(f"ERROR: no shards at {save_dir}/shard_*")
        sys.exit(2)

    seen = set()
    total = dups = 0
    seq_len = None
    dup_examples = []

    for p in shard_paths:
        ds = load_from_disk(p)
        for batch in ds.iter(batch_size=2000):
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
                if limit and total >= limit:
                    break
            if limit and total >= limit:
                break
        if limit and total >= limit:
            break

    dup_rate = dups / total if total else 0.0
    tokens = total * (seq_len or 0)
    print(f"shards={len(shard_paths)}  blocks={total:,}  unique={len(seen):,}  "
          f"duplicates={dups:,}  dup_rate={dup_rate:.4f}  "
          f"tokens={tokens/1e9:.2f}B (seq_len={seq_len})")

    if dups:
        print("FAIL: duplicate blocks found "
              "(prepare_data token_buffer not trimmed? short dataset re-looped?)")
        for ex in dup_examples:
            print("  dup block prefix:", ex)
        sys.exit(1)
    print("OK: no duplicate blocks")


if __name__ == "__main__":
    main()
