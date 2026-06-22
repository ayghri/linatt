#!/usr/bin/env bash
# Per-rank wrapper: pin EVERY rank to the same physical GPU 0 and rewrite
# LOCAL_RANK->0 so train_llm.init_dist()'s `set_device_index(LOCAL_RANK)`
# resolves to the single visible device. For same-GPU multi-rank smoke tests.
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
exec "$@"
