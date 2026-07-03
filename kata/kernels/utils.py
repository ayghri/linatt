import triton

FWD_CONFIGS = [
    # 32x32
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=2),
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=3),
    # 32x64
    triton.Config({"BT": 32, "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32, "BS": 64}, num_warps=4, num_stages=2),
    # 64x32
    triton.Config({"BT": 64, "BS": 32}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64, "BS": 32}, num_warps=4, num_stages=2),
    # 64x64
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=2),
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=3),
    triton.Config({"BT": 64, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64, "BS": 64}, num_warps=8, num_stages=2),
    # 64x128
    triton.Config({"BT": 64, "BS": 128}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64, "BS": 128}, num_warps=8, num_stages=2),
    # 128x64
    triton.Config({"BT": 128, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 128, "BS": 64}, num_warps=8, num_stages=2),
    # 128x128
    triton.Config({"BT": 128, "BS": 128}, num_warps=8, num_stages=1),
    triton.Config({"BT": 128, "BS": 128}, num_warps=8, num_stages=2),
]

# Bwd kernels carry more live tiles per program (per-group dq/dk
# accumulators); SRAM-friendly configs first.
BWD_CONFIGS = [
    # 16x16 (most conservative)
    triton.Config({"BT": 16, "BS": 16}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32, "BS": 16}, num_warps=4, num_stages=1),
    triton.Config({"BT": 16, "BS": 32}, num_warps=4, num_stages=1),
    # 32x32
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=2),
    triton.Config({"BT": 32, "BS": 32}, num_warps=8, num_stages=1),
    # 32x64 / 64x32
    triton.Config({"BT": 32, "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64, "BS": 32}, num_warps=4, num_stages=1),
    # 64x64
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=2),
    triton.Config({"BT": 64, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64, "BS": 64}, num_warps=8, num_stages=2),
    # 128x64 / 64x128 (may OOR on smaller SRAM; Triton drops)
    triton.Config({"BT": 128, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64, "BS": 128}, num_warps=8, num_stages=1),
    triton.Config({"BT": 128, "BS": 128}, num_warps=8, num_stages=1),
]

# depends on head-dim and group count, not on T (per-program work is T-independent),
# so keying on T forces a full autotune sweep for every distinct sequence length --
# which makes eval/generation (variable/growing T) crawl, especially on A100 where
# each sweep is slower. Tune once per (K_d, V_d, M/E) and reuse across all lengths.
FWD_KEY = ["d_k", "d_v", "M"]
BWD_KEY_M1 = ["d_k", "d_v"]
BWD_KEY_ME = ["d_k", "d_v", "E"]
