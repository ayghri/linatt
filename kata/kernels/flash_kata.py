import triton
import triton.language as tl


@triton.jit
def flash_kata_kernel(
    q_addr,
    k_addr,
    v_addr,
    o_addr,
    s_addr,
    z_addr,
    num_heads,
    seq_len,
    num_chunks,
    chunk_size: tl.constexpr,
    b_M: tl.constexpr,
    b_N: tl.constexpr,
    b_H: tl.constexpr,
):

    start_m = tl.program_id(0) * chunk_size
    off_hb = tl.program_id(1)
    off_b = off_hb // num_heads
    off_h = off_hb % num_heads

    q_ptr = tl.make_block_ptr(q_addr, (b_M, b_H), ())
