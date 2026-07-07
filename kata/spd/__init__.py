"""KATA-SPD attention: flash (O(T^2)) and chunked (O(T*E^2)) kernels.

SPD score (q,k split into M groups of width E = d_head/M):
  concat: A[t,s] = sum_i  (q_i . k_i)^2
  sum:    A[t,s] = sum_ij (q_i . k_j)^2
sum-normalized (no softmax). Two compute strategies, cleanly separated:

  flash (kata.spd.flash)  - O(T^2) parallel, trainable (fwd+bwd). Best at the
                            training/short-context regime.
  chunk (kata.spd.chunk)  - O(T*E^2) stateful, bf16 tensor-core. Wins at long
                            context (T >> chunk size); forward only for now.

Public API:
  spd_flash(q,k,v,M,mode,scale)          -> o            (differentiable)
  spd_flash_fwd(q,k,v,M,mode,scale,...)  -> (o, den)     (forward only)
  spd_chunk(q,k,v,M,mode,scale,C)        -> (o, den)     (bf16 chunked fwd)
  spd_chunk_fp32(...)                    -> (o, den)     (fp32 chunked fwd)
  references: spd_parallel_ref, spd_recurrent_ref, spd_chunked_ref,
              psi_concat, psi_sum, spd_scores
"""
from .reference import (
    psi_concat,
    psi_sum,
    spd_scores,
    spd_parallel_ref,
    spd_recurrent_ref,
    spd_chunked_ref,
)
from .flash import spd_attn_parallel as spd_flash
from .flash import spd_attn_parallel_fwd as spd_flash_fwd
from .chunk import spd_attn_chunked_fast as spd_chunk
from .chunk_fp32 import spd_attn_chunked_fwd as spd_chunk_fp32
from .chunk_scan import spd_chunk_scan as spd_chunk_tree

__all__ = [
    "spd_flash",
    "spd_flash_fwd",
    "spd_chunk",
    "spd_chunk_fp32",
    "spd_chunk_tree",
    "psi_concat",
    "psi_sum",
    "spd_scores",
    "spd_parallel_ref",
    "spd_recurrent_ref",
    "spd_chunked_ref",
]
