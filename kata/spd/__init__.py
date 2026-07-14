"""KATA-SPD attention kernels.

Only the kernels actually used by the models / paper are imported here, so this
file doubles as the manifest of what's live.

  flash_delta        (kata.spd.flash_delta)      -- DeltaKATA: O(T^2) scalar-kernel delta, no E^2 state
  flash_delta_state  (kata.spd.chunk_state_v2)   -- KATA-SSM (delta): O(T) HBM-state SPD delta
  spd_state_lin      (kata.spd.chunk_state_lin)  -- KATA-SSM (linear): O(T) 3-chain chunked-state,
                                                    fwd + O(T) 2-pass backward (differentiable); the
                                                    fast paper-benchmark kernel (1.4-1.6x flash_delta_state)
  spd_chunk_scan     (kata.spd.chunk_scan)       -- sum-mode parallel-prefix tree-scan
  spd_chunk_scan_cat (kata.spd.chunk_scan_cat)   -- concat/linear-feature tree-scan
"""
from .flash_delta import flash_delta
from .chunk_state_v2 import flash_delta_state
from .chunk_state_lin import spd_state_lin
from .chunk_scan import spd_chunk_scan
from .chunk_scan_cat import spd_chunk_scan_cat

__all__ = [
    "flash_delta",
    "flash_delta_state",
    "spd_state_lin",
    "spd_chunk_scan",
    "spd_chunk_scan_cat",
]
