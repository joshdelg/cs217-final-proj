"""
NKI elementwise-add kernel for Trainium.
Based on: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/framework_custom_op.html

Materialize inputs first, then profile just the NKI add kernel.
"""
import os
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
os.environ.setdefault("XLA_IR_DEBUG", "1")
os.environ.setdefault("XLA_HLO_DEBUG", "1")

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
except ImportError:
    raise ImportError("run_nki.py requires neuronxcc (e.g. from /opt/aws_neuron_... venv)")

from torch_xla.core import xla_model as xm


TILE_M = 128
TILE_N = 512


@nki.jit
def nki_tensor_add(a_input, b_input):
    """NKI kernel: element-wise add. Tiled with affine_range loops; no SPMD."""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    M, N = a_input.shape

    assert a_input.shape == b_input.shape, "Expected shapes to match"
    assert a_input.dtype == b_input.dtype, "Expected dtypes to match"
    assert M % TILE_M == 0, f"Partition dim {M} must be divisible by {TILE_M}"
    assert N % TILE_N == 0, f"Partition dim {N} must be divisible by {TILE_N}"

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            a_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=a_input.dtype, buffer=nl.sbuf)
            b_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=b_input.dtype, buffer=nl.sbuf)

            nisa.dma_copy(
                dst=a_tile,
                src=a_input[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
            )
            nisa.dma_copy(
                dst=b_tile,
                src=b_input[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
            )

            c_tile = nisa.tensor_tensor(a_tile, b_tile, nl.add)

            nisa.dma_copy(
                dst=c_output[m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N],
                src=c_tile,
            )

    return c_output


def main():
    device = xm.xla_device()

    # Materialize inputs (randn compiles + executes in its own NEFF)
    a = torch.randn(256, 1024, device=device, dtype=torch.float32)
    b = torch.randn(256, 1024, device=device, dtype=torch.float32)
    xm.mark_step()
    xm.wait_device_ops()

    # Only the NKI add goes into the profiled NEFF
    out = nki_tensor_add(a, b)
    xm.mark_step()
    xm.wait_device_ops()
    print("nki output sum:", out.cpu().sum().item())


if __name__ == "__main__":
    main()
