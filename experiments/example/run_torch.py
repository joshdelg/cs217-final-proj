"""
PyTorch elementwise-add kernel for Trainium.
Materialize inputs first, then profile just the add.
"""
import os
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

import torch
from torch_xla.core import xla_model as xm


def main():
    device = xm.xla_device()

    # Materialize inputs (randn compiles + executes in its own NEFF)
    a = torch.randn(256, 1024, device=device, dtype=torch.float32)
    b = torch.randn(256, 1024, device=device, dtype=torch.float32)
    xm.mark_step()
    xm.wait_device_ops()

    # Only the add goes into the profiled NEFF
    out = a + b
    xm.mark_step()
    xm.wait_device_ops()
    print("torch output sum:", out.cpu().sum().item())


if __name__ == "__main__":
    main()
