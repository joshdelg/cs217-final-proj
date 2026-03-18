"""
Minimal PyTorch-only kernel example for Trainium.
Run with NEURON_FRAMEWORK_DEBUG=1 so the compiler saves the NEFF.
Same operation as run_nki.py (elementwise add); same shapes for fair comparison.
"""
import os
os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")

import torch
import torch.nn as nn
from torch_xla.core import xla_model as xm


class TorchAddModule(nn.Module):
    """Elementwise add: result[i] = a[i] + b[i] (pure PyTorch)."""

    def forward(self, a, b):
        return a + b


def main():
    # Same shape as run_nki.py for comparison: (256, 1024) per framework_custom_op tutorial
    device = xm.xla_device()
    mod = TorchAddModule().to(device)
    a = torch.randn(256, 1024, device=device, dtype=torch.float32)
    b = torch.randn(256, 1024, device=device, dtype=torch.float32)
    out = mod(a, b)
    xm.mark_step()
    print("torch output shape:", out.shape)


if __name__ == "__main__":
    main()
