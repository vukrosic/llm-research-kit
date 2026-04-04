from dataclasses import dataclass
from typing import Optional, Tuple
from configs.llm_config import LLMConfig


@dataclass
class TitanXConfig(LLMConfig):
    """
    Config tuned for NVIDIA TITAN X (Pascal), 12 GB VRAM, CC 6.1.

    Pascal has no Tensor Cores but supports BF16 via software emulation.
    torch.compile is disabled — CC 6.1 is not in the kernel precompilation
    list for modern PyTorch and the JIT fallback adds startup overhead with
    no throughput benefit on this arch.

    Batch size 16 keeps VRAM usage well under 12 GB for the 88M param model
    at seq_len 2048.  Gradient accumulation is left at 1 — effective batch
    size 16 is already reasonable for pretraining at this scale.
    """

    # Throughput
    compile_model: bool = False   # CC 6.1 not in precompiled kernels
    # Pascal (CC 6.1) lacks flash attention support — SDPA falls back to
    # the "math" kernel which materialises full [B, H, T, T] attention maps.
    # At seq_len=2048 each batch element uses ~128 MB/layer → keep batch=2.
    # Effective batch size = 2 * 4 = 8 via gradient accumulation.
    batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Mixed precision — Pascal has software BF16; use_amp=True still helps
    # because it reduces memory bandwidth pressure even without HW BF16.
    use_amp: bool = True

    # Keep architecture identical to the 88M baseline so results are
    # directly comparable across GPU configs.
