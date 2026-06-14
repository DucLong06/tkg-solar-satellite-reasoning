"""Runtime GPU detection + batch/precision autoscaling for Colab A100 (40/80 GB).

CPU is the unchanged default: every helper is a no-op off CUDA, so the fp32
training path stays bit-identical. On CUDA we read total VRAM and pick a
conservative (batch, grad_accum) preset plus a precision, leaving everything
overridable from config.

Keep small — pure policy, no training logic.
"""

from __future__ import annotations

import torch


def detect_vram_gb() -> float | None:
    """Total VRAM of the current CUDA device in GB, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.total_memory / (1024 ** 3)


def supports_bf16() -> bool:
    """True if the CUDA device supports bf16 (A100/Ampere+)."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def pick_batch(vram_gb: float | None, img_size: int) -> tuple[int, int]:
    """Map detected VRAM -> (physical_batch, grad_accum) for an effective batch of 64.

    Conservative presets (ViT-base satellite encoder over a k-step window is the
    memory driver; larger img_size -> smaller physical batch). grad_accum makes up
    the effective batch when the physical batch is small.
    """
    if vram_gb is None:
        return 64, 1
    big_img = img_size >= 224
    if vram_gb >= 70:          # ~80 GB A100
        phys = 32 if big_img else 96
    elif vram_gb >= 35:        # ~40 GB A100
        phys = 16 if big_img else 64
    elif vram_gb >= 14:        # T4/V100/L4 16 GB
        phys = 8 if big_img else 32
    else:
        phys = 4 if big_img else 16
    effective = 64
    grad_accum = max(1, effective // phys)
    return phys, grad_accum


def resolve_runtime(cfg) -> None:
    """Mutate cfg in place: on CUDA + auto_batch, set batch/grad_accum/precision.

    Called BEFORE DataPipeline.load so the chosen batch size reaches the loaders.
    On CPU this is a no-op (precision falls back to fp32).
    """
    if cfg.device != "cuda" or not torch.cuda.is_available():
        cfg.precision = "fp32"
        cfg.grad_accum_steps = 1
        return

    if getattr(cfg, "precision", "auto") in ("auto", None):
        cfg.precision = "bf16" if supports_bf16() else "fp16"

    if getattr(cfg, "auto_batch", False):
        vram = detect_vram_gb()
        batch, grad_accum = pick_batch(vram, cfg.img_size)
        cfg.batch_size = batch
        cfg.grad_accum_steps = grad_accum
        print(
            f"[gpu_autoscale] VRAM~{vram:.0f}GB -> batch={batch} "
            f"grad_accum={grad_accum} precision={cfg.precision}"
        )
