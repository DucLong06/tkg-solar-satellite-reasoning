"""Device-agnostic training loop — Adam, MSE, val early-stop, best checkpoint.

Model-agnostic: any module with ``forward(batch) -> [B, N_HORIZONS]`` works
(LSTM baseline or full TKGSolarModel). Loss is computed on scaled targets;
the val MAE used for early-stopping is inverse-scaled to original units.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_pipeline.scaling import Scalers
from src.metrics.regression_metrics import compute_all


def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _amp_setup(config):
    """Resolve (autocast_context_factory, grad_scaler) from config.precision.

    Off CUDA, or precision fp32, this returns a no-op nullcontext + no scaler so
    the training path is bit-identical to the original fp32 loop. bf16 needs no
    scaler (full range); fp16 uses a GradScaler.
    """
    precision = getattr(config, "precision", "fp32")
    on_cuda = config.device == "cuda" and torch.cuda.is_available()
    if not on_cuda or precision not in ("bf16", "fp16"):
        return (lambda: nullcontext()), None
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if precision == "fp16" else None
    return (lambda: torch.autocast(device_type="cuda", dtype=dtype)), scaler


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: str):
    """Return (y_true_scaled, y_pred_scaled) stacked over the loader."""
    model.eval()
    yts, yps = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        pred = model(batch)
        yps.append(pred.cpu())
        yts.append(batch["target"].cpu())
    return torch.cat(yts), torch.cat(yps)


def evaluate_mae(model, loader, scalers: Scalers, device: str, mape_min_value: float) -> dict:
    yt, yp = predict_loader(model, loader, device)
    return compute_all(scalers.inverse_pv(yt.numpy()), scalers.inverse_pv(yp.numpy()), mape_min_value)


def fit(
    model: nn.Module,
    splits,
    config,
    loss_fn=None,
    verbose: bool = True,
    desc: str | None = None,
    resume: bool = False,
) -> dict:
    device = config.device
    model.to(device)

    # GPU utilization knobs (all no-ops off CUDA -> fp32 path unchanged).
    on_cuda = device == "cuda" and torch.cuda.is_available()
    if on_cuda and getattr(config, "channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    if on_cuda and getattr(config, "compile", False):
        try:
            model = torch.compile(model)
        except Exception as e:  # dynamic per-timestep GAT may not compile -> eager
            tqdm.write(f"torch.compile failed ({e}); continuing eager")
    autocast_ctx, scaler = _amp_setup(config)
    accum = max(1, getattr(config, "grad_accum_steps", 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = loss_fn or nn.MSELoss()

    # Per-batch progress: one transient bar per epoch (leave=False -> it clears
    # after each epoch, so no nesting glitch and no pile-up across epochs), plus
    # one concise summary line per epoch via tqdm.write. Without an inner bar the
    # heavy full model gives zero feedback during a multi-minute epoch and looks
    # hung. verbose=False (benchmark loops) stays fully silent.
    label = desc or config.model

    # Checkpoints are keyed by label (the model name) so the 5 benchmark models
    # each get their own files instead of overwriting a single config.model name.
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_{label}.pt"
    last_path = ckpt_dir / f"last_{label}.pt"

    best_val = float("inf")
    patience = 0
    start_epoch = 0
    history = {"train_loss": [], "val_mae": []}

    # Resume from the last full-state checkpoint (model + optimizer + bookkeeping)
    # so an interrupted run (e.g. Colab quota cutoff) continues instead of
    # restarting from epoch 0. Only active when resume=True; the resume file is
    # rewritten every epoch below, so it must live on persistent storage (Drive).
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        for state in optimizer.state.values():  # optimizer tensors -> training device
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        patience = ck["patience"]
        history = ck["history"]
        if verbose:
            tqdm.write(f"{label}: resume from epoch {start_epoch} (best val_mae={best_val:.5f})")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running = 0.0
        n = 0
        loader = splits.train_loader
        n_batches = len(loader)
        if verbose:
            loader = tqdm(loader, desc=f"{label} ep {epoch + 1}/{config.epochs}", leave=False)

        # Gradient accumulation: step the optimizer every `accum` micro-batches
        # (accum=1 -> step every batch, identical to the original loop).
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = _to_device(batch, device)
            with autocast_ctx():
                pred = model(batch)
                loss = loss_fn(pred, batch["target"])
            scaled = loss / accum
            (scaler.scale(scaled) if scaler else scaled).backward()

            is_step = (i + 1) % accum == 0 or (i + 1) == n_batches
            if is_step:
                if config.grad_clip:
                    if scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * len(batch["target"])
            n += len(batch["target"])
            if verbose:
                loader.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(n, 1)
        val = evaluate_mae(model, splits.val_loader, splits.scalers, device, config.mape_min_value)
        history["train_loss"].append(train_loss)
        history["val_mae"].append(val["mae"])
        if verbose:
            tqdm.write(
                f"{label} ep {epoch + 1}/{config.epochs}: "
                f"train_loss={train_loss:.4f} val_mae={val['mae']:.4f}"
            )

        if val["mae"] < best_val - 1e-6:
            best_val = val["mae"]
            patience = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_mae": best_val}, best_path)
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                if verbose:
                    tqdm.write(f"{label}: early stop at epoch {epoch + 1} (best val_mae={best_val:.5f})")
                break

        # Rewrite the resume checkpoint after every completed epoch so a mid-run
        # interruption loses at most the current epoch, not the whole run. done=False
        # marks it as still-in-progress (a callers' skip/retrain prompt keys on this).
        if resume:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "patience": patience,
                    "history": history,
                    "done": False,
                },
                last_path,
            )

    history["best_val_mae"] = best_val
    history["best_checkpoint"] = str(best_path)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state"])

    # Training ran to completion (early-stop or max epochs). Stamp the resume file
    # done=True so a re-run can offer skip-vs-retrain instead of resuming.
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        ck["done"] = True
        torch.save(ck, last_path)

    return history
