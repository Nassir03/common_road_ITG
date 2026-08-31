#!/usr/bin/env python
"""Train the paper-only CommonRoad-Geometric trajectory-prediction model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model.batching import batch_graph_samples
from model.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_CLIP_NORM,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    EARLY_STOPPING_PATIENCE,
)
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_loss, ade_fde


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, indices, batch_size: int, workers: int, device: torch.device):
    subset = Subset(dataset, indices)
    kwargs = dict(
        dataset=subset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, workers),
        collate_fn=batch_graph_samples,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _finite_or_raise(name: str, tensor: torch.Tensor, metadata) -> None:
    if torch.isfinite(tensor).all():
        return
    bad = int((~torch.isfinite(tensor)).sum().detach().cpu())
    ids = [m.get("benchmark_id") for m in metadata] if isinstance(metadata, list) else [str(metadata)]
    raise FloatingPointError(f"Non-finite {name}: {bad} values. Samples={ids[:8]}")


def run_dataset(
    model,
    dataset,
    indices,
    device,
    optimizer=None,
    batch_size: int = BATCH_SIZE,
    workers: int = NUM_WORKERS,
    grad_clip: float = GRAD_CLIP_NORM,
):
    training = optimizer is not None
    model.train(training)
    total_ade = 0.0
    total_fde = 0.0
    n_targets = 0
    n_batches = 0
    start = time.time()
    loader = _loader(dataset, indices, batch_size, workers, device)
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for step, batch in enumerate(loader, 1):
            target = batch["target_position"].to(device, non_blocking=True)
            mask = batch["target_mask"].to(device, non_blocking=True)
            if not bool(mask.any()):
                continue
            _finite_or_raise("target_position", target, batch["meta"])

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(batch)["position"]
            _finite_or_raise("prediction", prediction, batch["meta"])
            loss = ade_loss(prediction, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite ADE loss. Samples={[m.get('benchmark_id') for m in batch['meta']][:8]}"
                )

            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(torch.as_tensor(gradient_norm)):
                    raise FloatingPointError("Non-finite gradient norm")
                optimizer.step()

            metrics = ade_fde(prediction.detach(), target, mask)
            total_ade += metrics["ade"] * metrics["count"]
            total_fde += metrics["fde"] * metrics["count"]
            n_targets += metrics["count"]
            n_batches += 1

            if step % 50 == 0 or step == len(loader):
                print(
                    f"    batch {step}/{len(loader)} targets={n_targets} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )

    if n_targets == 0:
        return {
            "ade": float("nan"),
            "fde": float("nan"),
            "batches": 0,
            "targets": 0,
            "seconds": time.time() - start,
        }
    return {
        "ade": total_ade / n_targets,
        "fde": total_fde / n_targets,
        "batches": n_batches,
        "targets": n_targets,
        "seconds": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["boston", "pittsburgh", "singapore"])
    parser.add_argument("--data-root", default=None, help="Default: data/<city>/processed")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.city / "processed"
    output = Path(args.output) if args.output else ROOT / "outputs" / f"{args.city}_crgeo_paper.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never reuse a stale checkpoint from an earlier failed run.
    if output.exists():
        output.unlink()
    stale_results = output.with_suffix(".results.json")
    if stale_results.exists():
        stale_results.unlink()

    index_start = time.time()
    train_dataset = CommonRoadTemporalGraphDataset(data_root / "train")
    val_dataset = CommonRoadTemporalGraphDataset(data_root / "val")
    test_dataset = CommonRoadTemporalGraphDataset(data_root / "test")
    print(f"dataset index load={time.time() - index_start:.2f}s")
    print(f"city={args.city}")
    print(
        f"scenario_files train={len(train_dataset.files)} val={len(val_dataset.files)} test={len(test_dataset.files)}"
    )
    print(
        f"samples train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        raise RuntimeError("One split has zero valid temporal trajectory samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    model = CrGeoTrajectoryPredictionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_indices = train_dataset.epoch_indices(args.seed + epoch, args.max_train_samples)
        val_indices = list(range(len(val_dataset)))
        if args.max_val_samples > 0:
            val_indices = val_indices[: args.max_val_samples]

        train_metrics = run_dataset(
            model,
            train_dataset,
            train_indices,
            device,
            optimizer,
            args.batch_size,
            args.workers,
        )
        val_metrics = run_dataset(
            model,
            val_dataset,
            val_indices,
            device,
            None,
            args.batch_size,
            args.workers,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch={epoch:03d} "
            f"train_ADE={train_metrics['ade']:.6f} "
            f"val_ADE={val_metrics['ade']:.6f} "
            f"val_FDE={val_metrics['fde']:.6f} "
            f"train_s={train_metrics['seconds']:.1f} val_s={val_metrics['seconds']:.1f}",
            flush=True,
        )

        if math.isfinite(val_metrics["ade"]) and val_metrics["ade"] < best_val:
            best_val = val_metrics["ade"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": {},
                    "city": args.city,
                    "best_epoch": best_epoch,
                    "best_val_ade": best_val,
                    "data_root": str(data_root),
                    "seed": args.seed,
                    "method": "CommonRoad-Geometric paper-only Voronoi/Delaunay + causal VTV + edge-enhanced HGT",
                },
                output,
            )
            print(f"  saved best checkpoint -> {output}")
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(
                f"early stopping after {bad_epochs} epochs without validation improvement"
            )
            break

    if not output.exists():
        raise RuntimeError(
            "No checkpoint was saved because validation ADE was never finite. "
            "The run stops here with the real cause instead of a later FileNotFoundError."
        )

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_indices = list(range(len(test_dataset)))
    if args.max_test_samples > 0:
        test_indices = test_indices[: args.max_test_samples]
    test_metrics = run_dataset(
        model,
        test_dataset,
        test_indices,
        device,
        None,
        args.batch_size,
        args.workers,
    )

    result = {
        "city": args.city,
        "checkpoint": str(output),
        "best_epoch": best_epoch,
        "best_val_ADE_m": best_val,
        "test_ADE_m": test_metrics["ade"],
        "test_FDE_m": test_metrics["fde"],
        "test_targets": test_metrics["targets"],
        "history": history,
    }
    result_path = output.with_suffix(".results.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFINAL TEST")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
