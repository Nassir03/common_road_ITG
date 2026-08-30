#!/usr/bin/env python
"""Train either the paper baseline (Voronoi/Delaunay V2V) or the ITG extension."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model.config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP_NORM, SEED
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_loss, ade_fde


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_to_device_target(sample, device):
    return sample["target_position"].to(device), sample["target_mask"].to(device)


def run_dataset(model, dataset, indices, device, optimizer=None, grad_clip=GRAD_CLIP_NORM):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0; total_ade = 0.0; total_fde = 0.0; n_samples = 0; n_targets = 0
    start = time.time()
    for step, idx in enumerate(indices, 1):
        sample = dataset[idx]
        target, mask = sample_to_device_target(sample, device)
        if not bool(mask.any()):
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred = model(sample)["position"]
            loss = ade_loss(pred, target, mask)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        metrics = ade_fde(pred.detach(), target, mask)
        total_loss += float(loss.detach().cpu()); total_ade += metrics["ade"] * metrics["count"]; total_fde += metrics["fde"] * metrics["count"]
        n_samples += 1; n_targets += metrics["count"]
        if step % 250 == 0:
            elapsed = time.time() - start
            print(f"    step {step}/{len(indices)} ({elapsed:.1f}s)", flush=True)
    if n_samples == 0 or n_targets == 0:
        return {"loss": float("nan"), "ade": float("nan"), "fde": float("nan"), "samples": 0, "targets": 0}
    return {"loss": total_loss/n_samples, "ade": total_ade/n_targets, "fde": total_fde/n_targets, "samples": n_samples, "targets": n_targets}


def evaluate_split(model, dataset, device, max_samples=0):
    indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]
    with torch.no_grad():
        return run_dataset(model, dataset, indices, device, optimizer=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["boston", "pittsburgh", "singapore"])
    ap.add_argument("--v2v-mode", default="paper", choices=["paper", "itg"], help="paper=Delaunay/Voronoi baseline; itg=ROT->ROC->ROI->BFS->ITG")
    ap.add_argument("--data-root", default=None, help="Default data/<city>/processed")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--max-test-samples", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.city / "processed"
    output = Path(args.output) if args.output else ROOT / "outputs" / f"{args.city}_{args.v2v_mode}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    train_ds = CommonRoadTemporalGraphDataset(data_root / "train", v2v_mode=args.v2v_mode)
    val_ds = CommonRoadTemporalGraphDataset(data_root / "val", v2v_mode=args.v2v_mode)
    test_ds = CommonRoadTemporalGraphDataset(data_root / "test", v2v_mode=args.v2v_mode)
    print(f"city={args.city} mode={args.v2v_mode}")
    print(f"scenario_files train={len(train_ds.files)} val={len(val_ds.files)} test={len(test_ds.files)}")
    print(f"samples train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("One split has zero valid 15-observation + 5-future samples. Check preprocessing/dt and dataset content.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    model = CrGeoTrajectoryPredictionModel(v2v_edge_dim=train_ds.v2v_edge_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val = float("inf"); best_epoch = 0; history=[]
    for epoch in range(1, args.epochs+1):
        train_idx = train_ds.epoch_indices(args.seed + epoch, args.max_train_samples)
        val_idx = list(range(len(val_ds)))
        if args.max_val_samples > 0: val_idx = val_idx[:args.max_val_samples]
        train_m = run_dataset(model, train_ds, train_idx, device, optimizer=optimizer)
        with torch.no_grad():
            val_m = run_dataset(model, val_ds, val_idx, device, optimizer=None)
        row = {"epoch": epoch, "train": train_m, "val": val_m}; history.append(row)
        print(f"epoch={epoch:03d} train_ADE={train_m['ade']:.6f} val_ADE={val_m['ade']:.6f} val_FDE={val_m['fde']:.6f}", flush=True)
        if val_m["ade"] < best_val:
            best_val = val_m["ade"]; best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kwargs": {"v2v_edge_dim": train_ds.v2v_edge_dim},
                "city": args.city, "v2v_mode": args.v2v_mode,
                "best_epoch": best_epoch, "best_val_ade": best_val,
                "data_root": str(data_root), "seed": args.seed,
            }, output)
            print(f"  saved best checkpoint -> {output}")

    ckpt = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    with torch.no_grad():
        test_m = evaluate_split(model, test_ds, device, args.max_test_samples)
    result = {
        "city": args.city,
        "v2v_mode": args.v2v_mode,
        "checkpoint": str(output),
        "best_epoch": best_epoch,
        "best_val_ade_m": best_val,
        "test_ADE_m": test_m["ade"],
        "test_FDE_m": test_m["fde"],
        "test_samples": test_m["samples"],
        "test_targets": test_m["targets"],
        "history": history,
    }
    result_path = output.with_suffix(".results.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFINAL TEST")
    print(json.dumps({k:v for k,v in result.items() if k != "history"}, indent=2))
    print(f"results={result_path}")

if __name__ == "__main__":
    main()
