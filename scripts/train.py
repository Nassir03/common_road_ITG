#!/usr/bin/env python3
"""Train the paper-aligned cr-geo HGT + GRU trajectory model.

Training objective: ADE, as stated in Meyer et al. (2023).
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import (
    DECODER_HIDDEN_DIM, EPOCHS, GRAD_CLIP_NORM, HGT_HEADS, HGT_LAYERS,
    HIDDEN_DIM, LEARNING_RATE, OBS_STEPS, PRED_STEPS, SEED, WEIGHT_DECAY,
    WINDOW_STRIDE,
)
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import trajectory_ade_loss, trajectory_error_sums


def evaluate(model, dataset, device, max_samples: int = 0) -> tuple[float, float]:
    model.eval()
    ade_sum = fde_sum = 0.0
    ade_count = fde_count = 0
    limit = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    with torch.no_grad():
        for i in range(limit):
            sample = dataset[i]
            output = model(sample)
            a_sum, f_sum, a_count, f_count = trajectory_error_sums(
                output["position"],
                sample["target_position"].to(device),
                sample["target_mask"].to(device),
            )
            ade_sum += float(a_sum)
            fde_sum += float(f_sum)
            ade_count += a_count
            fde_count += f_count
    return ade_sum / max(ade_count, 1), fde_sum / max(fde_count, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed"))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--stride", type=int, default=WINDOW_STRIDE)
    p.add_argument("--output", default="outputs/crgeo_paper_model.pt")
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-val-samples", type=int, default=0, help="0 = all")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--heads", type=int, default=HGT_HEADS)
    p.add_argument("--hgt-layers", type=int, default=HGT_LAYERS)
    p.add_argument("--decoder-hidden-dim", type=int, default=DECODER_HIDDEN_DIM)
    args = p.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    train_ds = CommonRoadTemporalGraphDataset(data_root / "train", stride=args.stride)
    val_ds = CommonRoadTemporalGraphDataset(data_root / "val", stride=args.stride)
    print(f"samples train={len(train_ds)} val={len(val_ds)} device={device}")
    if train_ds.skipped_scenarios:
        print(f"warning: {len(train_ds.skipped_scenarios)} training scenarios skipped because MODEL_DT=0.2 s cannot be sampled exactly")

    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "hgt_layers": args.hgt_layers,
        "pred_steps": PRED_STEPS,
        "decoder_hidden_dim": args.decoder_hidden_dim,
    }
    model = CrGeoTrajectoryPredictionModel(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[dict] = []
    train_indices = list(range(len(train_ds)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_indices)
        epoch_indices = train_indices[: args.max_train_samples] if args.max_train_samples > 0 else train_indices
        loss_sum = 0.0

        for i in epoch_indices:
            sample = train_ds[i]
            optimizer.zero_grad(set_to_none=True)
            output_dict = model(sample)
            loss = trajectory_ade_loss(
                output_dict["position"],
                sample["target_position"].to(device),
                sample["target_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            loss_sum += float(loss.detach())

        train_ade = loss_sum / max(len(epoch_indices), 1)
        val_ade, val_fde = evaluate(model, val_ds, device, args.max_val_samples)
        row = {
            "epoch": epoch,
            "train_ADE_m": train_ade,
            "val_ADE_m": val_ade,
            "val_FDE_m": val_fde,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_ADE={train_ade:.4f}m "
            f"val_ADE={val_ade:.4f}m val_FDE={val_fde:.4f}m"
        )

        if val_ade < best_val:
            best_val = val_ade
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "dataset": {
                        "obs_steps": OBS_STEPS,
                        "pred_steps": PRED_STEPS,
                        "stride": args.stride,
                        "model_dt": 0.2,
                    },
                    "history": history,
                    "paper_alignment": "Meyer et al. 2023 cr-geo trajectory prediction",
                },
                output,
            )

    print(f"saved best checkpoint: {output}")
    print(json.dumps({"best_val_ADE_m": best_val}, indent=2))


if __name__ == "__main__":
    main()
