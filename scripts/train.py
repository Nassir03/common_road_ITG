#!/usr/bin/env python3
"""Train either the proximity baseline or the proposed ITG GNN."""
from pathlib import Path
import argparse
import json
import random
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.config import SEED, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WINDOW_STRIDE
from model.gnn_dataset import DynamicITGDataset
from model.gnn_model import SimpleCommonRoadITGGNN
from model.metrics import ade_fde


def evaluate(model, dataset, device, max_samples=0):
    model.eval(); total_ade = total_fde = 0.0; count = 0
    limit = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    with torch.no_grad():
        for i in range(limit):
            sample = dataset[i]
            pred = model(sample)
            ade, fde = ade_fde(pred, sample["target"].to(device))
            total_ade += float(ade); total_fde += float(fde); count += 1
    return total_ade / max(count, 1), total_fde / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed"))
    p.add_argument("--edge-mode", choices=["itg", "radius"], default="itg")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--stride", type=int, default=WINDOW_STRIDE)
    p.add_argument("--output", default="")
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-val-samples", type=int, default=0, help="0 = all")
    args = p.parse_args()

    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    train_ds = DynamicITGDataset(root / "train", stride=args.stride, edge_mode=args.edge_mode)
    val_ds = DynamicITGDataset(root / "val", stride=args.stride, edge_mode=args.edge_mode)
    test_ds = DynamicITGDataset(root / "test", stride=args.stride, edge_mode=args.edge_mode)
    print(f"samples train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device}")

    model = SimpleCommonRoadITGGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output = Path(args.output or f"outputs/{args.edge_mode}_model.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); history = []

    train_indices = list(range(len(train_ds)))
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_indices)
        if args.max_train_samples > 0:
            epoch_indices = train_indices[:args.max_train_samples]
        else:
            epoch_indices = train_indices
        total = 0.0
        for i in epoch_indices:
            sample = train_ds[i]
            optimizer.zero_grad()
            pred = model(sample)
            loss, _ = ade_fde(pred, sample["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
        train_ade = total / max(len(epoch_indices), 1)
        val_ade, val_fde = evaluate(model, val_ds, device, args.max_val_samples)
        row = {"epoch": epoch, "train_ADE": train_ade, "val_ADE": val_ade, "val_FDE": val_fde}
        history.append(row)
        print(f"epoch={epoch:03d} train_ADE={train_ade:.4f} val_ADE={val_ade:.4f} val_FDE={val_fde:.4f}")
        if val_ade < best_val:
            best_val = val_ade
            torch.save({
                "model_state": model.state_dict(), "edge_mode": args.edge_mode,
                "stride": args.stride, "history": history,
            }, output)

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_ade, test_fde = evaluate(model, test_ds, device)
    result = {"edge_mode": args.edge_mode, "test_ADE_m": test_ade, "test_FDE_m": test_fde, "checkpoint": str(output)}
    result_path = output.with_suffix(".results.json")
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
