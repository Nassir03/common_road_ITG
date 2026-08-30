# CommonRoad-ITG + GNN — Complete Implementation

This project implements the supplied CommonRoad-ITG method.

## What is implemented

- vehicle and lanelet nodes;
- L2L predecessor, successor, adjacent, merging, diverging, and **conflicting** relations;
- dynamic **multiple** V2L assignments and reverse L2V edges;
- ROT -> ROC -> ROI -> FIFO BFS -> inward ITG;
- VOI-specific hop and branch labels (no ambiguous contextual duplicate labels);
- direct/indirect, relative position, relative velocity, normalized distance edge features;
- ITG rebuilt at every observation time step;
- causal VTV temporal edges and Time2Vec;
- MAX_HOPS V2V GNN layers;
- lane GRU, heterogeneous message passing, temporal GRU, trajectory decoder;
- proximity-only radius baseline with the same neural architecture;
- ADE and FDE evaluation;
- location-disjoint train/validation/test split.

## Supplied-data split

The bundled `data/raw/{train,val,test}` directories were built from the user's attached data

- `data/splits/summary.json`
- `data/splits/split_manifest.csv`
- `data/splits/excluded.csv`


## 1. Install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
python -m pip install --upgrade pip
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m pytest -q
```
```bash
python scripts/preprocess.py
```

## 3. Train the proximity baseline

```bash
python scripts/train.py --edge-mode radius --epochs 30 --output outputs/radius_model.pt 
```

## 4. Train the proposed ITG model

```bash
python scripts/train.py \
  --edge-mode itg \
  --epochs 30 \
  --output outputs/itg_model.pt
```

For a quick execution check:

```bash
python scripts/train.py --edge-mode itg --epochs 30 --max-train-samples 20 --max-val-samples 10 --output outputs/itg_smoke.pt
```

## 5. Evaluate held-out test locations

```bash
python scripts/evaluate.py --checkpoint outputs/radius_model.pt --output-csv outputs/radius_test.csv
python scripts/evaluate.py --checkpoint outputs/itg_model.pt --output-csv outputs/itg_test.csv
```
