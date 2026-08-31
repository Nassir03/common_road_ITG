# CommonRoad-Geometric 

## What is implemented from the paper

- Vehicle nodes and lanelet nodes.
- V2V, V2L, L2V, L2L relations.
- Temporal VTV relations for the same vehicle at different observed times.
- Default V2V construction using Voronoi/Delaunay connectivity.
- Center-based V2L assignment: a vehicle is connected to all lanelets containing its center.
- L2L relation types: predecessor, successor, adjacent-left, adjacent-right, merging, diverging, conflicting.
- Vehicle features from Table II: position, orientation, yaw-rate, velocity, acceleration, width, length.
- V2V features from Table II: distance, relative position, relative orientation, relative velocity, relative acceleration.
- V2L features from Table II: left/right boundary distance, lateral offset, heading error, projected arclength, normalized arclength.
- L2L features from Table II: distance, relative position, relative orientation, intersection arclengths, adjacency type.
- VTV features: V2V motion features plus elapsed time.
- Edge-enhanced heterogeneous graph transformer (HGT-style) encoder using both node and edge features.
- Time2Vec encoding of VTV elapsed time.
- GRU encoding of variable-length lanelet boundary waypoint sequences.
- Learnable L2L adjacency-type embedding.
- GRU decoder predicting local `dx`, `dy`, `dtheta`, recursively integrated into future states.
- ADE training objective; ADE and FDE evaluation.
- Five future predictions for a 1.0 s horizon at 0.2 s intervals.

## Kaggle datasets

The code automatically checks these paths:
```text
/kaggle/input/datasets/abdullge26z811/boston/boston_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/boston

/kaggle/input/datasets/abdullge26z811/pittsburgh/pittsburgh_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/pittsburgh

/kaggle/input/datasets/abdullge26z811/singapore/singapore_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/singapore
```

## Kaggle: run one city from start to finish

```python
CITY = "boston"
# later: "pittsburgh"
# later: "singapore"
```

### 1. Check GPU

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### 2. Clone the repository

```python
%cd /kaggle/working
!rm -rf common_road_ITG
!git clone https://github.com/Nassir03/common_road_ITG.git
%cd /kaggle/working/common_road_ITG
```

### 3. Install only the extra Kaggle requirements

Do not replace Kaggle's CUDA-enabled PyTorch build.

```python
!python -m pip install -q -r requirements-kaggle.txt
```

### 4. Run tests

```python
!python -m pytest -q
```

Expected for this package:

```text
7 passed
```

### 5. Inspect the selected Kaggle dataset

```python
!python scripts/inspect_kaggle_data.py --city {CITY}
```

### 6. Small preprocessing test first

This checks the XML format without processing the whole city:

```python
!python scripts/prepare_city.py --city {CITY} --max-scenarios 20 --workers 4 --overwrite
```

Then validate the resulting graphs:

```python
!python scripts/validate_city.py --city {CITY} --samples 4
```

If validation succeeds, prepare the full city:

```python
!python scripts/prepare_city.py --city {CITY} --workers 4 --overwrite
```

Do not run the small preprocessing command again after the full command, because `--overwrite` replaces the processed folder.

### 7. Validate real processed data before training

```python
!python scripts/validate_city.py --city {CITY} --samples 8
```

This checks that graph tensors, targets, and model predictions are finite.

### 8. Training smoke test

```python
!python scripts/train.py \
    --city {CITY} \
    --epochs 1 \
    --max-train-samples 32 \
    --max-val-samples 16 \
    --max-test-samples 16 \
    --batch-size 4 \
    --workers 2 \
    --patience 0 \
    --output outputs/{CITY}_crgeo_smoke.pt
```

The output must contain finite `train_ADE`, `val_ADE`, `val_FDE`, and a saved checkpoint.

### 9. Full training

```python
!python scripts/train.py \
    --city {CITY} \
    --epochs 20 \
    --batch-size 4 \
    --workers 2 \
    --output outputs/{CITY}_crgeo_paper.pt
```

Early stopping is enabled by default.

### 10. Evaluate the held-out test split

```python
!python scripts/evaluate.py \
    --city {CITY} \
    --checkpoint outputs/{CITY}_crgeo_paper.pt \
    --output-csv outputs/{CITY}_crgeo_test.csv
```

The reported metrics are mean ADE and mean FDE in metres.

### 11. Inspect one prediction

```python
!python scripts/predict.py \
    --city {CITY} \
    --checkpoint outputs/{CITY}_crgeo_paper.pt \
    --index 0
```

### 12. Save Kaggle outputs

```python
!zip -r /kaggle/working/{CITY}_crgeo_paper_results.zip \
    outputs \
    data/{CITY}/summary.json \
    data/{CITY}/split_manifest.csv
```

After Boston completes, change only:

```python
CITY = "pittsburgh"
```

and rerun steps 5-12. Then do the same for Singapore.

## Paper reference metrics

The paper reports:

| City | Scenarios | ADE (m) | FDE (m) |
|---|---:|---:|---:|
| Singapore | 2372 | 0.106 | 0.227 |
| Boston | 938 | 0.138 | 0.316 |
| Pittsburgh | 1560 | 0.215 | 0.454 |

Those are published reference values, not guaranteed outputs of this independent reimplementation. The paper does not expose every experiment setting, and the Kaggle copy may contain a different number of scenarios (the user's prior Boston run showed 926 XML files rather than 938).

## Project layout

```text
model/
  config.py          paper-aligned constants and explicit implementation choices
  scenario.py        CommonRoad XML + lanelet/vehicle preprocessing
  geometry.py        Table-II graph feature calculations + Voronoi/Delaunay edges
  indexing.py        temporal window index
  gnn_dataset.py     heterogeneous temporal graph construction
  batching.py        block-diagonal graph batching
  gnn_model.py       edge-enhanced HGT + Time2Vec + lane GRU + trajectory GRU decoder
  metrics.py         ADE/FDE

scripts/
  inspect_kaggle_data.py
  prepare_city.py
  validate_city.py
  train.py
  evaluate.py
  predict.py
```
