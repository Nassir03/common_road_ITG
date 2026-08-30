# CommonRoad-ITG + Paper-Aligned cr-geo Trajectory Prediction

This is a Kaggle-ready implementation for the three CommonRoad-NuPlan city archives used by Meyer et al. (2023): Boston, Pittsburgh, and Singapore.

It supports two **fairly comparable V2V modes using the same heterogeneous temporal GNN**:

- `paper`: Delaunay/Voronoi-style V2V graph (the paper's default cr-geo V2V drawer)
- `itg`: `ROT -> ROC -> ROI -> FIFO BFS -> inward ITG`, while keeping the other graph types/model unchanged

## Important scientific scope

The 8-page paper states the graph types/features, HGT-style trajectory encoder, Time2Vec, lanelet GRU, GRU decoder, 1.0 s prediction horizon at 0.2 s, ADE training and ADE/FDE evaluation. It does **not** publish every training/splitting/hyperparameter detail. Therefore this project is a reproducible, paper-aligned implementation; it must not be described as a bit-for-bit reproduction of the authors' unpublished experiment setup.

The public cr-geo trajectory-prediction config is used for several implementation details that the paper does not enumerate (15 observed steps, 5 predicted steps, 8 graph layers, 16 heads, hidden size 256, Time2Vec 16, GRU decoder hidden 512, non-overlapping 20-step temporal collections).

## Kaggle dataset paths

The code automatically checks these paths:

```text
Boston:
/kaggle/input/datasets/abdullge26z811/boston/boston_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/boston

Pittsburgh:
/kaggle/input/datasets/abdullge26z811/pittsburgh/pittsburgh_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/pittsburgh

Singapore:
/kaggle/input/datasets/abdullge26z811/singapore/singapore_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/singapore
```

You choose one city and finish it before changing only the `CITY` variable for the next city.

---

# Exact Kaggle workflow

## 1. Kaggle settings

Create a notebook and enable:

- Accelerator: GPU
- Internet: On (needed only to clone GitHub)

## 2. Clone the repository

```python
!git clone https://github.com/Nassir03/common_road_ITG.git
%cd /kaggle/working/common_road_ITG
```

**The GitHub repository must contain the corrected files from this package first.** The public repository snapshot checked on 2026-08-30 has a stale README that still uses `--edge-mode`, while its current `train.py` does not accept that argument and its current `gnn_dataset.py` contains only the paper V2V graph. Push this corrected package before using the commands below.

## 3. Choose only one city

```python
CITY = "boston"   # later change only to "pittsburgh" or "singapore"
print(CITY)
```

## 4. Verify Kaggle data and GPU

```python
from pathlib import Path
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

root = Path(f"/kaggle/input/datasets/abdullge26z811/{CITY}")
files = list(root.rglob("*.xml"))
print("dataset root:", root)
print("XML files:", len(files))
print("first file:", files[0] if files else "NONE")
```

Paper scenario counts are:

- Boston: 938
- Pittsburgh: 1,560
- Singapore: 2,372

If your Kaggle upload contains a different count, the code uses the files actually present and prints a warning.

## 5. Install only missing lightweight dependencies

Kaggle already ships a CUDA-compatible PyTorch build. Do not replace it unless required.

```python
!python -m pip install -q -r requirements-kaggle.txt
```

## 6. Tests

```python
!python -m pytest -q
```

Expected for this package:

```text
5 passed
```

## 7. Prepare the chosen city

```python
!python scripts/prepare_city.py --city {CITY} --overwrite
```

This creates:

```text
data/<CITY>/processed/train/*.scenario.pt
data/<CITY>/processed/val/*.scenario.pt
data/<CITY>/processed/test/*.scenario.pt
data/<CITY>/split_manifest.csv
data/<CITY>/summary.json
```

The split is deterministic at the **scenario level**: 70% train, 15% validation, 15% test. This ratio is an explicit reproducibility choice because the paper does not publish an exact train/validation/test ratio.

## 8. Smoke test the paper baseline

```python
!python scripts/train.py --city {CITY} --v2v-mode paper --epochs 1 --max-train-samples 2 --max-val-samples 1 --max-test-samples 1 --output outputs/{CITY}_paper_smoke.pt
```

## 9. Smoke test the ITG extension

```python
!python scripts/train.py --city {CITY} --v2v-mode itg --epochs 1 --max-train-samples 2 --max-val-samples 1 --max-test-samples 1 --output outputs/{CITY}_itg_smoke.pt
```

If both complete and print `device=cuda`, start the full experiments.

## 10. Train paper baseline

```python
!python scripts/train.py --city {CITY} --v2v-mode paper --epochs 30 --output outputs/{CITY}_paper.pt
```

## 11. Train ITG extension

```python
!python scripts/train.py --city {CITY} --v2v-mode itg --epochs 30 --output outputs/{CITY}_itg.pt
```

## 12. Evaluate both on the held-out test split

```python
!python scripts/evaluate.py --city {CITY} --checkpoint outputs/{CITY}_paper.pt --output-csv outputs/{CITY}_paper_test.csv
!python scripts/evaluate.py --city {CITY} --checkpoint outputs/{CITY}_itg.pt --output-csv outputs/{CITY}_itg_test.csv
```

## 13. Compare

```python
!python scripts/compare_results.py --paper outputs/{CITY}_paper.results.json --itg outputs/{CITY}_itg.results.json
```

## 14. Inspect a prediction

```python
!python scripts/predict.py --city {CITY} --checkpoint outputs/{CITY}_itg.pt --index 0
```

## 15. Save outputs from Kaggle

```python
!zip -r /kaggle/working/{CITY}_commonroad_itg_results.zip outputs data/{CITY}/summary.json data/{CITY}/split_manifest.csv
```

Download the zip from Kaggle's Output panel.

---

# Changing to another dataset

Once Boston is completely finished, change **only**:

```python
CITY = "pittsburgh"
```

and rerun Steps 4-15.

After Pittsburgh is complete, change only:

```python
CITY = "singapore"
```

and rerun Steps 4-15.

The code resolves the correct Kaggle folder automatically.

---

# Data -> graph -> GNN pipeline

For every 20-step non-overlapping temporal window:

```text
15 observed states @ 0.2 s + 5 future states @ 0.2 s
                        |
                        v
        vehicle nodes + lanelet nodes
                        |
       +----------------+----------------+
       |                |                |
      V2L/L2V          L2L              VTV
       |                |                |
       +----------------+----------------+
                        |
                  V2V graph mode
                 /              \
        paper baseline          ITG extension
       Delaunay/Voronoi    ROT->ROC->ROI->BFS->ITG
                 \              /
                  edge-enhanced HGT
                        |
                  Time2Vec on VTV
                        |
                 lanelet GRU encoder
                        |
                    GRU decoder
                        |
           5 future (x,y,orientation) states
                        |
                      ADE/FDE
```

## Paper V2V edge features (8)

For source vehicle `i -> j`:

1. Euclidean distance
2. source-local relative x
3. source-local relative y
4. relative orientation
5. source-local relative velocity x
6. source-local relative velocity y
7. source-local relative acceleration x
8. source-local relative acceleration y

## ITG V2V edge features (12)

The ITG keeps those same 8 physical V2V features and adds:

9. normalized BFS hop = `hop / MAX_HOPS`
10. normalized branch = `branch / branch_count`
11. direct/indirect = `1` for hop 1, else `0`
12. normalized ITG distance = `min(distance / ROI_RADIUS, 1)`

Thus the experimental comparison changes the V2V topology and adds the information required to describe the hop-aware ITG, while V2L, L2V, L2L, VTV, HGT and decoder remain common.

## ITG construction

At each observed time step:

1. **ROT** = all vehicles in the current traffic snapshot.
2. **ROC** = undirected communication graph with pairwise distance <= `COMMUNICATION_RADIUS`.
3. **ROI(V_i)** = vehicles within `ROI_RADIUS` of each vehicle of interest `V_i`.
4. **BFS** = FIFO breadth-first search from `V_i` on the ROC graph, restricted to ROI and `MAX_HOPS`.
5. **ITG direction** = reverse each BFS-tree parent edge so influence points inward to the VOI.
6. Rebuild the graph at the next time step.

Default experimental ITG values:

```text
ROC radius = 35 m
ROI radius = 80 m
MAX_HOPS   = 4
```

These three ITG values are **research choices for the extension**; they are not parameters stated by the CommonRoad-Geometric paper.

---

# Paper alignment vs implementation choices

## Directly supported by Meyer et al. (2023)

- heterogeneous vehicle/lanelet graph
- V2V, V2L/L2V, L2L and VTV relations
- custom V2V edge drawer concept
- Table-II node/edge geometry
- edge-enhanced HGT concept
- GRU encoding of lanelet waypoint sequences
- learnable L2L adjacency embedding
- Time2Vec for VTV time delta
- GRU decoder producing local position/orientation transitions
- ADE training objective
- 1.0 s future horizon, 0.2 s interval (5 steps)
- ADE and FDE reporting

## Public cr-geo repository details used where the paper is silent

- 15 observed steps
- 20-step temporal collection and non-overlap
- 256-dimensional node hidden representation
- 8 GNN layers
- 16 attention heads
- Time2Vec dimension 16
- decoder GRU hidden size 512

## Explicit choices in this reproducible project, not claimed as paper constants

- seed 42
- scenario split 70/15/15
- AdamW, learning rate `1e-3`, weight decay `1e-5`
- 30 epochs
- ITG ROC/ROI radii and `MAX_HOPS`

Because those details are not all stated in the paper, **do not promise that your numerical ADE/FDE will exactly equal Table IV**. The published reference values are useful comparison targets, not guaranteed outputs of this independent implementation.

Paper Table-IV reference values:

| City | Scenarios | ADE (m) | FDE (m) |
|---|---:|---:|---:|
| Singapore | 2,372 | 0.106 | 0.227 |
| Boston | 938 | 0.138 | 0.316 |
| Pittsburgh | 1,560 | 0.215 | 0.454 |

Your scientific result is the controlled comparison between `paper` and `itg` under the same data split and model/training code.
