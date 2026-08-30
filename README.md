# CommonRoad-Geometric Paper Trajectory Prediction — Paper-Aligned Implementation

This version replaces the earlier custom **ITG/ROI/BFS/hop/branch** trajectory model with the method demonstrated in the CommonRoad-Geometric paper by Meyer et al. (2023).

## What the code now implements

```text
NuPlan
  -> CommonRoad conversion
  -> heterogeneous temporal graph
       Vehicle nodes + Lanelet nodes
       V2V + V2L + L2V + L2L + causal VTV edges
  -> feature encoding
       Time2Vec(VTV delta time)
       GRU(lane boundary waypoint sequence)
       learnable L2L relation embedding
  -> edge-enhanced HGT encoder
  -> vehicle embedding
  -> GRU decoder
  -> local [dx, dy, dtheta] sequence
  -> recurrent local-frame integration
  -> future trajectory at 0.2, 0.4, 0.6, 0.8, 1.0 seconds
  -> ADE training loss, ADE/FDE evaluation
```

The exact feature mapping is documented in `PAPER_ALIGNMENT.md`.

## Important before training

The **code is paper-aligned, but the data bundled in your original ZIP is not the paper’s NuPlan Singapore/Boston/Pittsburgh dataset.** Training on that data is useful for checking that the implementation works, but its ADE/FDE must not be compared directly with the paper’s Table IV.

For true paper reproduction, first obtain/convert the NuPlan scenarios with the official CommonRoad converter, then prepare separate city experiments for Singapore, Boston and Pittsburgh. The paper states that each experiment is trained and validated on data from the same city.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

## Data format

The preprocessing script expects already split CommonRoad XML files:

```text
data/raw/
  train/*.xml
  val/*.xml
  test/*.xml
```

Convert them to compact `.scenario.pt` files:

```bash
python scripts/preprocess.py
```

This produces:

```text
data/processed/
  train/*.scenario.pt
  val/*.scenario.pt
  test/*.scenario.pt
```

## Inspect one graph before training

```bash
python scripts/inspect_sample.py \
  --scenario-dir data/processed/train \
  --index 0
```

You should see:

- vehicle feature dimension = 10;
- V2V feature dimension = 8;
- V2L/L2V feature dimension = 6;
- all five edge types: V2V, V2L, L2V, L2L, VTV;
- prediction times `[0.2, 0.4, 0.6, 0.8, 1.0]`.

## Train

Full default architecture:

```bash
python scripts/train.py \
  --data-root data/processed \
  --epochs 30 \
  --output outputs/crgeo_paper_model.pt
```

Quick smoke test:

```bash
python scripts/train.py \
  --data-root data/processed \
  --epochs 1 \
  --stride 10 \
  --max-train-samples 5 \
  --max-val-samples 2 \
  --hidden-dim 32 \
  --heads 4 \
  --hgt-layers 1 \
  --decoder-hidden-dim 64 \
  --output outputs/smoke.pt
```

The small smoke configuration is only for checking execution. Use the defaults for the intended full model.

## Evaluate

```bash
python scripts/evaluate.py \
  --data-root data/processed \
  --split test \
  --checkpoint outputs/crgeo_paper_model.pt \
  --output-csv outputs/test_metrics.csv
```

The reported metrics are:

\[
ADE = \frac{1}{T}\sum_{k=1}^{T}\|\hat p_k-p_k\|_2
\]

and

\[
FDE = \|\hat p_T-p_T\|_2.
\]

## Predict one sample

```bash
python scripts/predict.py \
  --data-root data/processed \
  --split test \
  --checkpoint outputs/crgeo_paper_model.pt \
  --index 0
```

It prints predicted and ground-truth `(x, y, orientation)` for the five future steps.

## What was removed from the model path

The previous implementation used custom **ROT -> ROC -> ROI -> BFS -> ITG** construction, hop/branch labels, a radius baseline, and a 12-step XY-only decoder. Those are not the trajectory-prediction architecture described in the cr-geo paper, so they are no longer used by this implementation.
