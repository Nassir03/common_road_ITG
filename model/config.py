"""Configuration for the paper-only CommonRoad-Geometric reproduction.

Only graph/model components described in Meyer et al. (2023) are implemented:
- heterogeneous vehicle/lanelet graph: V2V, V2L, L2V, L2L
- temporal VTV edges with elapsed-time feature
- Voronoi/Delaunay V2V construction (the paper's default edge drawer)
- edge-enhanced HGT encoder
- Time2Vec for VTV delta-time
- GRU lanelet geometry encoder
- learned L2L adjacency-type embedding
- GRU decoder that emits local position/orientation deltas
- ADE training objective and ADE/FDE evaluation

The paper does not publish every hyperparameter (history length, hidden sizes,
optimizer, split ratio, batch size, etc.). Those necessary implementation
choices are isolated here and are not presented as paper-reported constants.
"""

SEED = 42

# Paper-reported prediction setup: 1.0 s horizon at 0.2 s intervals -> 5 steps.
MODEL_DT = 0.2
PRED_STEPS = 5

# The paper's Fig. 3 visualizes t-4,...,t (5 observed graph states), while the
# exact experiment history length is not stated in the paper. Five is therefore
# used as a small, transparent implementation default and can be changed here.
OBS_STEPS = 5
WINDOW_STRIDE = OBS_STEPS + PRED_STEPS  # non-overlapping windows for speed

# Default causal temporal drawer: connect a historic vehicle to future
# realizations within the observed cache. The paper defines a configurable
# T_max; using the whole 5-state cache is faithful and still inexpensive.
VTV_MAX_FUTURE_STEPS = OBS_STEPS - 1

# Feature dimensions from Table II.
VEHICLE_FEATURE_DIM = 10       # p(2), theta, yaw-rate, velocity(2), acceleration(2), width, length
V2V_EDGE_DIM = 8               # distance, rel-pos(2), rel-theta, rel-velocity(2), rel-acceleration(2)
V2L_EDGE_DIM = 6               # left/right distance, lateral offset, heading error, s, s/L
L2L_NUMERIC_EDGE_DIM = 6       # distance, rel-pos(2), rel-theta, s_src, s_dst
L2L_RELATION_COUNT = 7         # predecessor, successor, adjacent L/R, merging, diverging, conflicting
LANE_STATIC_DIM = 4            # lanelet origin position(2), length, orientation
LANE_GEOMETRY_DIM = 4          # lane-local left/right waypoint coordinates

# Practical edge-enhanced HGT implementation choices. The paper specifies the
# architecture family, not these exact widths/depths.
HIDDEN_DIM = 128
LANE_GRU_HIDDEN_DIM = 64
L2L_RELATION_EMBED_DIM = 8
TIME2VEC_DIM = 16
HGT_LAYERS = 3
HGT_HEADS = 4
DECODER_HIDDEN_DIM = 256
DROPOUT = 0.10

# Training choices not reported in the paper.
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 20
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 4
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 5

# Reproducible scenario-level split. The paper states same-city training and
# validation but does not publish an exact train/val/test ratio.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Spatial index used only to accelerate exact paper relations (V2L center
# assignment and conflicting-lanelet discovery). It does not alter the graph.
LANE_GRID_CELL_SIZE = 50.0
