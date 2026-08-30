"""Configuration for the paper-aligned CommonRoad-Geometric trajectory model.

The values labelled PAPER are stated by Meyer et al. (2023). Values labelled
REPOSITORY DETAIL are not reported in the 8-page paper; they follow the public
crgeo trajectory-prediction implementation where possible and are exposed here
so they can be changed without changing the model code.
"""

SEED = 42

# ---------------------------------------------------------------------------
# Temporal sampling
# ---------------------------------------------------------------------------
# PAPER: the shown experiment predicts 1.0 s at 0.2 s intervals => 5 steps.
MODEL_DT = 0.2
PRED_STEPS = 5

# REPOSITORY DETAIL: the paper does not state the number of past observations.
# The public crgeo trajectory project uses 15 observed model steps.
OBS_STEPS = 15

# Dataset collection stride measured in MODEL_DT steps. 1 means create a
# training window every 0.2 s when the source scenario supports it.
WINDOW_STRIDE = 1
MIN_CONTEXT_VEHICLES = 1

# ---------------------------------------------------------------------------
# Graph structure / features from Table II and Sec. III-A of the paper
# ---------------------------------------------------------------------------
VEHICLE_FEATURE_DIM = 10       # p(2), theta, yaw-rate, v(2), a(2), width, length
V2V_EDGE_DIM = 8               # distance, rel-pos(2), rel-theta, rel-v(2), rel-a(2)
V2L_EDGE_DIM = 6               # left/right dist, lateral offset, heading err, s, s/L
L2L_NUMERIC_EDGE_DIM = 6       # distance, rel-pos(2), rel-theta, s_src, s_dst
L2L_RELATION_COUNT = 7         # predecessor/successor/adjacent L/R/merge/diverge/conflict
LANE_STATIC_DIM = 4            # p_L(2), lane length, theta_L
LANE_GEOMETRY_DIM = 4          # local left(x,y) + local right(x,y) per waypoint pair

# PAPER: default V2V drawer in Table III is VoronoiEdgeDrawer (Delaunay edges).
V2V_EDGE_DRAWER = "voronoi"

# PAPER: default temporal drawer is causal. None means every earlier observation
# is connected to every later observation of the same vehicle in the window.
VTV_MAX_FUTURE_STEPS = None

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
# PAPER: edge-enhanced HGT encoder, Time2Vec for VTV delta-time, GRU lane encoder,
# learnable L2L type embedding, GRU trajectory decoder.
# Hidden sizes/layer count are implementation details not enumerated in the paper.
HIDDEN_DIM = 256
LANE_GRU_HIDDEN_DIM = 64
L2L_RELATION_EMBED_DIM = 10
TIME2VEC_DIM = 16
HGT_LAYERS = 8
HGT_HEADS = 16
DECODER_HIDDEN_DIM = 512

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
GRAD_CLIP_NORM = 5.0
