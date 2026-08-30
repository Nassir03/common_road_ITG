"""Central configuration.

Paper-supported choices:
- 1.0 s future horizon at 0.2 s intervals -> 5 predicted steps.
- heterogeneous vehicle/lanelet graph with V2V, V2L, L2V, L2L and VTV edges.
- edge-enhanced HGT encoder, lanelet GRU, Time2Vec on VTV, GRU decoder.

The paper does not publish every training hyperparameter. The values below keep
those choices explicit and easy to change.
"""
SEED = 42

# Temporal setup. The public crgeo trajectory project uses 15 observed + 5 future.
MODEL_DT = 0.2
OBS_STEPS = 15
PRED_STEPS = 5
WINDOW_STRIDE = OBS_STEPS + PRED_STEPS  # non-overlapping 20-step windows
MIN_CONTEXT_VEHICLES = 1

# ITG extension (experimental choices; not specified by Meyer et al.).
COMMUNICATION_RADIUS = 35.0  # ROC [m]
ROI_RADIUS = 80.0            # ROI [m]
MAX_HOPS = 4

# Feature dimensions from the paper's Table II.
VEHICLE_FEATURE_DIM = 10       # p(2), theta, yaw-rate, v(2), a(2), width, length
PAPER_V2V_EDGE_DIM = 8         # dist, rel-pos(2), rel-theta, rel-v(2), rel-a(2)
ITG_EXTRA_EDGE_DIM = 4         # normalized hop, branch, direct/indirect, normalized distance
ITG_V2V_EDGE_DIM = PAPER_V2V_EDGE_DIM + ITG_EXTRA_EDGE_DIM
V2L_EDGE_DIM = 6               # left/right distance, lateral offset, heading error, s, s/L
L2L_NUMERIC_EDGE_DIM = 6       # dist, rel-pos(2), rel-theta, s_src, s_dst
L2L_RELATION_COUNT = 7
LANE_STATIC_DIM = 4            # lane origin x/y, length, heading
LANE_GEOMETRY_DIM = 4          # lane-local left/right x/y per waypoint pair

# Neural network. These sizes follow the public crgeo trajectory project where possible.
HIDDEN_DIM = 256
LANE_GRU_HIDDEN_DIM = 64
L2L_RELATION_EMBED_DIM = 10
TIME2VEC_DIM = 16
HGT_LAYERS = 8
HGT_HEADS = 16
DECODER_HIDDEN_DIM = 512

# Training. Optimizer details are not stated in the 8-page paper.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
GRAD_CLIP_NORM = 1.0

# Reproducible same-city train/validation/test split.
# The paper says training/validation are city-specific but does not publish these ratios.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
