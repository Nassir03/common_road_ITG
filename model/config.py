"""Central experiment configuration.

The values are intentionally small and easy to change. Distances are metres.
"""
SEED = 42

# Dynamic ITG geometry
COMMUNICATION_RADIUS = 35.0   # ROC
ROI_RADIUS = 80.0             # ROI >= ROC
MAX_HOPS = 4

# Trajectory windows. 0.1 s is common in the supplied CommonRoad files,
# so 8 observed + 12 predicted steps is 0.8 s history + 1.2 s future.
OBS_STEPS = 8
PRED_STEPS = 12
WINDOW_STRIDE = 10
MIN_CONTEXT_VEHICLES = 2
MAX_TARGETS_PER_WINDOW = 4

# Features
LANE_POINTS = 20
VEHICLE_FEATURE_DIM = 10
ITG_EDGE_FEATURE_DIM = 9
LANE_FEATURE_DIM = 3
LANE_GEOMETRY_FEATURE_DIM = 4
L2L_RELATION_COUNT = 7

# Model/training
HIDDEN_DIM = 64
TIME2VEC_DIM = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
