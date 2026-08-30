import math

from model.geometry import v2l_feature_vector, v2v_feature_vector


def test_v2v_features_are_in_source_vehicle_frame():
    source = {"x": 0.0, "y": 0.0, "orientation": math.pi / 2, "velocity": 10.0, "acceleration": 2.0}
    target = {"x": 0.0, "y": 10.0, "orientation": math.pi / 2, "velocity": 12.0, "acceleration": 3.0}
    f = v2v_feature_vector(source, target)
    assert abs(f[0] - 10.0) < 1e-6       # distance
    assert abs(f[1] - 10.0) < 1e-6       # target is 10m ahead locally
    assert abs(f[2]) < 1e-6
    assert abs(f[3]) < 1e-6               # same heading
    assert abs(f[4] - 2.0) < 1e-6        # +2m/s longitudinal relative velocity
    assert abs(f[5]) < 1e-6
    assert abs(f[6] - 1.0) < 1e-6        # +1m/s^2 longitudinal relative accel
    assert abs(f[7]) < 1e-6


def test_v2l_features_match_paper_geometry():
    lane = {
        "left": [(0.0, 2.0), (10.0, 2.0)],
        "right": [(0.0, -2.0), (10.0, -2.0)],
        "center": [(0.0, 0.0), (10.0, 0.0)],
        "length": 10.0,
        "heading": 0.0,
    }
    state = {"x": 5.0, "y": 1.0, "orientation": 0.0}
    d_left, d_right, lateral, heading_error, s, s_norm = v2l_feature_vector(state, lane)
    assert abs(d_left - 1.0) < 1e-6
    assert abs(d_right - 3.0) < 1e-6
    assert abs(lateral + 1.0) < 1e-6
    assert abs(heading_error) < 1e-6
    assert abs(s - 5.0) < 1e-6
    assert abs(s_norm - 0.5) < 1e-6
