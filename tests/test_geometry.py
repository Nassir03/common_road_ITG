from model.geometry import v2v_feature_vector, delaunay_directed_edges


def test_v2v_feature_dimension_and_local_frame():
    a = {"x": 0.0, "y": 0.0, "orientation": 0.0, "velocity": 10.0, "acceleration": 0.0}
    b = {"x": 3.0, "y": 4.0, "orientation": 0.0, "velocity": 8.0, "acceleration": 0.0}
    features = v2v_feature_vector(a, b)
    assert len(features) == 8
    assert abs(features[0] - 5.0) < 1e-6
    assert abs(features[1] - 3.0) < 1e-6
    assert abs(features[2] - 4.0) < 1e-6
    assert abs(features[4] - (-2.0)) < 1e-6


def test_voronoi_delaunay_edges_are_directed():
    edges = set(delaunay_directed_edges([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
    assert (0, 1) in edges and (1, 0) in edges
    assert (0, 2) in edges and (2, 0) in edges
