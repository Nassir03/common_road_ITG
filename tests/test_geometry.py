from model.geometry import v2v_feature_vector

def test_v2v_feature_dimension_and_local_frame():
    a={"x":0.0,"y":0.0,"orientation":0.0,"velocity":10.0,"acceleration":0.0}
    b={"x":3.0,"y":4.0,"orientation":0.0,"velocity":8.0,"acceleration":0.0}
    f=v2v_feature_vector(a,b)
    assert len(f)==8
    assert abs(f[0]-5.0)<1e-6
    assert abs(f[1]-3.0)<1e-6
    assert abs(f[2]-4.0)<1e-6
    assert abs(f[4]-(-2.0))<1e-6
