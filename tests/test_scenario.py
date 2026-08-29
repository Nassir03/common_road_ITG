from model.scenario import point_in_polygon, assign_lanelets


def test_multiple_center_lanelet_assignments():
    lane={"left":[(0,0),(10,0)],"right":[(0,4),(10,4)],"center":[(0,2),(10,2)]}
    lane2={"left":[(0,1),(10,1)],"right":[(0,5),(10,5)],"center":[(0,3),(10,3)]}
    ids=assign_lanelets(5,2,{1:lane,2:lane2})
    assert ids==[1,2]
    assert point_in_polygon(5,2,lane["left"]+list(reversed(lane["right"])))
