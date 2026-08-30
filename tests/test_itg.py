import math
from model.itg import communication_graph, roi_members, bfs_tree, build_itg_multigraph, compute_rot


def state(x,y):
    return {"x":x,"y":y,"orientation":0.0,"velocity":0.0,"acceleration":0.0,"yaw_rate":0.0}


def test_chain_bfs_and_inward_edges():
    states={4:state(0,0),7:state(10,0),8:state(20,0),9:state(30,0)}
    g=communication_graph(states,11.0)
    allowed=roi_members(states,4,40.0)
    hops,parent,branch,count=bfs_tree(g,4,allowed,4)
    assert hops[7]==1 and hops[8]==2 and hops[9]==3
    edges=build_itg_multigraph(states,11.0,40.0,4)
    voi4={(e.source,e.target,e.hop,e.direct) for e in edges if e.voi==4}
    assert (7,4,1,True) in voi4
    assert (8,7,2,False) in voi4
    assert (9,8,3,False) in voi4


def test_rot_contains_snapshot():
    states={1:state(0,0),2:state(4,0)}
    rot=compute_rot(states)
    assert rot["vehicle_ids"]==[1,2]
    assert rot["center"]==(2.0,0.0)
    assert math.isclose(rot["radius"],2.0)
