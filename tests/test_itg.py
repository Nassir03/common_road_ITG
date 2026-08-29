from model.itg import VehicleState, bfs_tree, build_itg_for_vehicle


def test_fifo_bfs_shortest_hops():
    graph = {0:[1,2],1:[0,3],2:[0,3],3:[1,2,4],4:[3]}
    order,hops,parent=bfs_tree(graph,0,set(graph),4)
    assert order[0]==0 and hops[1]==1 and hops[2]==1 and hops[3]==2 and hops[4]==3
    assert parent[4]==3


def test_influence_points_inward_and_hops_are_voi_specific():
    states=[VehicleState(0,0,0,0,0),VehicleState(1,10,0,0,0),VehicleState(2,20,0,0,0)]
    snap=build_itg_for_vehicle(states,0,0,11,30,4)
    pairs={(e.source,e.target,e.hop) for e in snap.edges}
    assert (1,0,1) in pairs and (2,1,2) in pairs
    assert all(e.vehicle_of_interest==0 for e in snap.edges)


def test_roi_not_smaller_than_roc():
    states=[VehicleState(0,0,0,0,0)]
    try: build_itg_for_vehicle(states,0,0,20,10,2)
    except ValueError: return
    assert False
