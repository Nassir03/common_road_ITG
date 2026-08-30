"""TraInX-style ROT -> ROC -> ROI -> FIFO BFS -> inward ITG."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ITGEdge:
    voi: int
    source: int
    target: int
    hop: int
    branch: int
    branch_count: int
    direct: bool
    distance: float

def compute_rot(states: dict[int,dict]) -> dict[str, object]:
    """Describe the Region of Traffic (ROT) for the current snapshot.

    In this implementation the current snapshot itself is the ROT population.
    The returned center/radius are diagnostic metadata; ROC/ROI filtering is
    performed on this population.
    """
    if not states:
        return {"vehicle_ids": [], "center": (0.0, 0.0), "radius": 0.0}
    ids = sorted(states)
    cx = sum(float(states[i]["x"]) for i in ids) / len(ids)
    cy = sum(float(states[i]["y"]) for i in ids) / len(ids)
    radius = max(math.hypot(float(states[i]["x"])-cx, float(states[i]["y"])-cy) for i in ids)
    return {"vehicle_ids": ids, "center": (cx, cy), "radius": radius}

def communication_graph(states: dict[int,dict], radius: float) -> dict[int,list[int]]:
    ids=sorted(states); g={i:[] for i in ids}
    for a_i,a in enumerate(ids):
        sa=states[a]
        for b in ids[a_i+1:]:
            sb=states[b]
            if math.hypot(float(sb["x"])-float(sa["x"]),float(sb["y"])-float(sa["y"])) <= radius:
                g[a].append(b); g[b].append(a)
    return {k:sorted(v) for k,v in g.items()}

def roi_members(states: dict[int,dict], voi: int, radius: float) -> set[int]:
    s=states[voi]
    return {j for j,t in states.items() if j==voi or math.hypot(float(t["x"])-float(s["x"]),float(t["y"])-float(s["y"])) <= radius}

def bfs_tree(graph, source, allowed, max_hops):
    q=deque([source]); visited={source}; hops={source:0}; parent={}; branch={source:0}; next_branch=1
    while q:
        u=q.popleft()
        if hops[u]>=max_hops: continue
        for v in graph.get(u,[]):
            if v not in allowed or v in visited: continue
            visited.add(v); parent[v]=u; hops[v]=hops[u]+1
            if u==source:
                branch[v]=next_branch; next_branch+=1
            else:
                branch[v]=branch[u]
            q.append(v)
    return hops,parent,branch,max(next_branch-1,1)

def build_itg_multigraph(states: dict[int,dict], roc_radius: float, roi_radius: float, max_hops: int) -> list[ITGEdge]:
    """Union of every VOI-specific BFS tree, preserving contextual duplicate edges."""
    if roi_radius < roc_radius: raise ValueError("ROI radius must be >= ROC radius")
    graph=communication_graph(states,roc_radius); out=[]
    for voi in sorted(states):
        allowed=roi_members(states,voi,roi_radius); hops,parent,branch,branch_count=bfs_tree(graph,voi,allowed,max_hops)
        for node,p in parent.items():
            s,t=states[node],states[p]
            d=math.hypot(float(t["x"])-float(s["x"]),float(t["y"])-float(s["y"]))
            out.append(ITGEdge(voi,node,p,hops[node],branch[node],branch_count,hops[node]==1,d))
    return out
