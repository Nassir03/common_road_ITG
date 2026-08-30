"""Graph feature geometry for the CommonRoad-Geometric paper model."""
from __future__ import annotations
import math
import numpy as np

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def rotate_to_local(dx,dy,heading):
    c,s=math.cos(heading),math.sin(heading)
    return c*dx+s*dy, -s*dx+c*dy

def state_velocity_xy(state):
    speed=float(state.get("velocity",0.0)); h=float(state.get("orientation",0.0))
    return speed*math.cos(h), speed*math.sin(h)

def state_acceleration_xy(state):
    a=float(state.get("acceleration",0.0)); h=float(state.get("orientation",0.0))
    return a*math.cos(h), a*math.sin(h)

def vehicle_feature_vector(vehicle,state):
    vx,vy=state_velocity_xy(state); ax,ay=state_acceleration_xy(state)
    return [float(state["x"]),float(state["y"]),float(state.get("orientation",0.0)),float(state.get("yaw_rate",0.0)),vx,vy,ax,ay,float(vehicle.get("width",0.0)),float(vehicle.get("length",0.0))]

def v2v_feature_vector(source,target):
    sx,sy=float(source["x"]),float(source["y"]); tx,ty=float(target["x"]),float(target["y"])
    sh=float(source.get("orientation",0.0)); th=float(target.get("orientation",0.0))
    dx,dy=rotate_to_local(tx-sx,ty-sy,sh); dist=math.hypot(tx-sx,ty-sy)
    svx,svy=state_velocity_xy(source); tvx,tvy=state_velocity_xy(target)
    rvx,rvy=rotate_to_local(tvx-svx,tvy-svy,sh)
    sax,say=state_acceleration_xy(source); tax,tay=state_acceleration_xy(target)
    rax,ray=rotate_to_local(tax-sax,tay-say,sh)
    return [dist,dx,dy,wrap_angle(th-sh),rvx,rvy,rax,ray]

def _project_segment(point,a,b):
    px,py=point; ax,ay=a; bx,by=b; vx,vy=bx-ax,by-ay; den=vx*vx+vy*vy
    if den<=1e-12: return (ax,ay),0.0,math.hypot(px-ax,py-ay)
    u=max(0.0,min(1.0,((px-ax)*vx+(py-ay)*vy)/den)); q=(ax+u*vx,ay+u*vy)
    return q,u,math.hypot(px-q[0],py-q[1])

def project_to_polyline(point,points):
    if not points: return point,0.0,0.0,0.0
    if len(points)==1: return points[0],0.0,0.0,math.dist(point,points[0])
    best=None; prefix=0.0
    for a,b in zip(points,points[1:]):
        seg=math.dist(a,b); q,u,d=_project_segment(point,a,b); h=math.atan2(b[1]-a[1],b[0]-a[0]) if seg>1e-12 else 0.0
        cand=(q,prefix+u*seg,h,d)
        if best is None or d<best[3]: best=cand
        prefix+=seg
    return best

def v2l_feature_vector(state,lane):
    point=(float(state["x"]),float(state["y"]))
    _,_,_,dl=project_to_polyline(point,lane.get("left",[])); _,_,_,dr=project_to_polyline(point,lane.get("right",[]))
    _,s,h,_=project_to_polyline(point,lane.get("center",[])); lateral=(dl-dr)/2.0
    L=max(float(lane.get("length",0.0)),1e-12)
    return [dl,dr,lateral,wrap_angle(h-float(state.get("orientation",0.0))),s,s/L]

def lane_local_geometry(lane):
    left,right=lane.get("left",[]),lane.get("right",[]); n=min(len(left),len(right))
    if n==0: return [[0.0,0.0,0.0,0.0]]
    center=lane.get("center",[]); origin=center[0] if center else ((left[0][0]+right[0][0])/2,(left[0][1]+right[0][1])/2)
    h=float(lane.get("heading",0.0)); rows=[]
    for l,r in zip(left[:n],right[:n]):
        llx,lly=rotate_to_local(l[0]-origin[0],l[1]-origin[1],h); rlx,rly=rotate_to_local(r[0]-origin[0],r[1]-origin[1],h)
        rows.append([llx,lly,rlx,rly])
    return rows

def lane_static_feature_vector(lane):
    c=lane.get("center",[]); o=c[0] if c else (0.0,0.0)
    return [float(o[0]),float(o[1]),float(lane.get("length",0.0)),float(lane.get("heading",0.0))]

def _segment_intersection(a,b,c,d,eps=1e-9):
    r=np.asarray([b[0]-a[0],b[1]-a[1]],float); s=np.asarray([d[0]-c[0],d[1]-c[1]],float); qp=np.asarray([c[0]-a[0],c[1]-a[1]],float)
    cross=float(r[0]*s[1]-r[1]*s[0])
    if abs(cross)<=eps: return None
    t=float((qp[0]*s[1]-qp[1]*s[0])/cross); u=float((qp[0]*r[1]-qp[1]*r[0])/cross)
    if -eps<=t<=1+eps and -eps<=u<=1+eps: return max(0,min(1,t)),max(0,min(1,u))
    return None

def intersection_arclengths(a_points,b_points):
    ap=0.0
    for a0,a1 in zip(a_points,a_points[1:]):
        al=math.dist(a0,a1); bp=0.0
        for b0,b1 in zip(b_points,b_points[1:]):
            bl=math.dist(b0,b1); hit=_segment_intersection(a0,a1,b0,b1)
            if hit is not None:
                ta,tb=hit; return ap+ta*al,bp+tb*bl
            bp+=bl
        ap+=al
    return 0.0,0.0

def l2l_numeric_feature_vector(source,target):
    sc,tc=source.get("center",[]),target.get("center",[]); sp=sc[0] if sc else (0,0); tp=tc[0] if tc else (0,0)
    sh=float(source.get("heading",0.0)); th=float(target.get("heading",0.0)); dx,dy=rotate_to_local(tp[0]-sp[0],tp[1]-sp[1],sh)
    ss,st=intersection_arclengths(sc,tc)
    return [math.dist(sp,tp),dx,dy,wrap_angle(th-sh),ss,st]

def delaunay_directed_edges(points):
    n=len(points)
    if n<2: return []
    if n==2: return [(0,1),(1,0)]
    try:
        from scipy.spatial import Delaunay
        tri=Delaunay(np.asarray(points,float)); undirected=set()
        for simplex in tri.simplices:
            vals=[int(v) for v in simplex]
            for i in range(len(vals)):
                for j in range(i+1,len(vals)):
                    a,b=sorted((vals[i],vals[j]));
                    if a!=b: undirected.add((a,b))
        return sorted([(a,b) for a,b in undirected]+[(b,a) for a,b in undirected])
    except Exception:
        # Only for degenerate collinear/duplicate snapshots.
        return [(i,j) for i in range(n) for j in range(n) if i!=j]
