"""Paper-aligned edge-enhanced HGT + GRU trajectory decoder."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .config import (
    HIDDEN_DIM, LANE_GRU_HIDDEN_DIM, L2L_RELATION_EMBED_DIM, TIME2VEC_DIM,
    HGT_LAYERS, HGT_HEADS, DECODER_HIDDEN_DIM, PRED_STEPS,
    VEHICLE_FEATURE_DIM, V2L_EDGE_DIM, L2L_NUMERIC_EDGE_DIM, L2L_RELATION_COUNT,
    LANE_GEOMETRY_DIM, LANE_STATIC_DIM, PAPER_V2V_EDGE_DIM,
)
NODE_TYPES=("vehicle","lane")
REL_META={"v2v":("vehicle","vehicle"),"v2l":("vehicle","lane"),"l2v":("lane","vehicle"),"l2l":("lane","lane"),"vtv":("vehicle","vehicle")}

class Time2Vec(nn.Module):
    def __init__(self,dim=TIME2VEC_DIM):
        super().__init__(); self.frequency=nn.Parameter(torch.empty(dim)); self.phase=nn.Parameter(torch.zeros(dim)); nn.init.normal_(self.frequency)
    def forward(self,t):
        t=t.reshape(-1,1); raw=t*self.frequency.reshape(1,-1)+self.phase.reshape(1,-1); return torch.cat([raw[:,:1],torch.sin(raw[:,1:])],dim=-1)

class LaneletEncoder(nn.Module):
    def __init__(self,hidden_dim=HIDDEN_DIM):
        super().__init__(); self.gru=nn.GRU(LANE_GEOMETRY_DIM,LANE_GRU_HIDDEN_DIM,batch_first=True); self.static_norm=nn.LayerNorm(LANE_STATIC_DIM)
        self.output=nn.Sequential(nn.Linear(LANE_GRU_HIDDEN_DIM+LANE_STATIC_DIM,hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim))
    def forward(self,geometry,lengths,static_x):
        if static_x.size(0)==0: return static_x.new_empty((0,self.output[-1].normalized_shape[0]))
        packed=pack_padded_sequence(geometry,lengths.cpu(),batch_first=True,enforce_sorted=False); _,h=self.gru(packed)
        return self.output(torch.cat([h[-1],self.static_norm(static_x)],dim=-1))

class EdgeEnhancedHGTLayer(nn.Module):
    def __init__(self,hidden_dim,heads,edge_dims):
        super().__init__()
        if hidden_dim%heads: raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim=hidden_dim; self.heads=heads; self.head_dim=hidden_dim//heads
        self.q=nn.ModuleDict({t:nn.Linear(hidden_dim,hidden_dim) for t in NODE_TYPES}); self.k=nn.ModuleDict({t:nn.Linear(hidden_dim,hidden_dim) for t in NODE_TYPES}); self.v=nn.ModuleDict({t:nn.Linear(hidden_dim,hidden_dim) for t in NODE_TYPES})
        self.out=nn.ModuleDict({t:nn.Linear(hidden_dim,hidden_dim) for t in NODE_TYPES}); self.norm=nn.ModuleDict({t:nn.LayerNorm(hidden_dim) for t in NODE_TYPES}); self.skip=nn.ParameterDict({t:nn.Parameter(torch.zeros(1)) for t in NODE_TYPES})
        self.eq=nn.ModuleDict({r:nn.Linear(edge_dims[r],hidden_dim) for r in REL_META}); self.ek=nn.ModuleDict({r:nn.Linear(edge_dims[r],hidden_dim) for r in REL_META}); self.ev=nn.ModuleDict({r:nn.Linear(edge_dims[r],hidden_dim) for r in REL_META})
        self.ratt=nn.ParameterDict(); self.rmsg=nn.ParameterDict(); self.rprior=nn.ParameterDict()
        for r in REL_META:
            a=nn.Parameter(torch.empty(heads,self.head_dim,self.head_dim)); m=nn.Parameter(torch.empty(heads,self.head_dim,self.head_dim)); nn.init.xavier_uniform_(a); nn.init.xavier_uniform_(m)
            self.ratt[r]=a; self.rmsg[r]=m; self.rprior[r]=nn.Parameter(torch.ones(heads))
    @staticmethod
    def segment_softmax(scores,dst,n_dst):
        if scores.numel()==0: return scores
        ix=dst[:,None].expand(-1,scores.size(1)); mx=torch.full((n_dst,scores.size(1)),-torch.inf,dtype=scores.dtype,device=scores.device); mx.scatter_reduce_(0,ix,scores,reduce="amax",include_self=True)
        ex=torch.exp(scores-mx[dst]); den=torch.zeros((n_dst,scores.size(1)),dtype=scores.dtype,device=scores.device); den.scatter_add_(0,ix,ex); return ex/den[dst].clamp_min(1e-12)
    def forward(self,node_h,edge_index,edge_attr):
        rel_out={t:[] for t in NODE_TYPES}
        for r,(st,dt) in REL_META.items():
            ei=edge_index[r]
            if ei.numel()==0 or node_h[st].size(0)==0 or node_h[dt].size(0)==0: continue
            src,dst=ei[0],ei[1]; e=edge_attr[r]
            q=(self.q[dt](node_h[dt][dst])+self.eq[r](e)).view(-1,self.heads,self.head_dim)
            k=(self.k[st](node_h[st][src])+self.ek[r](e)).view(-1,self.heads,self.head_dim)
            v=(self.v[st](node_h[st][src])+self.ev[r](e)).view(-1,self.heads,self.head_dim)
            km=torch.einsum("ehd,hdf->ehf",k,self.ratt[r]); vm=torch.einsum("ehd,hdf->ehf",v,self.rmsg[r]); score=(q*km).sum(-1)*self.rprior[r].reshape(1,-1)/math.sqrt(self.head_dim)
            alpha=self.segment_softmax(score,dst,node_h[dt].size(0)); agg=node_h[dt].new_zeros((node_h[dt].size(0),self.heads,self.head_dim)); agg.index_add_(0,dst,alpha.unsqueeze(-1)*vm); agg=agg.reshape(node_h[dt].size(0),self.hidden_dim)
            mask=torch.zeros(node_h[dt].size(0),dtype=torch.bool,device=dst.device); mask[dst]=True; rel_out[dt].append((agg,mask))
        out={}
        for t in NODE_TYPES:
            h=node_h[t]
            if h.size(0)==0 or not rel_out[t]: out[t]=h; continue
            vals=torch.stack([x for x,_ in rel_out[t]],0); masks=torch.stack([m for _,m in rel_out[t]],0); vals=vals.masked_fill(~masks.unsqueeze(-1),-torch.inf); agg=vals.max(0).values; any_in=masks.any(0); agg=torch.where(any_in.unsqueeze(-1),agg,torch.zeros_like(agg))
            transformed=torch.relu(self.out[t](agg)); beta=torch.sigmoid(self.skip[t]); updated=beta*transformed+(1-beta)*h; updated=torch.where(any_in.unsqueeze(-1),updated,h); out[t]=self.norm[t](updated)
        return out

class LocalTrajectoryGRUDecoder(nn.Module):
    def __init__(self,context_dim,hidden_dim=DECODER_HIDDEN_DIM,pred_steps=PRED_STEPS):
        super().__init__(); self.pred_steps=pred_steps; self.init_hidden=nn.Sequential(nn.Linear(context_dim,hidden_dim),nn.Tanh()); self.cell=nn.GRUCell(3,hidden_dim); self.head=nn.Linear(hidden_dim,3); nn.init.normal_(self.head.weight,std=1e-3); nn.init.zeros_(self.head.bias)
    def forward(self,context,current_position,current_orientation):
        n=context.size(0); h=self.init_hidden(context); prev=context.new_zeros((n,3)); pos=current_position; ori=current_orientation; ps=[]; os=[]; ds=[]
        for _ in range(self.pred_steps):
            h=self.cell(prev,h); d=self.head(h); dx,dy,dth=d[:,0],d[:,1],d[:,2]; c,s=torch.cos(ori),torch.sin(ori); pos=pos+torch.stack([c*dx-s*dy,s*dx+c*dy],-1); ori=torch.atan2(torch.sin(ori+dth),torch.cos(ori+dth)); ps.append(pos); os.append(ori); ds.append(d); prev=d
        return {"position":torch.stack(ps,1),"orientation":torch.stack(os,1),"local_delta":torch.stack(ds,1)}

class CrGeoTrajectoryPredictionModel(nn.Module):
    def __init__(self,v2v_edge_dim=PAPER_V2V_EDGE_DIM,hidden_dim=HIDDEN_DIM,heads=HGT_HEADS,hgt_layers=HGT_LAYERS,pred_steps=PRED_STEPS,decoder_hidden_dim=DECODER_HIDDEN_DIM):
        super().__init__(); self.v2v_edge_dim=v2v_edge_dim
        self.vehicle_encoder=nn.Sequential(nn.LayerNorm(VEHICLE_FEATURE_DIM),nn.Linear(VEHICLE_FEATURE_DIM,hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim)); self.lane_encoder=LaneletEncoder(hidden_dim)
        self.l2l_embedding=nn.Embedding(L2L_RELATION_COUNT,L2L_RELATION_EMBED_DIM); self.time2vec=Time2Vec(TIME2VEC_DIM)
        dims={"v2v":v2v_edge_dim,"v2l":V2L_EDGE_DIM,"l2v":V2L_EDGE_DIM,"l2l":L2L_NUMERIC_EDGE_DIM+L2L_RELATION_EMBED_DIM,"vtv":PAPER_V2V_EDGE_DIM+TIME2VEC_DIM}
        self.hgt=nn.ModuleList([EdgeEnhancedHGTLayer(hidden_dim,heads,dims) for _ in range(hgt_layers)]); self.decoder=LocalTrajectoryGRUDecoder(hidden_dim,decoder_hidden_dim,pred_steps)
    def _attrs(self,sample,device):
        raw=sample["edge_attr"]; ln=raw["l2l_numeric"].to(device); lt=raw["l2l_type"].to(device); l2l=torch.cat([ln,self.l2l_embedding(lt)],-1) if ln.size(0) else ln.new_empty((0,L2L_NUMERIC_EDGE_DIM+L2L_RELATION_EMBED_DIM))
        vm=raw["vtv_motion"].to(device); dt=raw["vtv_delta_t"].to(device); vtv=torch.cat([vm,self.time2vec(dt)],-1) if vm.size(0) else vm.new_empty((0,PAPER_V2V_EDGE_DIM+TIME2VEC_DIM))
        return {"v2v":raw["v2v"].to(device),"v2l":raw["v2l"].to(device),"l2v":raw["l2v"].to(device),"l2l":l2l,"vtv":vtv}
    def forward(self,sample):
        device=next(self.parameters()).device; vh=self.vehicle_encoder(sample["vehicle_x"].to(device)); lh=self.lane_encoder(sample["lane_geometry"].to(device),sample["lane_geometry_lengths"].to(device),sample["lane_x"].to(device)); nodes={"vehicle":vh,"lane":lh}
        edges={k:v.to(device) for k,v in sample["edge_index"].items()}; attrs=self._attrs(sample,device)
        for layer in self.hgt: nodes=layer(nodes,edges,attrs)
        context=nodes["vehicle"][sample["latest_vehicle_node_index"].to(device)]
        return self.decoder(context,sample["current_position"].to(device),sample["current_orientation"].to(device))
