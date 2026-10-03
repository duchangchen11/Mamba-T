"""Frozen shared temporal encoders, zero-initialized residual, optional scalar gate."""
import torch
from torch import nn
from src.models.social_attention import SocialCrossAttention
VARIANTS=('emt_zr','emt_sr','emt_gsr','ett_sr')


class SocialResidualPredictor(nn.Module):
    def __init__(self,backbone,variant,seed=42):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.backbone=backbone
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.backbone.eval()
        # Independent module RNG gives all controls the same adapter initial state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.social=None if variant=='emt_zr' else SocialCrossAttention()
            torch.manual_seed(seed+10000)
            self.residual=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,24))
            nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
            torch.manual_seed(seed+20000)
            self.gate=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid()) if variant=='emt_gsr' else None
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward_context(self,target_context,neighbor_context,relation,mask,base_prediction):
        c=torch.zeros_like(target_context) if self.social is None else self.social(target_context,neighbor_context,relation,mask)
        combined=torch.cat((target_context,c),dim=-1)
        delta=self.residual(combined).reshape(-1,12,2)
        gate=torch.ones((len(combined),1),device=combined.device,dtype=combined.dtype) if self.gate is None else self.gate(combined)
        return {'future_pred':base_prediction+gate[:,:,None]*delta,'base_prediction':base_prediction,'residual':delta,'gate':gate,'social_context':c}
    def forward(self,target_history,neighbor_history,neighbor_mask,neighbor_relation):
        with torch.no_grad():
            _,h=self.backbone.encode(target_history)
            base=self.backbone.decoder(h).reshape(-1,12,2)
            n=torch.zeros((*neighbor_mask.shape,128),device=h.device,dtype=h.dtype)
            if self.social is not None and neighbor_mask.any():
                _,encoded=self.backbone.encode(neighbor_history[neighbor_mask])
                n[neighbor_mask]=encoded
        return self.forward_context(h,n,neighbor_relation,neighbor_mask,base)
