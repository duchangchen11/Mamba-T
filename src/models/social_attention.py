"""Relation-aware neighbor cross-attention with safe all-masked rows."""
import torch
from torch import nn


class SocialCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.relation_embedding=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,128))
        self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
    def forward(self,target_context,neighbor_context,relation,neighbor_mask):
        valid=neighbor_mask.bool();has=valid.any(1)
        # Never evaluate attention on all-masked rows. Preserve differentiable zeros.
        result=target_context*0
        if has.any():
            m=valid[has]
            n=neighbor_context[has].masked_fill(~m[...,None],0)
            r=relation[has].masked_fill(~m[...,None],0)
            kv=n+self.relation_embedding(r)
            c,_=self.attention(target_context[has,None],kv,kv,key_padding_mask=~m,need_weights=False)
            result=result.index_copy(0,has.nonzero(as_tuple=True)[0],c[:,0])
        return result
