"""SHARP multi-step forecasting model."""
import torch
from torch import nn


class SHARP(nn.Module):
    def __init__(self, base, router, adapter, router_checkpoint, adapter_checkpoint):
        super().__init__()
        self.base,self.router,self.adapter=base,router,adapter
        self.mode=adapter_checkpoint['mode']
        for name,key in [('level_mean','level_mean'),('level_scale','level_scale'),
                         ('context_mean','context_mean'),('context_scale','context_scale')]:
            self.register_buffer(name,torch.as_tensor(router_checkpoint[key],dtype=torch.float64))
        for name,key in [('own_mean','own_mean'),('own_scale','own_scale')]:
            self.register_buffer(name,torch.as_tensor(adapter_checkpoint[key],dtype=torch.float64))
        self.register_buffer('neighbors',adapter_checkpoint['neighbors'].long())
        self.register_buffer('distances',adapter_checkpoint['distances'].float())
        self.register_buffer('neighbor_level_scale',torch.tensor(adapter_checkpoint['level_scale'],dtype=torch.float32))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()

    def forward(self,x,te):
        base=self.base(x,te).squeeze(-1)
        history=x[...,0].transpose(1,2);b,n,_=history.shape
        relative=history-history[...,-1:]
        amplitude=relative.double().square().mean(-1).sqrt().clamp_min(.001)
        context=torch.cat([relative.double()/amplitude[...,None],
            ((history[...,-1].double()-self.level_mean)/self.level_scale)[...,None],
            ((history.mean(-1).double()-self.level_mean)/self.level_scale)[...,None],
            amplitude.log()[...,None],(relative[...,0].double()/amplitude)[...,None]],-1)
        context=((context-self.context_mean)/self.context_scale).float()
        node=torch.arange(n,device=x.device)[None].expand(b,-1).reshape(-1)
        temporal=self.router(history.reshape(-1,12),context.reshape(-1,16),node,'conditional')
        routed=base+temporal.reshape(b,n,48).transpose(1,2)
        own=torch.cat([relative,routed.transpose(1,2)-history[...,-1:]],-1)
        own=((own.double()-self.own_mean)/self.own_scale).float().reshape(-1,60)
        neighbors=history[:,self.neighbors] if self.mode=='lagged_neighbors' else history[:,:,None].expand(-1,-1,4,-1)
        nr=neighbors-neighbors[...,-1:]
        level=(neighbors[...,-1]-history[...,-1,None])/self.neighbor_level_scale
        tokens=[]
        for lag in (0,1,2,3):
            shifted=torch.zeros_like(nr);shifted[...,lag:]=nr[...,:12-lag]
            mask=torch.zeros_like(nr);mask[...,lag:]=1
            # Preserve float32 token preprocessing before the MLP.
            normalized=(shifted-self.own_mean[:12].float())/self.own_scale[:12].float()
            tokens.append(torch.cat([normalized*mask,mask,level[...,None],
                                     self.distances[None,:,:,None].expand(b,-1,-1,-1)],-1))
        bank=torch.stack(tokens,3).reshape(-1,4,4,26)
        correction=self.adapter(own,bank,node).reshape(b,n,48).transpose(1,2)*.01
        return (routed+correction).unsqueeze(-1)