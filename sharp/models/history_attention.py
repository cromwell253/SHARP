"""History-selective attention relative to the shared convex baseline."""
import torch
from torch import nn


class HistorySelectiveAttention(nn.Module):
    def __init__(self, nodes, logits):
        super().__init__()
        self.register_buffer('base_logits', torch.as_tensor(logits).clone())
        self.node = nn.Embedding(nodes, 8)
        self.state = nn.Sequential(nn.Linear(24, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.query = nn.Linear(32, 48*8)
        self.position = nn.Parameter(torch.randn(12, 8)*.2)
        self.history_key = nn.Linear(2, 8, bias=False)
        nn.init.zeros_(self.query.weight); nn.init.zeros_(self.query.bias)

    def forward(self, history, context, node, mode):
        relative = history-history[:, -1:]
        amplitude = relative.square().mean(-1, keepdim=True).sqrt().clamp_min(.001)
        normalized = relative/amplitude
        change = torch.cat([torch.zeros_like(normalized[:, :1]), normalized[:, 1:]-normalized[:, :-1]], -1)
        if mode == 'node_only':
            context = torch.zeros_like(context)
            normalized = torch.zeros_like(normalized); change = torch.zeros_like(change)
        state = self.state(torch.cat([context, self.node(node)], -1))
        query = self.query(state).reshape(-1, 48, 8)
        key = self.position[None]+self.history_key(torch.stack([normalized, change], -1))
        logits = self.base_logits[None]+torch.einsum('bhd,bld->bhl', query, key)/8**.5
        delta_weight = logits.softmax(-1)-self.base_logits.softmax(-1)[None]
        return torch.einsum('bhl,bl->bh', delta_weight, relative)