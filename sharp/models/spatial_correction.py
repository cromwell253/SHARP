"""Prediction-guided two-level attention over neighbor history views."""
import torch
from torch import nn


class SpatialIncrementCorrection(nn.Module):
    def __init__(self, nodes, hidden=32):
        super().__init__()
        self.node = nn.Embedding(nodes, 8)
        self.own = nn.Sequential(nn.Linear(68, 64), nn.GELU(), nn.Linear(64, hidden), nn.LayerNorm(hidden))
        self.tokens = nn.Sequential(nn.Linear(26, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.lag_query = nn.Linear(hidden, hidden, bias=False)
        self.lag_key = nn.Linear(hidden, hidden, bias=False)
        self.lag_value = nn.Linear(hidden, hidden, bias=False)
        self.edge_query = nn.Linear(hidden, hidden, bias=False)
        self.edge_key = nn.Linear(hidden, hidden, bias=False)
        self.edge_value = nn.Linear(hidden, hidden, bias=False)
        self.update = nn.Sequential(nn.Linear(hidden*2, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden))
        self.output = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 48))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, own, tokens, node):
        state = self.own(torch.cat([own, self.node(node)], -1))
        bank = self.tokens(tokens)  # sample, neighbor, lag, hidden
        lag_weight = torch.einsum('bd,bkld->bkl', self.lag_query(state), self.lag_key(bank))/state.shape[-1]**.5
        edges = (lag_weight.softmax(-1)[..., None]*self.lag_value(bank)).sum(2)
        edge_weight = torch.einsum('bd,bkd->bk', self.edge_query(state), self.edge_key(edges))/state.shape[-1]**.5
        context = (edge_weight.softmax(-1)[..., None]*self.edge_value(edges)).sum(1)
        state = state+self.update(torch.cat([state, context], -1))
        return self.output(state)