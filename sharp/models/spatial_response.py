"""Own-history linear projection subtraction and spatial response."""
import torch
from torch import nn


class SpatialResponse(nn.Module):
    def __init__(self, graph, local, projection, gamma=.25):
        super().__init__(); self.graph, self.local, self.gamma = graph, local, gamma
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('projection', torch.as_tensor(projection, dtype=torch.float64))
        nodes = graph.base.node_num
        device = next(graph.parameters()).device
        zero = torch.zeros(1, 12, nodes, 1, device=device)
        te = torch.zeros(1, 12, nodes, 2, dtype=torch.long, device=device)
        te[..., 0] = torch.arange(12, device=device)[None, :, None]
        graph.eval(); local.eval()
        with torch.no_grad():
            reference = graph.encode(zero, zero, te).detach()
        self.register_buffer('reference', reference)

    def train(self, mode=True):
        super().train(mode); self.graph.eval(); self.local.eval(); return self

    def forward(self, x, te):
        expected = torch.arange(12, device=te.device)[None, :, None].expand_as(te[..., 0])
        if not torch.equal(te[..., 0], expected) or torch.count_nonzero(te[..., 1]).item():
            raise ValueError('Cached context requires the fixed relative-position/weekday-zero contract')
        anchor, local = self.local.components(x)
        h = x-x.mean(1, keepdim=True)
        response = self.graph.encode(h, torch.zeros_like(h), te)-self.reference
        phi = x.double()-x[:, -1:].double()
        redundant = torch.einsum('lh,blnc->bhnc', self.projection, phi).to(x.dtype)
        return anchor+local+self.gamma*(response-redundant)