"""Shared convex history baseline and own-node predictor."""
import torch
from torch import nn


class HistoricalPredictor(nn.Module):
    def __init__(self, anchor_logits, mean, scale):
        super().__init__()
        self.register_buffer('anchor_logits', torch.as_tensor(anchor_logits).detach().clone())
        self.register_buffer('feature_mean', torch.as_tensor(mean, dtype=torch.float64))
        self.register_buffer('feature_scale', torch.as_tensor(scale, dtype=torch.float64))
        d = len(mean)
        self.network = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 48))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def response_from_features(self, f):
        z = ((f.double()-self.feature_mean)/self.feature_scale).float()
        return self.network(z)*.01

    def components(self, x):
        anchor = torch.einsum('hl,blnc->bhnc', self.anchor_logits.softmax(-1), x)
        history = x[..., 0].transpose(1, 2)
        last = history[..., -1:]
        own = torch.cat([history-last, anchor[..., 0].transpose(1, 2)-last], -1)
        identity = torch.eye(x.shape[2], device=x.device, dtype=torch.float64)[None].expand(len(x), -1, -1)
        f = torch.cat([own.double(), identity], -1)
        local = self.response_from_features(f).transpose(1, 2).unsqueeze(-1)
        return anchor, local

    def forward(self, x, te=None):
        anchor, local = self.components(x)
        return anchor+local