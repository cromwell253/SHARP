"""Centered spatial response backbone."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

VARIANTS = ('level_only', 'temporal_difference', 'field_state', 'field_difference')


def stats_key(coordinates):
    return hashlib.sha256(np.asarray(coordinates, dtype=np.float32).tobytes()).hexdigest()[:16]


class SpatialBackbone(nn.Module):
    def __init__(self, nodes, steps, horizon, adjacency, coordinates, variant=VARIANTS[0],
                 hidden=16, device='cpu', native_path=None, partition=None):
        super().__init__(); assert variant in VARIANTS
        self.variant, self.branch, self.disable_field = variant, None, False
        path = Path(native_path)
        # Inactive field buffers are restored by the published state dictionary.
        stats = {'template': [0.0]*nodes, 'scale': 1.0}
        self.register_buffer('field_template', torch.as_tensor(stats['template'], dtype=torch.float32).reshape(1, 1, nodes, 1))
        self.register_buffer('field_scale', torch.tensor(stats['scale'], dtype=torch.float32))
        if variant == 'level_only':
            self.level_logits = nn.Parameter(torch.zeros(horizon, steps)); self.to(device); return
        spec = importlib.util.spec_from_file_location('relative_native_'+stats_key(coordinates), path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        original, restored, reordered = [torch.as_tensor(partition[k], dtype=torch.long, device=device)
                                         for k in ('original', 'restored', 'reordered')]
        self.base = module.PatchSTG(tem_patchsize=12, tem_patchnum=1, node_num=nodes,
            spa_patchsize=partition['size'], spa_patchnum=len(reordered)//partition['size'],
            tod=1440, dow=7, layers=3, factors=1, input_dims=16, node_dims=16, tod_dims=8, dow_dims=8,
            ori_parts_idx=original, reo_parts_idx=restored, reo_all_idx=reordered)
        self.base.regression_conv = nn.Conv2d(48, horizon, (1, 1))
        nn.init.zeros_(self.base.regression_conv.weight); nn.init.zeros_(self.base.regression_conv.bias)
        self.level_logits = nn.Parameter(torch.zeros(horizon, steps))
        self.to(device)

    def encode(self, centered, field, te):
        b, _, n, _ = centered.shape
        x1 = torch.cat([centered, te[..., :1]/self.base.tod, field], -1).float()
        x = self.base.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        x = torch.cat([x, self.base.time_in_day_emb[te[:, -1:, :, 0].long()],
                       self.base.day_in_week_emb[te[:, -1:, :, 1].long()],
                       self.base.node_emb[None, None].expand(b, 1, n, -1)], -1)
        x = x[:, :, self.base.reo_all_idx]
        for block in self.base.spa_encoder:
            x = block(x)
        restored = x.new_zeros(b, 1, n, 48)
        restored[:, :, self.base.ori_parts_idx] = x[:, :, self.base.reo_parts_idx]
        return self.base.regression_conv(restored.transpose(2, 3).reshape(b, 48, n, 1))

    def difference(self, centered, field, te):
        # Replay dropout masks so the difference isolates signal, not random masks.
        cpu_before = torch.get_rng_state() if self.training else None
        cuda_before = torch.cuda.get_rng_state(centered.device) if self.training and centered.is_cuda else None
        positive = self.encode(centered, field, te)
        if self.training:
            cpu_after = torch.get_rng_state(); torch.set_rng_state(cpu_before)
            if centered.is_cuda:
                cuda_after = torch.cuda.get_rng_state(centered.device)
                torch.cuda.set_rng_state(cuda_before, centered.device)
        reference = self.encode(torch.zeros_like(centered), torch.zeros_like(field), te)
        if self.training:
            torch.set_rng_state(cpu_after)
            if centered.is_cuda:
                torch.cuda.set_rng_state(cuda_after, centered.device)
        return positive-reference

    def forward(self, x, te):
        level = torch.einsum('hl,blnc->bhnc', self.level_logits.softmax(-1), x)
        if self.variant == 'level_only':
            return level
        centered = x-x.mean(1, keepdim=True)
        field = (x-x.mean(2, keepdim=True)-self.field_template)/self.field_scale
        if self.variant == 'temporal_difference' or self.disable_field:
            field = torch.zeros_like(field)
        residual = self.encode(centered, field, te) if self.variant == 'field_state' else self.difference(centered, field, te)
        return level+residual


def time_indices(x):
    te = torch.zeros((*x.shape[:3], 2), dtype=torch.long, device=x.device)
    te[..., 0] = torch.arange(x.shape[1], device=x.device)[None, :, None]
    return te