"""Load published SHARP weights and predict in the original stress units."""
from pathlib import Path
import json
import numpy as np
import torch
from .models.backbone import SpatialBackbone, time_indices
from .models.local_predictor import HistoricalPredictor
from .models.spatial_response import SpatialResponse
from .models.history_attention import HistorySelectiveAttention
from .models.spatial_correction import SpatialIncrementCorrection
from .models.sharp import SHARP
from .models.patchstg import PatchSTG


class MatchedPatchSTG(torch.nn.Module):
    def __init__(self, nodes, partition):
        super().__init__()
        ids=[torch.tensor(partition[k],dtype=torch.long) for k in ('original','restored','reordered')]
        self.base=PatchSTG(12,1,nodes,partition['size'],len(ids[2])//partition['size'],1440,7,
                          1,1,8,8,4,4,*ids)
        self.base.regression_conv=torch.nn.Conv2d(24,48,(1,1))
    def forward(self,x,te):
        return self.base(x,te)


def load_model(root, dataset='JAPD', seed=2026, variant='SHARP', device='cpu'):
    root=Path(root)
    ck=torch.load(root/'checkpoints'/dataset/f'seed{seed}'/(variant+'.pt'),
                  map_location='cpu',weights_only=True)
    cfg=json.loads((root/'configs'/'geometry'/f'{dataset}.json').read_text())
    state=ck['state_dict'];nodes=ck['nodes']
    if variant=='PatchSTG':
        net=MatchedPatchSTG(nodes,cfg['matched_partition'])
    else:
        graph=SpatialBackbone(nodes,12,48,None,np.zeros((nodes,2)),
              'temporal_difference',native_path=Path(__file__).parent/'models/patchstg.py',partition=cfg['partition'])
        local=HistoricalPredictor(state['base.local.anchor_logits'],state['base.local.feature_mean'],state['base.local.feature_scale'])
        graph.load_state_dict({k.removeprefix('base.graph.'):v for k,v in state.items() if k.startswith('base.graph.')})
        base=SpatialResponse(graph,local,state['base.projection'],gamma=ck['gamma'])
        router=HistorySelectiveAttention(nodes,state['router.base_logits'])
        adapter=SpatialIncrementCorrection(nodes)
        rc={k:state[k] for k in ('level_mean','level_scale','context_mean','context_scale')}
        lc={k:state[k] for k in ('own_mean','own_scale','neighbors','distances')}
        lc.update(mode=ck['mode'],level_scale=float(state['neighbor_level_scale']))
        net=SHARP(base,router,adapter,rc,lc)
    net.load_state_dict(state,strict=True)
    net.to(device).eval()
    return Predictor(net,ck)


class Predictor:
    def __init__(self,model,config):
        self.model,self.config=model,config
    def predict(self,history):
        x=np.asarray(history,dtype=np.float32)
        if x.ndim!=3 or x.shape[1:]!=(12,self.config['nodes']) or not np.isfinite(x).all():
            raise ValueError('history must be finite [batch,12,nodes] in physical units')
        x=(x-self.config['mean'])/self.config['std']
        z=torch.from_numpy(np.ascontiguousarray(x[...,None])).to(next(self.model.parameters()).device)
        with torch.inference_mode():y=self.model(z,time_indices(z))[...,0]
        return y.cpu().numpy()*self.config['std']+self.config['mean']
