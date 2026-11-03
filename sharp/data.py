"""Chronological windows with the published training-only normalization."""
from pathlib import Path
import numpy as np

def read_series(root,dataset):
    with np.load(Path(root)/'data'/dataset/'series.npz',allow_pickle=False) as z:
        return z['data']

def starts(length,split='test',stride=5):
    train_end,val_end=int(length*.6),int(length*.8)
    s=np.arange(12,length-48+1,stride,dtype=np.int64)
    if split=='train':return s[s+48<=train_end]
    if split=='val':return s[(s>=train_end)&(s+48<=val_end)]
    if split=='test':return s[s>=val_end]
    raise ValueError(split)

def batches(data,indices,batch=32):
    for off in range(0,len(indices),batch):
        s=indices[off:off+batch]
        yield s,np.stack([data[t-12:t] for t in s]),np.stack([data[t:t+48] for t in s])

def metric_pair(error):
    return {'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.square(error).mean()))}

def metrics(error):
    return {'overall':metric_pair(error),'minute5':metric_pair(error[:,4]),
            'first5':metric_pair(error[:,:5]),'by_lead':{str(h):metric_pair(error[:,h-1]) for h in (12,24,48)}}
