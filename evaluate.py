"""Evaluate released weights; never overwrites archived paper results."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sharp.inference import load_model
from sharp.data import read_series,starts,batches,metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='JAPD');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--variant',default='SHARP',choices=['SHARP','PatchSTG','C0','H','S'])
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int)
    p.add_argument('--output',default='runs/evaluation.json');a=p.parse_args()
    root=Path(__file__).resolve().parent
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    m=load_model(root,a.dataset,a.seed,a.variant,a.device);data=read_series(root,a.dataset)
    ids=starts(len(data),a.split);ids=ids[:a.limit] if a.limit else ids
    errors=[]
    for _,x,y in batches(data,ids):
        prediction=m.predict(x)
        target=(y-m.config['mean'])/m.config['std']
        target=target*m.config['std']+m.config['mean']
        errors.append(prediction-target)
    result={'dataset':a.dataset,'seed':a.seed,'variant':a.variant,'split':a.split,
            'windows':len(ids),'metrics':metrics(np.concatenate(errors))}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
