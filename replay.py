"""One-minute frozen-model replay under training-only threshold scenarios."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sharp.inference import load_model
from sharp.data import read_series,starts,batches,metric_pair

def event_metrics(prediction,raw,target_starts,threshold):
    future=np.stack([raw[t:t+5] for t in target_starts]);truth=(future>threshold).any(1)
    alarm=(prediction>threshold).any(1);available={int(t):i for i,t in enumerate(target_starts)}
    tp=int((truth&alarm).sum());fp=int((~truth&alarm).sum());fn=int((truth&~alarm).sum());tn=int((~truth&~alarm).sum())
    current=raw[target_starts-1]>threshold;pre=~current
    pt=int((truth&alarm&pre).sum());pf=int((~truth&alarm&pre).sum())
    events=[]
    for node in range(raw.shape[1]):
        above=raw[:,node]>threshold[node]
        for onset in np.flatnonzero(above[1:]&~above[:-1])+1:
            opportunities=list(range(int(onset)-4,int(onset)+1))
            if not all(t in available for t in opportunities):continue
            valid=[t for t in opportunities if not above[t-1] and alarm[available[t],node]]
            lead=int(onset-min(valid)+1) if valid else None
            events.append({'node':node,'onset_index':int(onset),'detected':bool(valid),'lead_minutes':lead})
    leads=[e['lead_minutes'] for e in events if e['detected']]
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'window_precision':tp/(tp+fp) if tp+fp else None,
      'pre_onset_window_precision':pt/(pt+pf) if pt+pf else None,
      'event_count':len(events),'event_hits':len(leads),'event_recall':len(leads)/len(events) if events else None,
      'lead_median_minutes':float(np.median(leads)) if leads else None,'events':events}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='JAPD');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--variant',default='SHARP',choices=['SHARP','PatchSTG']);p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='runs/replay.json');p.add_argument('--limit',type=int);a=p.parse_args()
    root=Path(__file__).resolve().parent;torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    predictor=load_model(root,a.dataset,a.seed,a.variant,a.device);raw=read_series(root,a.dataset)
    bounds=starts(len(raw),'test',5);ids=np.arange(bounds[0],bounds[-1]+1);ids=ids[:a.limit] if a.limit else ids
    pred=[]
    for _,x,_ in batches(raw,ids):pred.append(predictor.predict(x)[:,:5])
    prediction=np.concatenate(pred);training=raw[:int(len(raw)*.6)]
    result={'dataset':a.dataset,'variant':a.variant,'seed':a.seed,'windows':len(ids),'thresholds':{}}
    for q in [.9,.95,.99]:
        threshold=np.quantile(training,q,axis=0)
        result['thresholds'][str(q)]=event_metrics(prediction,raw,ids,threshold)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    np.savez_compressed(out.with_suffix('.npz'),prediction=prediction,target_start=ids)
    print(out)
if __name__=='__main__':main()
