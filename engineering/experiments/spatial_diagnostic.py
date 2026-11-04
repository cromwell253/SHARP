"""Descriptive held-out correlations for geometric four-neighbor pairs."""
from pathlib import Path
import argparse,csv
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();P=Path(a.repo);rows=[]
for ds in ['JAPD','WF1','WF13112','WF15212','WF1905','WF2301','WF2903']:
 raw=np.load(P/'data'/ds/'series.npz')['data'];x=raw[int(len(raw)*.8):].astype(np.float64);dx=np.diff(x,axis=0)
 with open(P/'data'/ds/'coordinates.csv',encoding='utf-8-sig') as f:
  records=list(csv.DictReader(f));keys=[k for k in records[0] if k.lower() in ['x','y','z','x_m','y_m','z_m']];coordinates=np.array([[float(r[k]) for k in keys] for r in records])
 distances=((coordinates[:,None]-coordinates[None])**2).sum(2);np.fill_diagonal(distances,np.inf)
 neighbors=distances.argsort(1)[:,:4];near={(min(i,int(j)),max(i,int(j))) for i in range(len(coordinates)) for j in neighbors[i]};other={(i,j) for i in range(len(coordinates)) for j in range(i+1,len(coordinates))}-near
 row={'dataset':ds,'test_points':len(x),'neighbor_pairs':len(near),'other_pairs':len(other)}
 for kind,array in [('level',x),('increment',dx)]:
  with np.errstate(divide='ignore',invalid='ignore'):corr=np.corrcoef(array,rowvar=False)
  for name,pairs in [('neighbors',near),('other_pairs',other)]:
   values=np.array([corr[i,j] for i,j in sorted(pairs)]);finite=values[np.isfinite(values)];row[f'{kind}_correlation_{name}']=float(finite.mean());row[f'{kind}_valid_{name}']=len(finite)
 row['equal_consecutive_fraction']=float((dx==0).mean());rows.append(row)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(out)
