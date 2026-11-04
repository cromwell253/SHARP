"""Frozen-checkpoint engineering extension; no fitting or threshold tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,csv,time,gc
import numpy as np
import torch
R=Path(__file__).resolve().parents[2];P=R;O=R/'runs/engineering_extension';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(P))
from sharp.inference import load_model
from sharp.data import starts,batches,read_series,metric_pair
from replay import event_metrics
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
DATA=['JAPD','WF1','WF13112','WF15212','WF1905','WF2301','WF2903']
def replay_ids(length):
 s=starts(length,'test',5);return np.arange(s[0],s[-1]+1)
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def csvsave(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def infer(m,raw,ids):
 out=[]
 for off in range(0,len(ids),32):
  ss=ids[off:off+32];x=np.stack([raw[s-12:s] for s in ss]);out.append(m.predict(x)[:,:5])
 return np.concatenate(out)
def episode_score(pred,raw,ids,thr):
 truth=(np.stack([raw[s:s+5] for s in ids])>thr).any(1)
 alarm=(pred>thr).any(1)&(raw[ids-1]<=thr)
 total=hits=actual=0;lead=[]
 for n in range(raw.shape[1]):
  above=raw[:,n]>thr[n];onsets=np.flatnonzero(above[1:]&~above[:-1])+1
  onsets=onsets[(onsets>=ids[0]+4)&(onsets<=ids[-1])];actual+=len(onsets);used=set()
  first=np.flatnonzero(alarm[:,n]&~np.r_[False,alarm[:-1,n]])
  # One alarm episode uses its first issue only and matches at most one onset.
  for k in first:
   issue=int(ids[k])-1;total+=1
   match=[int(u) for u in onsets if issue<u<=issue+5 and int(u) not in used]
   if match:used.add(match[0]);hits+=1;lead.append(match[0]-issue)
 precision=hits/total if total else 0.;recall=hits/actual if actual else 0.
 return {'alarm_episodes':total,'matched_episodes':hits,'actual_events':actual,'event_precision':precision,'event_recall':recall,'event_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'miss_rate':1-recall if actual else None,'false_alarms_per_sensor_hour':(total-hits)/(len(ids)*raw.shape[1]/60),'lead_mean':float(np.mean(lead)) if lead else None,'lead_median':float(np.median(lead)) if lead else None}
allrows=[];rob=[];candidates=[]
for ds in DATA:
 raw=read_series(P,ds);ids=replay_ids(len(raw));short=starts(len(raw),'test',5);thresholds={q:np.quantile(raw[:int(len(raw)*.6)],q,axis=0) for q in [.9,.95,.99]};saved={}
 for seed in [2026,2027,2028]:
  for variant in ['SHARP','PatchSTG']:
   marker=O/f'{ds}_{seed}_{variant}.json';cache=O/f'{ds}_{seed}_{variant}_first5.npz'
   if marker.exists():
    x=json.loads(marker.read_text());allrows+=x['events'];rob+=x['robustness'];continue
   begin=time.perf_counter();m=load_model(P,ds,seed,variant,'cuda')
   if cache.exists():pred=np.load(cache)['prediction']
   else:pred=infer(m,raw,ids);np.savez_compressed(cache,prediction=pred)
   events=[]
   for q,thr in thresholds.items():
    prior=event_metrics(pred,raw,ids,thr);e=episode_score(pred,raw,ids,thr)
    events.append({'dataset':ds,'seed':seed,'model':variant,'quantile':q,'original_event_count':prior['event_count'],'original_event_recall':prior['event_recall'],'original_lead_median':prior['lead_median_minutes'],**e})
   rs=[]
   # Repeated masks use exactly the same seeded array for both models.
   for condition in ['complete','stale','missing_20261101','missing_20261102','missing_20261103']:
    rng=np.random.default_rng(int(condition.split('_')[1])) if condition.startswith('missing') else None
    errors=[]
    for off in range(0,len(short),128):
     ss=short[off:off+128];x=np.stack([raw[s-12:s] for s in ss]);y=np.stack([raw[s:s+5] for s in ss])
     if condition=='stale':x[:,-2:]=x[:,-3:-2]
     if rng is not None:
      mask=rng.random(x.shape)<.1;mask[:,0]=False
      for h in range(1,12):x[:,h]=np.where(mask[:,h],x[:,h-1],x[:,h])
     pp=m.predict(x)[:,:5];yy=(y-m.config['mean'])/m.config['std'];yy=yy*m.config['std']+m.config['mean'];errors.append(pp-yy)
    rs.append({'dataset':ds,'seed':seed,'model':variant,'condition':condition,**metric_pair(np.concatenate(errors))})
   record={'events':events,'robustness':rs,'seconds':time.perf_counter()-begin};dump(marker,record);allrows+=events;rob+=rs
   del m,pred;gc.collect();torch.cuda.empty_cache();print('COMPLETE',ds,seed,variant,round(record['seconds'],1),flush=True)
csvsave('event_metrics.csv',allrows);csvsave('robustness_metrics.csv',rob)
# Transparent case selection from paired seed2026 forecasts, 95% scenarios.
for ds in DATA:
 raw=read_series(P,ds);ids=replay_ids(len(raw));thr=np.quantile(raw[:int(len(raw)*.6)],.95,axis=0)
 preds={}
 for v,oldn in [('SHARP','SHARP'),('PatchSTG','PatchSTG_matched')]:preds[v]=np.load(O/f'{ds}_2026_{v}_first5.npz')['prediction']
 for node in range(raw.shape[1]):
  above=raw[:,node]>thr[node]
  for onset in np.flatnonzero(above[1:]&~above[:-1])+1:
   if onset<ids[0]+4 or onset>ids[-1]:continue
   positions=np.flatnonzero((ids>=onset-4)&(ids<=onset)&~above[ids-1])
   s=(preds['SHARP'][positions,:,node]>thr[node]).any(1);b=(preds['PatchSTG'][positions,:,node]>thr[node]).any(1)
   if bool(s.any())==bool(b.any()):continue
   group='SHARP_success' if s.any() else 'SHARP_failure';k=positions[np.flatnonzero(s if s.any() else b)[0]];start=int(ids[k]);truth=raw[start:start+5,node]
   mae=float(np.abs(preds['SHARP'][k,:,node]-truth).mean())
   candidates.append({'group':group,'dataset':ds,'node':node,'onset':int(onset),'target_start':start,'threshold':float(thr[node]),'sharp_mae_first5':mae})
selected=[]
for group in ['SHARP_success','SHARP_failure']:
 pool=sorted([x for x in candidates if x['group']==group],key=lambda x:(x['sharp_mae_first5'],DATA.index(x['dataset']),x['onset'],x['node']))
 assert pool,group;selected.append(pool[(len(pool)-1)//2])
dump(O/'case_candidates.json',candidates);dump(O/'selected_cases.json',selected)
for case in selected:
 raw=read_series(P,case['dataset']);s=case['target_start'];n=case['node'];payload={'history':raw[s-12:s,n],'truth':raw[s:s+48,n]}
 for v in ['SHARP','PatchSTG']:
  m=load_model(P,case['dataset'],2026,v,'cuda');payload[v]=m.predict(raw[s-12:s][None])[0,:,n]
 np.savez_compressed(O/(case['group']+'_curves.npz'),**payload)
# Integrated accelerated sequential software replay, with actual stage timings.
for variant in ['SHARP','PatchSTG']:
 ds='WF1';raw=read_series(P,ds);ids=starts(len(raw),'test',1)[:1000];threshold=np.quantile(raw[:int(len(raw)*.6)],.95,axis=0);m=load_model(P,ds,2026,variant,'cuda');buffer=raw[ids[0]-12:ids[0]-1].tolist();timings=[]
 for _ in range(50):m.predict(raw[ids[0]-12:ids[0]][None])
 for s in ids:
  message=json.dumps({'index':int(s-1),'values':raw[s-1].tolist()});torch.cuda.synchronize();t0=time.perf_counter_ns();received=json.loads(message);t1=time.perf_counter_ns();buffer.append(received['values']);buffer=buffer[-12:];history=np.asarray(buffer,np.float32)[None];t2=time.perf_counter_ns();pp=m.predict(history);torch.cuda.synchronize();t3=time.perf_counter_ns();alarms=(pp[0,:5]>threshold).any(0);output=json.dumps({'index':received['index'],'forecast':pp[0,:5].tolist(),'alarms':alarms.tolist()});t4=time.perf_counter_ns()
  timings.append({'target_start':int(s),'decode_ms':(t1-t0)/1e6,'buffer_ms':(t2-t1)/1e6,'prediction_ms':(t3-t2)/1e6,'alarm_encode_ms':(t4-t3)/1e6,'total_ms':(t4-t0)/1e6,'deadline_miss':int(t4-t0>60e9)})
 csvsave(f'pipeline_{variant}.csv',timings)
dump(O/'protocol.json',{'seeds':[2026,2027,2028],'quantiles':[.9,.95,.99],'frozen_models':True,'episode_matching':'Merge consecutive pre-onset alarm windows per node; use the first issue of each predicted alarm segment; greedily match at most one observed onset in the next five minutes; actual onsets require five complete prediction opportunities. No test-threshold adjustment.','false_alarm_exposure':'all evaluated sensor-minutes / 60','case_rule':'seed2026 q95; discordant event detections; lower median SHARP five-minute MAE within each success/failure pool; fixed dataset/onset/node tie break','pipeline':'WF1 seed2026, first1000 consecutive test updates, accelerated RAM JSON input replay; no network or physical sensor transmission; measured elapsed times, virtual one-minute data indices; deadline60s','missing_masks':[20261101,20261102,20261103],'status':'COMPLETE'})
print('ALL COMPLETE',flush=True)
