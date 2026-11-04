"""Validation-selected warning calibration for frozen multi-step forecasters."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse,json,csv,hashlib,time,gc
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

DS=['JAPD','WF1','WF13112','WF15212','WF1905','WF2301','WF2903'];SEEDS=[2026,2027,2028];MODELS=['SHARP','PatchSTG'];QS=[.9,.95,.99]
FAMILIES=['margin','bias_margin','logistic','tree'];BUDGETS=[.01,.02,.05]
PERCENTILES=[50,75,90,95,97,98,99,99.3,99.5,99.7,99.8,99.9,99.95,99.97,99.99,100]
PROTO={'name':'frozen_forecast_warning_calibration','primary_quantile':.95,'datasets':DS,'model_seeds':SEEDS,'quantiles':QS,'families':FAMILIES,'primary_endpoint':'macro event F1 over seven datasets after averaging seeds','operating_points':['validation maximum event F1']+BUDGETS,'selection':'One shared family selected by mean validation-second-half q95 F1 over both models, all datasets and seeds. Per-case coefficients fit on validation-first-half only; thresholds use validation-second-half only. Test evaluation begins only after selection.json is frozen.','validation_split':'chronological halves, first-half fitting targets contained before midpoint, second-half threshold selection; five-step target boundary purge','fit_window_stride':5,'max_fit_samples':150000,'node_scale_floor_MPa':.01,'bias_shrinkage_samples':256,'logistic':{'C':1.,'class_weight':'balanced','max_iter':200},'tree':{'max_iter':60,'max_leaf_nodes':7,'min_samples_leaf':50,'learning_rate':.05,'l2_regularization':10.,'max_bins':63,'early_stopping':False,'random_state':20261104},'threshold_score_percentiles':PERCENTILES,'false_alarm_budgets_per_sensor_hour':BUDGETS,'event_protocol':'Existing first-issue alarm-episode one-to-one greedy matching within five minutes; current reading must be at or below the fixed training-only node threshold. Onsets require all five prior prediction opportunities.','test_scope':'Reassessment on the previously examined chronological test split; no test labels used to select calibration family, parameters, or operating point. Not a newly collected blind external test.','history_reference':'fixed logistic history-only detector, identical fitting and threshold selection; reported separately','secondary_forecast_endpoints':['five-minute maximum stress MAE','five-step MAE/RMSE when current stress is within0.5 training-node SD below q95 threshold']}
PROTO['empty_event_selection_fallback']='If the threshold-selection half has no observed onsets, the primary operating point preserves the original direct-threshold rule for both forecasters; no threshold is learned from absence of events. Budget operating points preserve the validation false-alarm constraint; the history-only reference retains its original selection rule. Added from validation-only data before any calibrated test evaluation.'

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def savecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def indices(raw,split):
 lo=int(len(raw)*(.6 if split=='val' else .8));hi=int(len(raw)*(.8 if split=='val' else 1.));first=12+int(np.ceil((lo-12)/5))*5;last=12+int(np.floor((hi-48-12)/5))*5
 return np.arange(first,last+1,dtype=np.int64)
def dataset(repo,ds):return np.load(repo/'data'/ds/'series.npz')['data']
def cachepath(out,ds,seed,model,split):return out/'cache'/split/f'{ds}_{seed}_{model}.npy'
def ensure_prediction(repo,out,ds,seed,model,split,reuse):
 raw=dataset(repo,ds);ids=indices(raw,split);path=cachepath(out,ds,seed,model,split);meta=path.with_suffix('.json');ck=repo/'checkpoints'/ds/f'seed{seed}'/(model+'.pt');key={'checkpoint_sha256':digest(ck),'shape':[len(ids),5,raw.shape[1]],'first_target_start':int(ids[0]),'last_target_start':int(ids[-1])}
 if path.exists() and meta.exists() and json.loads(meta.read_text())['key']==key:return
 path.parent.mkdir(parents=True,exist_ok=True);pred=None;source='fresh inference'
 if split=='test':
  alias='PatchSTG_matched' if model=='PatchSTG' else model
  names=[f'{ds}_{seed}_{model}_first5.npz']+([f'{ds}_{alias}_replay_first5.npz'] if seed==2026 else [])
  for base in reuse:
   for name in names:
    p=base/name
    if p.exists():
     with np.load(p) as z:
      if 'target_start_index' in z:assert np.array_equal(z['target_start_index'],ids)
      pred=z['prediction'];assert list(pred.shape)==key['shape'];source='existing frozen-checkpoint test outputs';break
   if pred is not None:break
 if pred is None:
  import torch,sys
  sys.path.insert(0,str(repo));from sharp.inference import load_model
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
  m=load_model(repo,ds,seed,model,'cuda' if torch.cuda.is_available() else 'cpu');chunks=[]
  batch=128 if split=='val' else 32
  for off in range(0,len(ids),batch):
   ii=ids[off:off+batch];x=np.stack([raw[s-12:s] for s in ii]);chunks.append(m.predict(x)[:,:5])
  pred=np.concatenate(chunks);del m;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 assert np.isfinite(pred).all();temp=path.with_suffix('.tmp.npy');np.save(temp,pred.astype(np.float32));os.replace(temp,path);dump(meta,{'key':key,'source':source})

def context(raw,ids,q):
 train=raw[:int(len(raw)*.6)].astype(np.float64);std=np.maximum(train.std(0),.01);mu=train.mean(0);thr=np.quantile(train,q,axis=0);current=raw[ids-1];pre=current<=thr
 future=np.stack([raw[s:s+5] for s in ids]);onsets=[]
 for n in range(raw.shape[1]):
  above=raw[:,n]>thr[n];u=np.flatnonzero(above[1:]&~above[:-1])+1;onsets.append(u[(u>=ids[0]+4)&(u<=ids[-1])])
 return {'std':std,'mu':mu,'threshold':thr,'current':current,'pre':pre,'future':future,'onsets':onsets,'ids':ids,'hours':len(ids)*raw.shape[1]/60}

def features(raw,ids,pred,c):
 h=np.stack([raw[s-12:s] for s in ids]);cur=c['current'];s=c['std'];th=c['threshold'];pmax=pred.max(1)
 columns=[(pmax-th)/s,(pred.mean(1)-th)/s,(pred[:,4]-th)/s,(pred[:,0]-cur)/s,(pred[:,4]-cur)/s,pred.std(1)/s,(pred.min(1)-cur)/s,(cur-th)/s,(cur-h[:,-2])/s,(cur-h[:,-4])/s,(cur-h[:,-7])/s,h.std(1)/s,(cur-h.mean(1))/s,(pred[:,4]-pred[:,0])/s,np.ptp(h,axis=1)/s,(cur-c['mu'])/s]
 x=np.stack(columns,-1).astype(np.float32);np.clip(x,-20,20,out=x);return x
HISTORY_COLUMNS=[7,8,9,10,11,12,14,15]

def fit_head(family,x,y):
 if len(np.unique(y))<2:return {'kind':'constant','probability':float(y.mean())}
 if family=='logistic':
  scaler=StandardScaler();z=scaler.fit_transform(x);m=LogisticRegression(C=1.,class_weight='balanced',max_iter=200,random_state=20261104);m.fit(z,y);return {'kind':'logistic','scaler':scaler,'model':m}
 counts=np.bincount(y.astype(int),minlength=2);weights=np.where(y,len(y)/(2*counts[1]),len(y)/(2*counts[0]));m=HistGradientBoostingClassifier(**PROTO['tree']);m.fit(x,y,sample_weight=weights);return {'kind':'tree','model':m}
def probability(head,x):
 if head['kind']=='constant':return np.full(len(x),head['probability'],dtype=np.float32)
 if head['kind']=='logistic':x=head['scaler'].transform(x)
 return head['model'].predict_proba(x)[:,1]

def score_metrics(score,tau,c):
 alarm=(score>tau)&c['pre'];first=alarm&~np.vstack([np.zeros((1,alarm.shape[1]),bool),alarm[:-1]]);total=int(first.sum());hits=0;leads=[]
 for n,u in enumerate(c['onsets']):
  if not len(u):continue
  issue=c['ids'][first[:,n]]-1;ix=np.searchsorted(u,issue,side='right');valid=(ix<len(u))&(u[np.minimum(ix,len(u)-1)]<=issue+5);used_until=0
  for t,j0 in zip(issue[valid],ix[valid]):
   j=max(int(j0),used_until)
   if j<len(u) and u[j]<=t+5:hits+=1;leads.append(int(u[j]-t));used_until=j+1
 actual=sum(map(len,c['onsets']));p=hits/total if total else 0.;r=hits/actual if actual else 0.
 return {'actual_events':actual,'alarm_episodes':total,'matched_events':hits,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'false_alarms_per_sensor_hour':(total-hits)/c['hours'],'lead_mean_minutes':float(np.mean(leads)) if leads else None,'lead_median_minutes':float(np.median(leads)) if leads else None}
def thresholds(score,c):
 v=score[c['pre']];return np.unique(np.r_[np.quantile(v,np.asarray(PERCENTILES)/100) if len(v) else [],0,.5,np.inf])
def select_points(score,c,fallback_score=None):
 records=[(float(t),score_metrics(score,float(t),c)) for t in thresholds(score,c)]
 key=lambda x:(x[1]['f1'],-x[1]['false_alarms_per_sensor_hour'],x[1]['recall'],x[0])
 chosen={'f1':max(records,key=key)}
 for b in BUDGETS:
  eligible=[x for x in records if x[1]['false_alarms_per_sensor_hour']<=b+1e-12];chosen[f'budget_{b}']=max(eligible,key=lambda x:(x[1]['recall'],x[1]['precision'],-x[1]['false_alarms_per_sensor_hour'],x[0]))
 result={k:{'threshold':None if np.isinf(v[0]) else v[0],'validation_metrics':v[1]} for k,v in chosen.items()}
 if sum(map(len,c['onsets']))==0 and fallback_score is not None:result['f1']={'threshold':0.,'validation_metrics':score_metrics(fallback_score,0,c),'override':'direct_threshold_no_validation_events'}
 return result
def threshold_value(x):return np.inf if x is None else float(x)

def stage_infer(a):
 for ds in DS:
  for seed in SEEDS:
   for model in MODELS:
    t=time.perf_counter();ensure_prediction(a.repo,a.out,ds,seed,model,'val',[]);print('VAL_CACHE',ds,seed,model,round(time.perf_counter()-t,2),flush=True)

def stage_select(a):
 rows=[];casefiles=[]
 for ds in DS:
  raw=dataset(a.repo,ds);ids=indices(raw,'val');mid=int((int(len(raw)*.6)+int(len(raw)*.8))/2);fit_indices=np.flatnonzero(ids+5<=mid)[::5];tune_indices=np.flatnonzero(ids>=mid)
  for seed in SEEDS:
   for model in MODELS:
    pred=np.load(cachepath(a.out,ds,seed,model,'val'),mmap_mode='r')
    for q in QS:
     stem=f'{ds}_{seed}_{model}_{q}';recordpath=a.out/'validation_cases'/(stem+'.json')
     if recordpath.exists():rec=json.loads(recordpath.read_text());rows+=rec['rows'];casefiles.append(str(recordpath.relative_to(a.out)));continue
     t=time.perf_counter();cf=context(raw,ids[fit_indices],q);ct=context(raw,ids[tune_indices],q);xf=features(raw,ids[fit_indices],pred[fit_indices],cf);xt=features(raw,ids[tune_indices],pred[tune_indices],ct)
     y=(cf['future']>cf['threshold']).any(1);train_mask=cf['pre'].reshape(-1);flat=np.flatnonzero(train_mask)
     if len(flat)>150000:flat=flat[np.linspace(0,len(flat)-1,150000).astype(int)]
     target=y.reshape(-1)[flat];ftrain=xf.reshape(-1,16)[flat];ftune=xt.reshape(-1,16);shape=xt.shape[:2]
     residual=(cf['future'].max(1)-pred[fit_indices].max(1))/cf['std'];global_bias=float(residual.mean());bias=(residual.sum(0)+256*global_bias)/(len(residual)+256)
     fitted={'bias':bias,'logistic':fit_head('logistic',ftrain,target),'tree':fit_head('tree',ftrain,target),'history':fit_head('logistic',ftrain[:,HISTORY_COLUMNS],target)}
     scores={'margin':xt[...,0],'bias_margin':xt[...,0]+bias,'logistic':probability(fitted['logistic'],ftune).reshape(shape),'tree':probability(fitted['tree'],ftune).reshape(shape),'history':probability(fitted['history'],ftune[:,HISTORY_COLUMNS]).reshape(shape)}
     selected={};rr=[]
     for family,score in scores.items():
      points=select_points(score,ct,fallback_score=xt[...,0] if family!='history' else None);selected[family]=points
      for op,v in points.items():rr.append({'dataset':ds,'seed':seed,'model':model,'quantile':q,'family':family,'operating_point':op,'threshold':v['threshold'],**v['validation_metrics']})
     hp=a.out/'heads'/(stem+'.joblib');hp.parent.mkdir(exist_ok=True,parents=True);joblib.dump(fitted,hp,compress=3)
     rec={'dataset':ds,'seed':seed,'model':model,'quantile':q,'fit_windows':len(fit_indices),'tune_windows':len(tune_indices),'fit_samples':len(flat),'fit_positive_samples':int(target.sum()),'selected_points':selected,'rows':rr};dump(recordpath,rec);rows+=rr;casefiles.append(str(recordpath.relative_to(a.out)))
     print('SELECT',ds,seed,model,q,'F1', {f:round(selected[f]['f1']['validation_metrics']['f1'],4) for f in FAMILIES},'seconds',round(time.perf_counter()-t,2),flush=True)
     del xf,xt,scores,fitted;gc.collect()
 metrics={family:float(np.mean([r['f1'] for r in rows if r['family']==family and r['operating_point']=='f1' and r['quantile']==.95])) for family in FAMILIES}
 selected=max(FAMILIES,key=lambda f:(metrics[f],-FAMILIES.index(f)));dump(a.out/'selection.json',{'selected_shared_family':selected,'validation_macro_f1_q95':metrics,'protocol_sha256':digest(a.out/'protocol.json'),'validation_cases':casefiles,'selection_uses_test':False});savecsv(a.out/'validation_selection_records.csv',rows);print('FROZEN_SELECTION',selected,metrics,flush=True)

def paired_intervals(rows, family):
 output=[]
 datasets=sorted({r['dataset'] for r in rows})
 for q in QS:
  for op in ['f1']+[f'budget_{b}' for b in BUDGETS]:
   chosen=[r for r in rows if r['quantile']==q and r['method']==family and r['operating_point']==op]
   for metric in ['precision','recall','f1','false_alarms_per_sensor_hour','lead_mean_minutes']:
    delta=[]
    for ds in datasets:
     values={m:[r[metric] for r in chosen if r['dataset']==ds and r['model']==m and r[metric] is not None] for m in MODELS}
     if all(values.values()):delta.append(np.mean(values['SHARP'])-np.mean(values['PatchSTG']))
    delta=np.asarray(delta);rng=np.random.default_rng(20261104)
    boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    output.append({'quantile':q,'operating_point':op,'metric':metric,'mean_difference':float(delta.mean()),'ci_lower':float(np.quantile(boot,.025)),'ci_upper':float(np.quantile(boot,.975)),'datasets_positive':int((delta>0).sum()),'datasets_negative':int((delta<0).sum()),'datasets_tied':int((delta==0).sum()),'datasets_available':len(delta)})
 return output

def stage_test(a):
 selection_path=a.out/'selection.json';selection=json.loads(selection_path.read_text());family=selection['selected_shared_family'];selection_hash=digest(selection_path);rows=[];forecast=[];artifacts={}
 for ds in DS:
  raw=dataset(a.repo,ds);ids=indices(raw,'test')
  for seed in SEEDS:
   for model in MODELS:
    ensure_prediction(a.repo,a.out,ds,seed,model,'test',a.reuse);pred=np.load(cachepath(a.out,ds,seed,model,'test'),mmap_mode='r')
    for q in QS:
     stem=f'{ds}_{seed}_{model}_{q}';rec=json.loads((a.out/'validation_cases'/(stem+'.json')).read_text());heads=joblib.load(a.out/'heads'/(stem+'.joblib'));c=context(raw,ids,q);x=features(raw,ids,pred,c);flat=x.reshape(-1,16);shape=x.shape[:2]
     if family=='margin':score=x[...,0]
     elif family=='bias_margin':score=x[...,0]+heads['bias']
     else:score=probability(heads[family],flat).reshape(shape)
     raw_metrics=score_metrics(x[...,0],0,c);rows.append({'dataset':ds,'seed':seed,'model':model,'quantile':q,'method':'direct_threshold','operating_point':'unmodified',**raw_metrics})
     for op,choice in rec['selected_points'][family].items():rows.append({'dataset':ds,'seed':seed,'model':model,'quantile':q,'method':family,'operating_point':op,**score_metrics(x[...,0] if choice.get('override') else score,threshold_value(choice['threshold']),c)})
     hscore=probability(heads['history'],flat[:,HISTORY_COLUMNS]).reshape(shape);hpoint=rec['selected_points']['history']['f1'];rows.append({'dataset':ds,'seed':seed,'model':model,'quantile':q,'method':'history_only','operating_point':'f1',**score_metrics(hscore,threshold_value(hpoint['threshold']),c)})
     artifacts[stem]={'head':heads.get(family),'bias':heads['bias'] if family=='bias_margin' else None,'selected_points':rec['selected_points'][family],'family':family}
     if q==.95:
      err=pred-c['future'];mask=c['pre']&((c['threshold']-c['current'])<=.5*c['std']);e=err.transpose(0,2,1)[mask];forecast.append({'dataset':ds,'seed':seed,'model':model,'near_threshold_windows':int(mask.sum()),'near_threshold_mae':float(np.abs(e).mean()) if len(e) else None,'near_threshold_rmse':float(np.sqrt(np.square(e).mean())) if len(e) else None,'five_minute_max_mae':float(np.abs(pred.max(1)-c['future'].max(1)).mean())})
     del x;gc.collect()
    print('TEST',ds,seed,model,flush=True)
 assert digest(selection_path)==selection_hash
 savecsv(a.out/'test_case_metrics.csv',rows);savecsv(a.out/'forecast_secondary_metrics.csv',forecast)
 summary=[]
 keys=sorted({(r['quantile'],r['model'],r['method'],r['operating_point']) for r in rows})
 for q,model,method,op in keys:
  rr=[r for r in rows if (r['quantile'],r['model'],r['method'],r['operating_point'])==(q,model,method,op)];assert len(rr)==21
  fields=['precision','recall','f1','false_alarms_per_sensor_hour','lead_mean_minutes'];summary.append({'quantile':q,'model':model,'method':method,'operating_point':op,**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in fields}})
 savecsv(a.out/'test_macro_summary.csv',summary);savecsv(a.out/'paired_effect_intervals.csv',paired_intervals(rows,family));joblib.dump(artifacts,a.out/'selected_calibrators.joblib',compress=3);dump(a.out/'test_manifest.json',{'frozen_selection_sha256':selection_hash,'rows':len(rows),'completed':True,'test_labels_used_for_selection':False})
 print('PRIMARY_RESULTS',json.dumps([r for r in summary if r['quantile']==.95 and r['operating_point'] in ['f1','unmodified']],ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['infer','select','test'],required=True);p.add_argument('--reuse',type=Path,action='append',default=[]);a=p.parse_args();a.repo=a.repo.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 protocol=a.out/'protocol.json'
 if protocol.exists():assert json.loads(protocol.read_text(encoding='utf-8'))==PROTO
 else:dump(protocol,PROTO)
 {'infer':stage_infer,'select':stage_select,'test':stage_test}[a.stage](a)
if __name__=='__main__':main()
