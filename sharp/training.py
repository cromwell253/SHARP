"""Node-wise training kernels for the two SHARP correction stages.

Configure RUN and CONFIG_HASH before calling train_stage. Frozen base models
and split arrays are supplied by the caller; this module never loads test data.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
RUN=Path('runs/training')
CONFIG_HASH=''
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_checkpoint(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);torch.save(v,p)

def metrics(pred, target, std):
    error = (pred.astype(np.float64)-target.astype(np.float64))*std
    def pair(e):
        return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt(np.square(e).mean()))}
    return {'overall': pair(error), 'by_lead': {str(k): pair(error[:, k-1]) for k in (12, 24, 48)}}

def context_features(history, level_mean, level_scale):
    relative = history-history[..., -1:]
    amplitude = np.maximum(np.sqrt(np.mean(relative.astype(np.float64)**2, -1)), .001)
    normalized = relative/amplitude[..., None]
    level = (history[..., -1]-level_mean)/level_scale
    # 12 shape values plus last level, mean level, amplitude and end-to-end slope.
    return np.concatenate([normalized, level[..., None],
        ((history.mean(-1)-level_mean)/level_scale)[..., None],
        np.log(amplitude)[..., None], (relative[..., 0]/amplitude)[..., None]], -1)

def own_features(x, forecast):
    h = x[..., 0].transpose(0, 2, 1)
    return np.concatenate([h-h[..., -1:], forecast.transpose(0, 2, 1)-h[..., -1:]], -1).astype(np.float32)

def token_batch(histories, indices, nodes, neighbors, distances, mode, history_mean, history_scale, level_scale):
    # Everything is from observed12-step history; no future targets enter tokens.
    chosen = neighbors[nodes] if mode == 'lagged_neighbors' else nodes[:, None].expand(-1, 4)
    history = histories[indices[:, None], chosen]
    relative = history-history[..., -1:]
    level = (history[..., -1]-histories[indices, nodes, -1][:, None])/level_scale
    tokens = []
    for lag in (0, 1, 2, 3):
        shifted = torch.zeros_like(relative)
        shifted[..., lag:] = relative[..., :12-lag]
        mask = torch.zeros_like(relative); mask[..., lag:] = 1
        normalized = (shifted-history_mean)/history_scale
        normalized = normalized*mask
        tokens.append(torch.cat([normalized, mask, level[..., None], distances[nodes, :, None]], -1))
    return torch.stack(tokens, 2)

def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def evaluate_nodes(net,forward,histories,features,nodes,base,target,std,output_scale):
    pieces=[];net.eval()
    with torch.inference_mode():
        for flat in torch.arange(len(histories)*nodes,device=histories.device).split(4096):
            w,n=flat//nodes,flat%nodes;pieces.append(forward(net,histories,features,w,n).cpu().numpy())
    delta=np.concatenate(pieces).reshape(len(histories),nodes,48).transpose(0,2,1)*output_scale
    prediction=base+delta
    return metrics(prediction,target,std),prediction

def train_stage(alias,seed,stage,net,forward,histories,features,base,targets,std,normalization,config,provenance):
    stage_config=config[stage];job=RUN/stage/'jobs'/alias/f'seed{seed}';job.mkdir(parents=True,exist_ok=True)
    if (job/'result.json').exists():
        result=read_json(job/'result.json');assert result['checkpoint_sha']==sha(job/'best.pt')
        assert result['manifest_sha']==CONFIG_HASH
        ck=torch.load(job/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model'])
        return result,ck
    nodes=histories['train'].shape[1];device=histories['train'].device
    baseline=metrics(base['val'],targets['val'],std)
    target=torch.as_tensor((targets['train']-base['train']).transpose(0,2,1).copy()/stage_config['output_scale'],device=device)
    opt=torch.optim.Adam(net.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    best,best_score,best_epoch,bad=cpu_state(net),baseline,0,0
    history=[];epoch=0;elapsed=0.;last=job/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        assert saved['manifest_sha']==CONFIG_HASH
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        best={k:v.cpu() for k,v in saved['best'].items()}
        best_score,best_epoch,bad=saved['best_score'],saved['best_epoch'],saved['bad']
        history,epoch,elapsed=saved['history'],saved['epoch'],saved['elapsed']
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device) if device.type=='cuda' else None
        print('RESUME',alias,seed,stage,epoch,flush=True)
    started=time.monotonic()
    stopped=epoch>=stage_config['minimum_epochs'] and bad>=stage_config['patience']
    for epoch in range(epoch+1,stage_config['epochs']+1 if not stopped else epoch+1):
        net.train()
        order=torch.randperm(len(histories['train'])*nodes,device=device)
        loss_sum=0.;count=0
        for flat in order.split(stage_config['batch']):
            w,n=flat//nodes,flat%nodes
            predicted=forward(net,histories['train'],features['train'],w,n)
            loss=(predicted-target[w,n]).abs().mean()
            if stage=='router':loss=loss/.01
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),config['clip'],error_if_nonfinite=True);opt.step()
            loss_sum+=loss.item()*len(flat);count+=len(flat)
        score,_=evaluate_nodes(net,forward,histories['val'],features['val'],nodes,base['val'],targets['val'],std,stage_config['output_scale'])
        if score['overall']['mae']<best_score['overall']['mae']:
            best,best_score,best_epoch,bad=cpu_state(net),score,epoch,0
        else:bad+=1
        history.append({'epoch':epoch,'train_loss':loss_sum/count,'val_metrics':score})
        save_checkpoint(last,{'model':net.state_dict(),'optimizer':opt.state_dict(),'best':best,'best_score':best_score,
            'best_epoch':best_epoch,'bad':bad,'history':history,'epoch':epoch,'elapsed':elapsed+time.monotonic()-started,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.empty(0,dtype=torch.uint8),'manifest_sha':CONFIG_HASH})
        write_json(job/'progress.json',{'epoch':epoch,'best_epoch':best_epoch,'best_metrics':best_score})
        print('EPOCH',alias,seed,stage,epoch,'best',best_epoch,score['overall'],flush=True)
        if epoch>=stage_config['minimum_epochs'] and bad>=stage_config['patience']:break
    net.load_state_dict(best)
    ck={'model':best,'nodes':nodes,**normalization,'provenance':provenance,'manifest_sha':CONFIG_HASH}
    torch.save(ck,job/'best.pt')
    result={'status':'PASS','dataset':alias,'seed':seed,'stage':stage,'best_epoch':best_epoch,'epochs_completed':epoch,
        'val_metrics':best_score,'base_metrics':baseline,'history':history,'seconds':elapsed+time.monotonic()-started,
        'test_access':False,'checkpoint_sha':sha(job/'best.pt'),'manifest_sha':CONFIG_HASH}
    write_json(job/'result.json',result);print('STAGE_COMPLETE',alias,seed,stage,flush=True)
    return result,ck
