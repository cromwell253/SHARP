"""Local frozen-model monitoring prototype for a complete UI latency check."""
from pathlib import Path
import argparse,json,time,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
import torch

ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
P=Path(args.repo).resolve();O=Path(args.out).resolve();O.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(P))
from sharp.inference import load_model
from sharp.data import read_series,starts
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
raw=read_series(P,'WF1');ids=starts(len(raw),'test',1)[:1000];threshold=np.quantile(raw[:int(len(raw)*.6)],.95,axis=0)
device='cuda' if torch.cuda.is_available() else 'cpu';models={};buffers={}
for name in ['SHARP','PatchSTG']:
 models[name]=load_model(P,'WF1',2026,name,device)
 for _ in range(50):models[name].predict(raw[ids[0]-12:ids[0]][None])
 buffers[name]=raw[ids[0]-12:ids[0]-1].tolist()
setup={'dataset':'WF1','nodes':raw.shape[1],'history':raw[ids[0]-12:ids[0]-1].tolist(),'updates':[{'index':int(s-1),'values':raw[s-1].tolist()} for s in ids],'threshold':threshold.tolist()}
html=Path(__file__).with_name('ui_monitor_ui.html').read_bytes()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def respond(self,payload,kind='application/json'):
  b=payload if isinstance(payload,bytes) else json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  if self.path=='/setup':self.respond(setup)
  else:self.respond(html,'text/html; charset=utf-8')
 def do_POST(self):
  x=json.loads(self.rfile.read(int(self.headers['Content-Length'])));name=x['model']
  if self.path=='/reset':buffers[name]=raw[ids[0]-12:ids[0]-1].tolist();self.respond({'ok':True});return
  buffers[name].append(x['values']);buffers[name]=buffers[name][-12:];history=np.asarray(buffers[name],np.float32)[None]
  t0=time.perf_counter_ns();pred=models[name].predict(history)
  if device=='cuda':torch.cuda.synchronize()
  model_ms=(time.perf_counter_ns()-t0)/1e6
  self.respond({'index':x['index'],'forecast':pred[0,:5].tolist(),'alarms':(pred[0,:5]>threshold).any(0).tolist(),'model_ms':model_ms})
server=HTTPServer(('127.0.0.1',0),Handler)
ready={'port':server.server_address[1],'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'torch':torch.__version__,'dataset':'WF1','seed':2026,'warmups':50,'updates_per_model':1000,'first_target_start':int(ids[0]),'last_target_start':int(ids[-1]),'scope':'Local monitoring prototype: loopback JSON input, buffer, normalization and48-step inference, first5 threshold checks, JSON output, canvas rendering and double requestAnimationFrame. Accelerated replay; no physical sensor or mine network.'}
(O/'backend_ready.json').write_text(json.dumps(ready,indent=2),encoding='utf-8');print(json.dumps(ready),flush=True);server.serve_forever()
