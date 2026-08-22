#!/usr/bin/env python3
"""V11.7: progressive resolution. Trajectory is authoritative; high-confidence instantaneous recall fills trajectory MISSes."""
from __future__ import annotations
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.transition_memory import BalanceState,stability_score

def load(p):
 e=defaultdict(list);schema=None
 for line in p.open(encoding='utf-8'):
  try:r=json.loads(line)
  except Exception:continue
  fb=r.get('full_body_state');s=r.get('state')
  if not fb or not s:continue
  names=tuple(fb.get('names',[]));v=np.asarray(fb.get('vector',[]),float)
  if schema is None:schema=names
  if names!=schema or len(v)!=len(schema):raise SystemExit('schema mismatch')
  e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items())),schema
def vec(r):return np.asarray(r['full_body_state']['vector'],float)
def bal(r):
 s=r['state'];return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.)))
def windows(rows,c,h):
 for i in range(c-1,len(rows)-h):yield rows[i-c+1:i+1],rows[i+1:i+h+1]
def cosine(a,b):
 na=np.linalg.norm(a);nb=np.linalg.norm(b);return float(np.dot(a,b)/(na*nb)) if na>1e-12 and nb>1e-12 else 0.
def train(eps,runs,context,horizon,progress):
 m=IndexedFullBodyTrajectoryMemory(context=context);m.fit_scales([vec(r) for run in runs for r in eps[run]])
 corr=0
 for run in runs:
  for hist,fut in windows(eps[run],5,horizon):
   old=stability_score(bal(hist[0]),1.);now=stability_score(bal(hist[-1]),1.)
   if old-now<.01:continue
   scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));gain=float(scores[j]-now)
   if gain<.03:continue
   corr+=1;m.observe([vec(x) for x in hist[-context:]],vec(fut[j]),gain)
   if progress and corr%progress==0:print(f'[V11.7 ctx={context}] corrective={corr} proto={m.size}',flush=True)
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--horizon',type=int,default=12);p.add_argument('--progress-every',type=int,default=500);a=p.parse_args();t=time.time()
 eps,names=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*.70))));tr,te=ids[:split],ids[split:]
 body=sum(str(n).startswith(('joint_pos_deg:','joint_speed_deg_s:')) for n in names)
 if len(names)!=62 or body!=46:raise SystemExit(f'bad V11.5 schema total={len(names)} body={body}')
 print(f'V11.7 episodes={len(ids)} train={tr} holdout={te}',flush=True)
 print('Training fine trajectory memory...',flush=True);fine=train(eps,tr,5,a.horizon,a.progress_every)
 print('Training coarse instantaneous memory...',flush=True);coarse=train(eps,tr,1,a.horizon,a.progress_every)
 frozen_f=fine.stats().copy();frozen_c=coarse.stats().copy();rows=[]
 for run in te:
  for hist,fut in windows(eps[run],5,a.horizon):
   cur=vec(hist[-1]);scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));true=vec(fut[j]);truth=(true-cur)/fine.scales
   rf=fine.recall([vec(x) for x in hist],min_confidence=.40);rc=coarse.recall([cur],min_confidence=.40)
   cf=cosine((rf.target_state-cur)/fine.scales,truth) if rf else None
   cc=cosine((rc.target_state-cur)/coarse.scales,(true-cur)/coarse.scales) if rc else None
   rows.append((rf,rc,cf,cc))
 print('\n'+'='*104);print('RoboCOP — V11.7 PROGRESSIVE RESOLUTION / FROZEN HOLDOUT');print('='*104)
 print(f'Fine prototypes: {fine.size} | coarse prototypes: {coarse.size} | holdout windows: {len(rows)}')
 print(f"Memory frozen: {'PASS' if frozen_f==fine.stats() and frozen_c==coarse.stats() else 'FAIL'}")
 print('\nPolicy: trajectory first; instantaneous is fallback only on trajectory MISS.')
 print(f"{'fallback conf':>13} {'coverage':>10} {'fine':>7} {'fallback':>9} {'cos mean':>10} {'>=.70':>9}")
 for th in (.40,.50,.60,.65,.70,.75,.80,.85,.90):
  cs=[];nf=nb=0
  for rf,rc,cf,cc in rows:
   if rf is not None:nf+=1;cs.append(cf)
   elif rc is not None and rc.confidence>=th:nb+=1;cs.append(cc)
  cov=len(cs)/max(1,len(rows));aligned=sum(c>=.70 for c in cs)/max(1,len(cs));mean=float(np.mean(cs)) if cs else 0.
  print(f'{th:13.2f} {100*cov:9.2f}% {nf:7d} {nb:9d} {mean:10.4f} {100*aligned:8.2f}%')
 print('\nReference V11.6 fine-only target: coverage=8.35%, cos=0.6829, >=.70=68.32%')
 print(f'Elapsed: {time.time()-t:.1f}s');print('='*104)
if __name__=='__main__':main()
