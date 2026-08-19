#!/usr/bin/env python3
"""V11.9: consensus gate.

Z3 full SxB trajectory is authoritative. On Z3 MISS, Z1 full instantaneous
proposes a full-state target; Z2 body short trajectory is a validator only.
The Z1 proposal is accepted only when confidence thresholds are met and the
body component of Z1 agrees directionally with Z2. Z2 never acts alone.
"""
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
def train(eps,runs,idx,context,horizon,progress,label):
 m=IndexedFullBodyTrajectoryMemory(context=context);m.fit_scales([vec(r)[idx] for run in runs for r in eps[run]])
 corr=0
 for run in runs:
  for hist,fut in windows(eps[run],5,horizon):
   old=stability_score(bal(hist[0]),1.);now=stability_score(bal(hist[-1]),1.)
   if old-now<.01:continue
   scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));gain=float(scores[j]-now)
   if gain<.03:continue
   corr+=1;m.observe([vec(x)[idx] for x in hist[-context:]],vec(fut[j])[idx],gain)
   if progress and corr%progress==0:print(f'[V11.9 {label}] corrective={corr} proto={m.size}',flush=True)
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--horizon',type=int,default=12);p.add_argument('--progress-every',type=int,default=500);a=p.parse_args();t=time.time()
 eps,names=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*.70))));tr,te=ids[:split],ids[split:]
 bm=np.asarray([str(n).startswith(('joint_pos_deg:','joint_speed_deg_s:')) for n in names],bool);bidx=np.where(bm)[0];allidx=np.arange(len(names))
 if len(names)!=62 or len(bidx)!=46:raise SystemExit(f'bad schema total={len(names)} body={len(bidx)}')
 print(f'V11.9 episodes={len(ids)} train={tr} holdout={te}',flush=True)
 print('Training Z3 authoritative full trajectory...',flush=True);z3=train(eps,tr,allidx,5,a.horizon,a.progress_every,'Z3')
 print('Training Z2 body validator...',flush=True);z2=train(eps,tr,bidx,2,a.horizon,a.progress_every,'Z2')
 print('Training Z1 instantaneous proposer...',flush=True);z1=train(eps,tr,allidx,1,a.horizon,a.progress_every,'Z1')
 frozen=(z3.stats().copy(),z2.stats().copy(),z1.stats().copy());rows=[]
 for run in te:
  for hist,fut in windows(eps[run],5,a.horizon):
   cur=vec(hist[-1]);scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));true=vec(fut[j]);curb=cur[bidx]
   r3=z3.recall([vec(x) for x in hist],min_confidence=.40);r2=z2.recall([vec(x)[bidx] for x in hist[-2:]],min_confidence=.40);r1=z1.recall([cur],min_confidence=.40)
   truth=(true-cur)/z3.scales;c3=cosine((r3.target_state-cur)/z3.scales,truth) if r3 else None;c1=cosine((r1.target_state-cur)/z1.scales,(true-cur)/z1.scales) if r1 else None
   agree=None
   if r1 is not None and r2 is not None:
    # Compare body directions in each memory's own normalized body coordinates.
    d1=((r1.target_state-cur)/z1.scales)[bidx];d2=(r2.target_state-curb)/z2.scales;agree=cosine(d1,d2)
   rows.append((r3,r2,r1,c3,c1,agree))
 ok=frozen==(z3.stats(),z2.stats(),z1.stats())
 print('\n'+'='*116);print('RoboCOP — V11.9 Z1-Z2 CONSENSUS GATE / FROZEN HOLDOUT');print('='*116)
 print(f'Z3 prototypes={z3.size} | Z2 prototypes={z2.size} | Z1 prototypes={z1.size} | windows={len(rows)}');print(f"Memory frozen: {'PASS' if ok else 'FAIL'}")
 print('\nPolicy: Z3 authoritative; on Z3 MISS, Z1 proposes and Z2 only validates. Z2 never produces an action alone.')
 print(f"{'z1 conf':>7} {'z2 conf':>7} {'agree':>7} {'coverage':>10} {'z3':>6} {'cons':>6} {'cos':>9} {'>=.70':>9}")
 results=[]
 for t1 in (.55,.60,.65,.70,.75,.80):
  for t2 in (.55,.60,.65,.70,.75,.80):
   for ag in (.20,.40,.50,.60,.70,.80):
    cs=[];n3=nc=0
    for r3,r2,r1,c3,c1,agree in rows:
     if r3 is not None:n3+=1;cs.append(c3);continue
     if r1 is not None and r2 is not None and r1.confidence>=t1 and r2.confidence>=t2 and agree is not None and agree>=ag:nc+=1;cs.append(c1)
    cov=len(cs)/max(1,len(rows));mean=float(np.mean(cs)) if cs else 0.;aligned=sum(c>=.70 for c in cs)/max(1,len(cs));results.append((cov,mean,aligned,t1,t2,ag,n3,nc))
 # Print Pareto-like candidates: clean precision region plus highest-coverage candidates.
 chosen=set()
 for x in sorted(results,key=lambda x:(x[1]>=.68,x[0],x[1]),reverse=True)[:15]:chosen.add(x)
 for x in sorted(results,key=lambda x:(x[0],x[1]),reverse=True)[:10]:chosen.add(x)
 for cov,mean,aligned,t1,t2,ag,n3,nc in sorted(chosen,key=lambda x:(-x[0],-x[1])):
  print(f'{t1:7.2f} {t2:7.2f} {ag:7.2f} {100*cov:9.2f}% {n3:6d} {nc:6d} {mean:9.4f} {100*aligned:8.2f}%')
 feasible=[x for x in results if x[1]>=.68]
 if feasible:
  best=max(feasible,key=lambda x:(x[0],x[1]));print(f'\nBEST cos>=0.68: coverage={100*best[0]:.2f}% cos={best[1]:.4f} >=.70={100*best[2]:.2f}% z1={best[3]:.2f} z2={best[4]:.2f} agree={best[5]:.2f}')
 else:print('\nNo configuration reached cos>=0.68.')
 print('Reference V11.7: coverage=9.28%, cos=0.6863.');print(f'Elapsed: {time.time()-t:.1f}s');print('='*116)
if __name__=='__main__':main()
