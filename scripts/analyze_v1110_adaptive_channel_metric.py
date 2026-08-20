#!/usr/bin/env python3
"""V11.10 adaptive channel metric benchmark.

Learns one positive weight per full-body channel from TRAINING episodes only.
A channel is weighted by how strongly its trajectory distance separates pairs
with aligned future recovery directions from pairs with conflicting directions.
No holdout row is used during weight learning or memory training.
"""
from __future__ import annotations
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory.adaptive_channel_trajectory_memory import AdaptiveChannelTrajectoryMemory
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

def corrective_examples(eps,runs,horizon):
 out=[]
 for run in runs:
  for hist,fut in windows(eps[run],5,horizon):
   old=stability_score(bal(hist[0]),1.);now=stability_score(bal(hist[-1]),1.)
   if old-now<.01:continue
   scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));gain=float(scores[j]-now)
   if gain<.03:continue
   out.append((np.stack([vec(x) for x in hist]),vec(fut[j]),gain))
 return out

def robust_scales(vectors):
 x=np.asarray(vectors,float);med=np.median(x,axis=0);mad=np.median(np.abs(x-med),axis=0);q25,q75=np.percentile(x,[25,75],axis=0)
 robust=np.maximum(1.4826*mad,(q75-q25)/1.349);std=np.std(x,axis=0);floor=np.maximum(1e-4,.05*np.where(std>1e-8,std,1.));return np.maximum(robust,floor)

def learn_weights(examples,scales,pairs=60000,seed=11710):
 rng=np.random.default_rng(seed);n=len(examples)
 if n<20:raise SystemExit('not enough corrective examples to learn channel weights')
 traj=np.asarray([x[0] for x in examples]);cur=traj[:,-1,:];target=np.asarray([x[1] for x in examples]);corr=(target-cur)/scales[None,:]
 i=rng.integers(0,n,size=pairs);j=rng.integers(0,n,size=pairs);mask=i!=j;i=i[mask];j=j[mask]
 # Outcome similarity is defined in normalized full-body correction space.
 ci=corr[i];cj=corr[j];ni=np.linalg.norm(ci,axis=1);nj=np.linalg.norm(cj,axis=1);den=ni*nj;sim=np.zeros(len(i));ok=den>1e-12;sim[ok]=np.sum(ci[ok]*cj[ok],axis=1)/den[ok]
 pos=sim>=.70;neg=sim<=.20
 # Channelwise trajectory RMS for each sampled pair.
 dz=(traj[i]-traj[j])/scales[None,None,:];dch=np.sqrt(np.mean(dz*dz,axis=1))
 if np.count_nonzero(pos)<100 or np.count_nonzero(neg)<100:raise SystemExit('insufficient positive/negative pair separation')
 pmed=np.median(dch[pos],axis=0);nmed=np.median(dch[neg],axis=0)
 # Good discriminators keep aligned outcomes close and conflicting outcomes far.
 raw=(nmed+1e-4)/(pmed+1e-4);raw=np.clip(raw,.35,2.50);raw=raw/np.mean(raw)
 return raw,dict(pairs=int(len(i)),positive=int(np.count_nonzero(pos)),negative=int(np.count_nonzero(neg)))

def train(memory,examples,progress,label):
 for k,(hist,target,gain) in enumerate(examples,1):
  memory.observe(hist,target,gain)
  if progress and k%progress==0:print(f'[V11.10 {label}] corrective={k} proto={memory.size}',flush=True)

def evaluate(memory,eps,runs,horizon):
 frozen=memory.stats().copy();hold=rec=0;cs=[];rms=[];conf=[]
 for run in runs:
  for hist,fut in windows(eps[run],5,horizon):
   hold+=1;cur=vec(hist[-1]);scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));true=vec(fut[j])
   rr=memory.recall([vec(x) for x in hist],min_confidence=.40)
   if rr is None:continue
   rec+=1;conf.append(rr.confidence);rms.append(rr.rms_distance);truth=(true-cur)/memory.scales;pred=(rr.target_state-cur)/memory.scales;cs.append(cosine(pred,truth))
 return dict(hold=hold,rec=rec,coverage=rec/max(1,hold),cos=float(np.mean(cs)) if cs else 0.,aligned=sum(c>=.70 for c in cs)/max(1,len(cs)),rms=float(np.mean(rms)) if rms else 0.,confidence=float(np.mean(conf)) if conf else 0.,frozen=(frozen==memory.stats()),index=memory.index_stats())

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--horizon',type=int,default=12);p.add_argument('--progress-every',type=int,default=500);p.add_argument('--pairs',type=int,default=60000);a=p.parse_args();t=time.time()
 eps,names=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*.70))));tr,te=ids[:split],ids[split:]
 body=sum(str(n).startswith(('joint_pos_deg:','joint_speed_deg_s:')) for n in names)
 if len(names)!=62 or body!=46:raise SystemExit(f'bad schema total={len(names)} body={body}')
 train_vectors=[vec(r) for run in tr for r in eps[run]];examples=corrective_examples(eps,tr,a.horizon);sc=robust_scales(train_vectors);weights,pstat=learn_weights(examples,sc,a.pairs)
 print(f'V11.10 episodes={len(ids)} train={tr} holdout={te} corrective={len(examples)}',flush=True)
 print(f'Weight learning pairs={pstat["pairs"]} positive={pstat["positive"]} negative={pstat["negative"]}',flush=True)
 order=np.argsort(weights)
 print('Lowest-weight channels:');
 for q in order[:8]:print(f'  {names[q]} weight={weights[q]:.4f}')
 print('Highest-weight channels:');
 for q in order[-8:][::-1]:print(f'  {names[q]} weight={weights[q]:.4f}')
 # Baseline recomputed on same code/data for exact comparison.
 base=IndexedFullBodyTrajectoryMemory(context=5);base.fit_scales(train_vectors);print('Training baseline V11.6 metric...',flush=True);train(base,examples,a.progress_every,'BASE')
 weighted=AdaptiveChannelTrajectoryMemory(context=5,channel_weights=weights);weighted.fit_scales(train_vectors);print('Training adaptive weighted metric...',flush=True);train(weighted,examples,a.progress_every,'WEIGHTED')
 rb=evaluate(base,eps,te,a.horizon);rw=evaluate(weighted,eps,te,a.horizon)
 print('\n'+'='*108);print('RoboCOP — V11.10 ADAPTIVE PER-CHANNEL METRIC / FROZEN HOLDOUT');print('='*108)
 print(f"{'variant':18s} {'coverage':>10s} {'recall':>12s} {'cos':>9s} {'>=.70':>9s} {'rms':>9s} {'conf':>9s} {'proto':>8s}")
 for label,m,r in [('baseline',base,rb),('weighted',weighted,rw)]:
  print(f"{label:18s} {100*r['coverage']:9.2f}% {r['rec']:5d}/{r['hold']:<6d} {r['cos']:9.4f} {100*r['aligned']:8.2f}% {r['rms']:9.4f} {r['confidence']:9.4f} {m.size:8d}")
 print(f"\nWeighted memory frozen: {'PASS' if rw['frozen'] else 'FAIL'} | baseline frozen: {'PASS' if rb['frozen'] else 'FAIL'}")
 ws=weighted.weight_stats();print(f"Weights min/mean/max/std: {ws['min']:.4f}/{ws['mean']:.4f}/{ws['max']:.4f}/{ws['std']:.4f}")
 print(f"Weighted candidates/query: {rw['index']['mean_candidates']:.1f}")
 print('Reference V11.7 clean point: coverage=9.28%, cos=0.6863.')
 print(f'Elapsed: {time.time()-t:.1f}s');print('='*108)
if __name__=='__main__':main()
