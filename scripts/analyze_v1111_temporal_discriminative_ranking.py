#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.temporal_discriminative_trajectory_memory import TemporalDiscriminativeTrajectoryMemory
from memory.transition_memory import BalanceState,stability_score

def load(p):
 e=defaultdict(list);names=None
 for line in p.open(encoding='utf8'):
  try:r=json.loads(line)
  except:continue
  if not r.get('full_body_state') or not r.get('state'):continue
  n=tuple(r['full_body_state']['names']);
  if names is None:names=n
  if n!=names:raise SystemExit('schema mismatch')
  e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items())),names
def v(r):return np.asarray(r['full_body_state']['vector'],float)
def b(r):
 s=r['state'];return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0)))
def win(rows,c=5,h=12):
 for i in range(c-1,len(rows)-h):yield rows[i-c+1:i+1],rows[i+1:i+h+1]
def cos(a,b):
 na=np.linalg.norm(a);nb=np.linalg.norm(b);return float(np.dot(a,b)/(na*nb)) if na*nb>1e-12 else 0.
def examples(eps,runs,h=12):
 out=[]
 for run in runs:
  for hist,fut in win(eps[run],5,h):
   old=stability_score(b(hist[0]),1.);now=stability_score(b(hist[-1]),1.)
   if old-now<.01:continue
   ss=np.asarray([stability_score(b(x),1.) for x in fut]);j=int(np.argmax(ss));gain=float(ss[j]-now)
   if gain>=.03:out.append((np.stack([v(x) for x in hist]),v(fut[j]),gain))
 return out
def scales(xs):
 x=np.asarray(xs);med=np.median(x,0);mad=np.median(abs(x-med),0);q25,q75=np.percentile(x,[25,75],axis=0);rob=np.maximum(1.4826*mad,(q75-q25)/1.349);sd=np.std(x,0);return np.maximum(rob,np.maximum(1e-4,.05*np.where(sd>1e-8,sd,1.)))
def learn(ex,sc,pairs=60000,seed=1111):
 rng=np.random.default_rng(seed);n=len(ex);traj=np.asarray([x[0] for x in ex]);cur=traj[:,-1];tar=np.asarray([x[1] for x in ex]);corr=(tar-cur)/sc
 i=rng.integers(0,n,pairs);j=rng.integers(0,n,pairs);m=i!=j;i=i[m];j=j[m];a=corr[i];bb=corr[j];den=np.linalg.norm(a,axis=1)*np.linalg.norm(bb,axis=1);sim=np.zeros(len(i));ok=den>1e-12;sim[ok]=np.sum(a[ok]*bb[ok],1)/den[ok];pos=sim>=.70;neg=sim<=.20
 dz=np.abs((traj[i]-traj[j])/sc[None,None,:]);pm=np.median(dz[pos],0);nm=np.median(dz[neg],0);w=(nm+1e-4)/(pm+1e-4);w=np.clip(w,.35,2.5);w/=np.mean(w)
 return w,int(pos.sum()),int(neg.sum())
def train(m,ex,label):
 for k,(h,t,g) in enumerate(ex,1):
  m.observe(h,t,g)
  if k%500==0:print(f'[V11.11 {label}] corrective={k} proto={m.size}',flush=True)
def eval(m,eps,runs,h=12):
 frozen=m.stats().copy();N=R=0;cs=[]
 for run in runs:
  for hist,fut in win(eps[run],5,h):
   N+=1;cur=v(hist[-1]);ss=np.asarray([stability_score(b(x),1.) for x in fut]);true=v(fut[int(np.argmax(ss))]);rr=m.recall([v(x) for x in hist],min_confidence=.40)
   if rr is None:continue
   R+=1;cs.append(cos((rr.target_state-cur)/m.scales,(true-cur)/m.scales))
 return R,N,R/max(1,N),float(np.mean(cs)) if cs else 0.,sum(x>=.70 for x in cs)/max(1,len(cs)),frozen==m.stats()
def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--pairs',type=int,default=60000);a=p.parse_args();st=time.time();eps,names=load(a.trace);ids=list(eps);sp=int(round(.7*len(ids)));tr,te=ids[:sp],ids[sp:];ex=examples(eps,tr);xs=[v(r) for run in tr for r in eps[run]];sc=scales(xs);w,np_,nn_=learn(ex,sc,a.pairs)
 print(f'V11.11 train={tr} holdout={te} corrective={len(ex)} pairs positive={np_} negative={nn_}')
 flat=np.argsort(w.ravel());print('Highest temporal weights:')
 for q in flat[-12:][::-1]:
  tt,ch=np.unravel_index(q,w.shape);print(f'  t-{4-tt} {names[ch]} weight={w[tt,ch]:.4f}')
 base=IndexedFullBodyTrajectoryMemory(context=5);base.fit_scales(xs);rank=TemporalDiscriminativeTrajectoryMemory(context=5,temporal_weights=w);rank.fit_scales(xs);train(base,ex,'BASE');train(rank,ex,'RANK');rb=eval(base,eps,te);rr=eval(rank,eps,te)
 print('\n'+'='*100);print('RoboCOP — V11.11 TEMPORAL DISCRIMINATIVE RANKING / FROZEN HOLDOUT');print('='*100);print(f"{'variant':18s} {'coverage':>10s} {'recall':>12s} {'cos':>9s} {'>=.70':>9s} {'proto':>8s}")
 for label,m,r in [('baseline',base,rb),('temporal-rank',rank,rr)]:print(f'{label:18s} {100*r[2]:9.2f}% {r[0]:5d}/{r[1]:<6d} {r[3]:9.4f} {100*r[4]:8.2f}% {m.size:8d}')
 print(f"Frozen baseline={'PASS' if rb[5] else 'FAIL'} temporal={'PASS' if rr[5] else 'FAIL'}");print('Reference V11.7: coverage=9.28%, cos=0.6863.');print(f'Elapsed: {time.time()-st:.1f}s');print('='*100)
if __name__=='__main__':main()
