#!/usr/bin/env python3
"""Consolidation benchmark: original RoboCOP transition memory vs best V11.7.

Same V11.5 dataset and temporal split (episodes 1-14 train, 15-20 holdout).
Important: the memories have different native objectives, so the report keeps
native quality metrics separate instead of pretending they are identical.
Original: one-step stabilizing action recall from 6D BalanceState.
V11.7: 5-step full-body recovery trajectory, with high-confidence instantaneous
fallback at 0.70 on trajectory MISS.
"""
from __future__ import annotations
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory.transition_memory import BalanceState,ResolutiveTransitionMemory,stability_score
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory

def load(p):
 e=defaultdict(list);schema=None
 for line in p.open(encoding='utf-8'):
  try:r=json.loads(line)
  except Exception:continue
  if not r.get('state') or not r.get('full_body_state') or r.get('baseline_action') is None:continue
  names=tuple(r['full_body_state'].get('names',[]));v=np.asarray(r['full_body_state'].get('vector',[]),float)
  if schema is None:schema=names
  if names!=schema or len(v)!=len(schema):raise SystemExit('schema mismatch')
  e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items())),schema
def bal(r):
 s=r['state'];return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.)))
def vec(r):return np.asarray(r['full_body_state']['vector'],float)
def act(r):return np.asarray(r['baseline_action'],float)
def windows(rows,c,h):
 for i in range(c-1,len(rows)-h):yield rows[i-c+1:i+1],rows[i+1:i+h+1]
def cosine(a,b):
 na=np.linalg.norm(a);nb=np.linalg.norm(b);return float(np.dot(a,b)/(na*nb)) if na>1e-12 and nb>1e-12 else 0.
def nrmse(a,b):
 scale=max(1e-9,float(np.sqrt(np.mean(b*b))));return float(np.sqrt(np.mean((a-b)**2))/scale)

def train_original(eps,runs):
 m=ResolutiveTransitionMemory(min_gain=.005,target_height=1.0);seen=0;admitted=0
 for run in runs:
  rows=eps[run]
  for i in range(len(rows)-1):
   seen+=1
   if m.observe(bal(rows[i]),act(rows[i]),bal(rows[i+1])):admitted+=1
 return m,seen,admitted

def eval_original(m,eps,runs):
 frozen=m.stats().copy();q=rec=0;cs=[];es=[];conf=[];layers=defaultdict(int)
 for run in runs:
  rows=eps[run]
  for i,r in enumerate(rows):
   q+=1;recent=bal(rows[i-1]) if i>0 else None;rr=m.recall(bal(r),recent_state=recent,min_confidence=.65)
   if rr is None:continue
   rec+=1;pred=np.asarray(rr.action,float);truth=act(r)
   cs.append(cosine(pred,truth));es.append(nrmse(pred,truth));conf.append(rr.confidence);layers[rr.layer]+=1
 return dict(queries=q,recalls=rec,coverage=rec/max(1,q),cos=float(np.mean(cs)) if cs else 0.,nrmse=float(np.mean(es)) if es else 0.,confidence=float(np.mean(conf)) if conf else 0.,layers=dict(layers),frozen=(frozen==m.stats()))

def corrective_examples(eps,runs,horizon,context):
 out=[]
 for run in runs:
  for hist,fut in windows(eps[run],5,horizon):
   old=stability_score(bal(hist[0]),1.);now=stability_score(bal(hist[-1]),1.)
   if old-now<.01:continue
   scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));gain=float(scores[j]-now)
   if gain<.03:continue
   out.append(([vec(x) for x in hist[-context:]],vec(fut[j]),gain))
 return out

def train_v117(eps,runs,horizon,context):
 m=IndexedFullBodyTrajectoryMemory(context=context);m.fit_scales([vec(r) for run in runs for r in eps[run]])
 ex=corrective_examples(eps,runs,horizon,context)
 for hist,target,gain in ex:m.observe(hist,target,gain)
 return m,len(ex)

def eval_v117(fine,coarse,eps,runs,horizon,fb_conf=.70):
 ff=fine.stats().copy();fc=coarse.stats().copy();q=rec=nf=nb=0;cs=[];conf=[]
 for run in runs:
  for hist,fut in windows(eps[run],5,horizon):
   q+=1;cur=vec(hist[-1]);scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));true=vec(fut[j]);truth=(true-cur)/fine.scales
   rr=fine.recall([vec(x) for x in hist],min_confidence=.40)
   if rr is not None:
    nf+=1
   else:
    rr=coarse.recall([cur],min_confidence=.40)
    if rr is None or rr.confidence<fb_conf:continue
    nb+=1
   rec+=1;conf.append(rr.confidence)
   scales=fine.scales if nf+nb and rr.target_state.shape==fine.scales.shape else coarse.scales
   pred=(rr.target_state-cur)/scales
   truth2=(true-cur)/scales
   cs.append(cosine(pred,truth2))
 return dict(queries=q,recalls=rec,coverage=rec/max(1,q),cos=float(np.mean(cs)) if cs else 0.,aligned=sum(c>=.70 for c in cs)/max(1,len(cs)),confidence=float(np.mean(conf)) if conf else 0.,fine=nf,fallback=nb,frozen=(ff==fine.stats() and fc==coarse.stats()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--horizon',type=int,default=12);a=p.parse_args();t=time.time()
 eps,names=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*.70))));tr,te=ids[:split],ids[split:]
 print(f'COMPARE dataset episodes={len(ids)} train={tr} holdout={te} rows={sum(len(eps[x]) for x in ids)}')
 t0=time.time();orig,seen,adm=train_original(eps,tr);orig_train=time.time()-t0
 t0=time.time();fine,nftrain=train_v117(eps,tr,a.horizon,5);coarse,nctrain=train_v117(eps,tr,a.horizon,1);vtrain=time.time()-t0
 ro=eval_original(orig,eps,te);rv=eval_v117(fine,coarse,eps,te,a.horizon,.70)
 so=orig.stats();sf=fine.stats();sc=coarse.stats()
 print('\n'+'='*116);print('RoboCOP — ORIGINAL vs BEST V11.7 / CONSOLIDATION HOLDOUT');print('='*116)
 print('Same train/holdout episodes. Native objectives are intentionally reported separately.')
 print('\nORIGINAL — 6D one-step transition/action memory')
 print(f"  train transitions seen/admitted: {seen}/{adm}")
 print(f"  records/confirmed:              {so['records']}/{so['confirmed_records']}")
 print(f"  holdout action recalls:         {ro['recalls']}/{ro['queries']} ({100*ro['coverage']:.2f}%)")
 print(f"  action cosine mean:             {ro['cos']:.4f}")
 print(f"  action NRMSE mean:              {ro['nrmse']:.4f}")
 print(f"  confidence mean:                {ro['confidence']:.4f}")
 print(f"  layers:                         {ro['layers']}")
 print(f"  frozen holdout:                 {'PASS' if ro['frozen'] else 'FAIL'}")
 print(f"  train seconds:                  {orig_train:.1f}")
 print('\nBEST V11.7 — full-body trajectory + instantaneous fallback@0.70')
 print(f"  corrective examples fine/coarse: {nftrain}/{nctrain}")
 print(f"  prototypes fine/coarse:           {sf['records']}/{sc['records']}")
 print(f"  confirmed fine/coarse:            {sf['confirmed_records']}/{sc['confirmed_records']}")
 print(f"  holdout recovery recalls:          {rv['recalls']}/{rv['queries']} ({100*rv['coverage']:.2f}%)")
 print(f"  fine/fallback recalls:              {rv['fine']}/{rv['fallback']}")
 print(f"  recovery direction cosine mean:     {rv['cos']:.4f}")
 print(f"  recovery cosine >=0.70:              {100*rv['aligned']:.2f}%")
 print(f"  confidence mean:                     {rv['confidence']:.4f}")
 print(f"  frozen holdout:                      {'PASS' if rv['frozen'] else 'FAIL'}")
 print(f"  train seconds:                       {vtrain:.1f}")
 print('\nINTERPRETATION RULE')
 print('  Do not compare original action-cosine numerically with V11.7 recovery-cosine as if they were the same target.')
 print('  This stage compares representation, generalization, coverage, consolidation and native prediction quality.')
 print('  The next active A/B will use the same behavioral endpoint for both: walk duration/falls/stability.')
 print(f'\nTotal elapsed: {time.time()-t:.1f}s');print('='*116)
if __name__=='__main__':main()
