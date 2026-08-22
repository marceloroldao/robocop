#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from memory.recovery_target_memory import RecoveryTargetMemory
from memory.transition_memory import BalanceState, stability_score

def state(row):
 s=row['state']; return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.0)))
def load(path):
 e=defaultdict(list)
 for line in path.open(encoding='utf-8'):
  try:r=json.loads(line)
  except:continue
  if r.get('state'): e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items()))
def windows(rows,c,h):
 for i in range(c-1,len(rows)-h):
  hist=[state(x) for x in rows[i-c+1:i+1]]; fut=[state(x) for x in rows[i+1:i+h+1]]
  yield hist,fut

def main():
 p=argparse.ArgumentParser(); p.add_argument('--trace',type=Path,default=ROOT/'results/bahiart_multi_episode/combined_trace.jsonl'); p.add_argument('--train-fraction',type=float,default=.70); p.add_argument('--context',type=int,default=5); p.add_argument('--horizon',type=int,default=12); a=p.parse_args()
 eps=load(a.trace); ids=list(eps); split=max(1,min(len(ids)-1,int(round(len(ids)*a.train_fraction)))); tr,te=ids[:split],ids[split:]
 m=RecoveryTargetMemory()
 tw=0
 for run in tr:
  for hist,fut in windows(eps[run],a.context,a.horizon): tw+=1; m.observe_window(hist,fut)
 frozen=m.stats().copy(); n=rec=direct=interp=0; conf=[]; coh=[]; target_dist=[]; direction_cos=[]; predicted_gain=[]; actual_best_gain=[]; by=defaultdict(lambda:[0,0,0])
 scale=np.asarray([.10,.25,.25,.60,.35,.10])
 for run in te:
  for hist,fut in windows(eps[run],a.context,a.horizon):
   n+=1; by[run][0]+=1; rr=m.recall(hist)
   if rr is None: continue
   rec+=1; direct+=int(rr.direct); interp+=int(not rr.direct); by[run][1]+=1; by[run][2]+=int(not rr.direct); conf.append(rr.confidence); coh.append(rr.coherence)
   cur=hist[-1].vector(); scores=np.asarray([stability_score(x,m.target_height) for x in fut]); j=int(np.argmax(scores)); true=fut[j].vector(); now=stability_score(hist[-1],m.target_height)
   target_dist.append(float(np.sqrt(np.mean(((rr.target_state-true)/scale)**2))))
   x=(rr.target_state-cur)/scale; y=(true-cur)/scale; nx=np.linalg.norm(x); ny=np.linalg.norm(y); direction_cos.append(float(np.dot(x,y)/(nx*ny)) if nx>0 and ny>0 else 0.)
   pred_state=BalanceState(*[float(v) for v in rr.target_state]); predicted_gain.append(stability_score(pred_state,m.target_height)-now); actual_best_gain.append(float(scores[j]-now))
 after=m.stats(); mean=lambda x:float(np.mean(x)) if x else 0.; p95=lambda x:float(np.percentile(x,95)) if x else 0.
 aligned=sum(c>=.70 for c in direction_cos); positive=sum(g>0 for g in predicted_gain)
 print('='*88); print('RoboCOP — V10.1 RECOVERY-TARGET INTERPOLATION / EPISODE HOLDOUT'); print('='*88)
 print(f'Episodes total:                 {len(ids)}'); print(f'Training episodes:              {tr}'); print(f'Holdout episodes:               {te}'); print(f'Context / recovery horizon:     {a.context} / {a.horizon}'); print(f'Training windows:               {tw}')
 print(f'Recovery prototypes:            {frozen["records"]}'); print(f'Confirmed recovery targets:     {frozen["confirmed_records"]}'); print(f'Merged observations:            {frozen["merged"]}'); print(f'Max confirmations:              {frozen["max_confirmations"]}'); print(f'Memory unchanged in holdout:    {"PASS" if frozen==after else "FAIL"}'); print()
 print(f'Holdout windows:                {n}'); print(f'Target recalls:                 {rec} ({100*rec/n if n else 0:.2f}%)'); print(f'Direct / interpolated:          {direct} / {interp}'); print(f'Recall confidence mean:         {mean(conf):.4f}'); print(f'Neighborhood coherence mean:    {mean(coh):.4f}'); print(f'Target-state distance mean/p95: {mean(target_dist):.4f} / {p95(target_dist):.4f}'); print(f'Recovery direction cosine mean: {mean(direction_cos):.4f}'); print(f'Direction aligned >=0.70:       {aligned}/{len(direction_cos)} ({100*aligned/len(direction_cos) if direction_cos else 0:.2f}%)'); print(f'Predicted positive gain:        {positive}/{len(predicted_gain)} ({100*positive/len(predicted_gain) if predicted_gain else 0:.2f}%)'); print(f'Predicted gain mean:            {mean(predicted_gain):.4f}'); print(f'Observed best gain mean:        {mean(actual_best_gain):.4f}'); print(); print('HOLDOUT COVERAGE BY EPISODE')
 for run in te:
  w,r,i=by[run]; print(f'run={run:03d} windows={w:4d} recalls={r:4d} interp={i:4d} rate={(100*r/w if w else 0):6.2f}%')
 print('='*88)
if __name__=='__main__': main()
