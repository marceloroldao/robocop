#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.transition_memory import BalanceState,stability_score

def balance(r):
 s=r['state']; return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.)))
def body(r):return np.asarray(r['full_body_state']['vector'],dtype=float)
def load(path):
 e=defaultdict(list); schema=None
 with path.open(encoding='utf-8') as f:
  for line in f:
   try:r=json.loads(line)
   except:continue
   if not r.get('state') or not r.get('full_body_state'):continue
   names=tuple(r['full_body_state'].get('names',[])); schema=names if schema is None else schema
   if names!=schema:raise SystemExit('V11 schema changed inside trace')
   e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items()))
def windows(rows,c,h):
 for i in range(c-1,len(rows)-h):yield rows[i-c+1:i+1],rows[i+1:i+h+1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--train-fraction',type=float,default=.70);p.add_argument('--context',type=int,default=5);p.add_argument('--horizon',type=int,default=12);p.add_argument('--progress-every',type=int,default=500);a=p.parse_args()
 t0=time.time();eps=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*a.train_fraction))));tr,te=ids[:split],ids[split:]
 vectors=[body(r) for run in tr for r in eps[run]];dims=vectors[0].size;m=IndexedFullBodyTrajectoryMemory(context=a.context);m.fit_scales(vectors)
 print(f'[V11.1] loaded episodes={len(ids)} train={tr} holdout={te} channels={dims}',flush=True)
 tw=cand=corr=0
 for run in tr:
  for hist,fut in windows(eps[run],a.context,a.horizon):
   tw+=1;bh=[balance(x) for x in hist];old=stability_score(bh[0],1.);now=stability_score(bh[-1],1.)
   if old-now<.01:continue
   cand+=1;bs=[balance(x) for x in fut];scores=np.asarray([stability_score(x,1.) for x in bs]);j=int(np.argmax(scores));gain=float(scores[j]-now)
   if gain<.03:continue
   corr+=1;m.observe([body(x) for x in hist],body(fut[j]),gain)
   if corr%a.progress_every==0:print(f'[V11.1] train corrective={corr} prototypes={m.size} elapsed={time.time()-t0:.1f}s',flush=True)
 frozen=m.stats().copy();hold=rec=direct=interp=0;conf=[];coh=[];rms=[];mx=[];den=[];fac=[];direction=[];td=[];obs_gain=[];by=defaultdict(lambda:[0,0,0])
 for run in te:
  print(f'[V11.1] holdout run={run} start',flush=True)
  for hist,fut in windows(eps[run],a.context,a.horizon):
   hold+=1;by[run][0]+=1;rr=m.recall([body(x) for x in hist])
   if rr is None:continue
   rec+=1;direct+=int(rr.direct);interp+=int(not rr.direct);by[run][1]+=1;by[run][2]+=int(not rr.direct);conf.append(rr.confidence);coh.append(rr.coherence);rms.append(rr.rms_distance);mx.append(rr.max_channel_error);den.append(rr.local_density);fac.append(rr.resolution_factor)
   cur=body(hist[-1]);fb=[balance(x) for x in fut];scores=np.asarray([stability_score(x,1.) for x in fb]);j=int(np.argmax(scores));true=body(fut[j]);x=(rr.target_state-cur)/m.scales;y=(true-cur)/m.scales;nx=np.linalg.norm(x);ny=np.linalg.norm(y);direction.append(float(np.dot(x,y)/(nx*ny)) if nx>0 and ny>0 else 0.);td.append(float(np.sqrt(np.mean(((rr.target_state-true)/m.scales)**2))));obs_gain.append(float(scores[j]-stability_score(balance(hist[-1]),1.)))
  print(f'[V11.1] holdout run={run} done recalls={by[run][1]}/{by[run][0]} elapsed={time.time()-t0:.1f}s',flush=True)
 mean=lambda x:float(np.mean(x)) if x else 0.;p95=lambda x:float(np.percentile(x,95)) if x else 0.;aligned=sum(x>=.70 for x in direction);idx=m.index_stats()
 print('='*94);print('RoboCOP — V11.1 INDEXED FULL BODY SENSOR ADDRESS / EPISODE HOLDOUT');print('='*94)
 print(f'Episodes total:                   {len(ids)}');print(f'Training episodes:                {tr}');print(f'Holdout episodes:                 {te}');print(f'Raw sensor channels:              {dims}');print(f'Trajectory address dimensions:    {dims*a.context} ({a.context} x {dims})');print(f'Training windows:                 {tw}');print(f'Candidate degrading windows:      {cand}');print(f'Corrective windows admitted:      {corr}');print(f'Full-body prototypes:             {frozen["records"]}');print(f'Confirmed prototypes:             {frozen["confirmed_records"]}');print(f'Merged observations:              {frozen["merged"]}');print(f'Memory unchanged in holdout:      {"PASS" if frozen==m.stats() else "FAIL"}')
 print(f'Index buckets:                    {idx["buckets"]}');print(f'Mean candidates/query:            {idx["mean_candidates"]:.2f}');print(f'Analysis elapsed seconds:         {time.time()-t0:.2f}');print();print(f'Holdout windows:                  {hold}');print(f'Full-body recalls:                {rec} ({100*rec/hold if hold else 0:.2f}%)');print(f'Direct / interpolated:            {direct} / {interp}');print(f'MISS:                             {hold-rec} ({100*(hold-rec)/hold if hold else 0:.2f}%)');print(f'Recall confidence mean:           {mean(conf):.4f}');print(f'Neighborhood coherence mean:      {mean(coh):.4f}');print(f'Trajectory RMS mean/p95:          {mean(rms):.4f} / {p95(rms):.4f}');print(f'Max channel error mean/p95:       {mean(mx):.4f} / {p95(mx):.4f}');print(f'Local density mean/p95:           {mean(den):.2f} / {p95(den):.2f}');print(f'Resolution factor mean/p95:       {mean(fac):.4f} / {p95(fac):.4f}');print(f'Recovery direction cosine mean:   {mean(direction):.4f}');print(f'Direction aligned >=0.70:         {aligned}/{len(direction)} ({100*aligned/len(direction) if direction else 0:.2f}%)');print(f'Target distance mean/p95:         {mean(td):.4f} / {p95(td):.4f}');print(f'Observed best gain mean:          {mean(obs_gain):.4f}');print();print('HOLDOUT COVERAGE BY EPISODE')
 for run in te:
  w,r,i=by[run];print(f'run={run:03d} windows={w:5d} recalls={r:5d} interp={i:5d} rate={(100*r/w if w else 0):6.2f}%')
 print('='*94)
if __name__=='__main__':main()
