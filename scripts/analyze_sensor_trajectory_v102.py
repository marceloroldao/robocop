#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from memory.sensor_trajectory_memory import SensorTrajectoryMemory
from memory.transition_memory import BalanceState, stability_score

def st(row):
 s=row['state']; return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.0)))
def load(path):
 e=defaultdict(list)
 for line in path.open(encoding='utf-8'):
  try:r=json.loads(line)
  except:continue
  if r.get('state'):e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items()))
def wins(rows,c,h):
 for i in range(c-1,len(rows)-h):
  yield [st(x) for x in rows[i-c+1:i+1]],[st(x) for x in rows[i+1:i+h+1]]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--trace',type=Path,default=ROOT/'results/bahiart_multi_episode/combined_trace.jsonl'); p.add_argument('--train-fraction',type=float,default=.70); p.add_argument('--context',type=int,default=5); p.add_argument('--horizon',type=int,default=12); a=p.parse_args()
 eps=load(a.trace); ids=list(eps); split=max(1,min(len(ids)-1,int(round(len(ids)*a.train_fraction)))); tr,te=ids[:split],ids[split:]
 m=SensorTrajectoryMemory(); tw=0
 for run in tr:
  for hist,fut in wins(eps[run],a.context,a.horizon):tw+=1;m.observe_window(hist,fut)
 frozen=m.stats().copy(); total=rec=direct=interp=0; conf=[]; coh=[]; dist=[]; sensor_max=[]; cos=[]; pd=[]; ag=[]; by=defaultdict(lambda:[0,0,0])
 scale=m.sensor_scale
 for run in te:
  for hist,fut in wins(eps[run],a.context,a.horizon):
   total+=1;by[run][0]+=1; rr=m.recall(hist)
   if rr is None:continue
   rec+=1;direct+=int(rr.direct);interp+=int(not rr.direct);by[run][1]+=1;by[run][2]+=int(not rr.direct);conf.append(rr.confidence);coh.append(rr.coherence);dist.append(rr.rms_distance);sensor_max.append(rr.max_sensor_error)
   cur=hist[-1].vector(); scores=np.asarray([stability_score(x,m.target_height) for x in fut]);j=int(np.argmax(scores));true=fut[j].vector();now=stability_score(hist[-1],m.target_height)
   x=(rr.target_state-cur)/scale;y=(true-cur)/scale;nx=np.linalg.norm(x);ny=np.linalg.norm(y);cos.append(float(np.dot(x,y)/(nx*ny)) if nx>0 and ny>0 else 0.)
   pred=BalanceState(*[float(v) for v in rr.target_state]);pd.append(stability_score(pred,m.target_height)-now);ag.append(float(scores[j]-now))
 after=m.stats();mean=lambda x:float(np.mean(x)) if x else 0.;p95=lambda x:float(np.percentile(x,95)) if x else 0.;aligned=sum(x>=.70 for x in cos);positive=sum(x>0 for x in pd)
 print('='*92);print('RoboCOP — V10.2 SENSORWISE TRAJECTORY ADDRESS / EPISODE HOLDOUT');print('='*92)
 print(f'Episodes total:                  {len(ids)}');print(f'Training episodes:               {tr}');print(f'Holdout episodes:                {te}');print(f'Address:                         S + dS + ddS (18 dimensions)');print(f'Hard gate:                       every sensor/derivative must be compatible');print(f'Context / recovery horizon:      {a.context} / {a.horizon}');print(f'Training windows:                {tw}');print(f'Recovery prototypes:             {frozen["records"]}');print(f'Confirmed recovery targets:      {frozen["confirmed_records"]}');print(f'Merged observations:             {frozen["merged"]}');print(f'Max confirmations:               {frozen["max_confirmations"]}');print(f'Memory unchanged in holdout:     {"PASS" if frozen==after else "FAIL"}');print()
 print(f'Holdout windows:                 {total}');print(f'Target recalls:                  {rec} ({100*rec/total if total else 0:.2f}%)');print(f'Direct / interpolated:           {direct} / {interp}');print(f'MISS:                            {total-rec} ({100*(total-rec)/total if total else 0:.2f}%)');print(f'Recall confidence mean:          {mean(conf):.4f}');print(f'Neighborhood coherence mean:     {mean(coh):.4f}');print(f'Address RMS distance mean/p95:   {mean(dist):.4f} / {p95(dist):.4f}');print(f'Max sensor error mean/p95:       {mean(sensor_max):.4f} / {p95(sensor_max):.4f}');print(f'Recovery direction cosine mean:  {mean(cos):.4f}');print(f'Direction aligned >=0.70:        {aligned}/{len(cos)} ({100*aligned/len(cos) if cos else 0:.2f}%)');print(f'Predicted positive gain:         {positive}/{len(pd)} ({100*positive/len(pd) if pd else 0:.2f}%)');print(f'Predicted gain mean:             {mean(pd):.4f}');print(f'Observed best gain mean:         {mean(ag):.4f}');print();print('HOLDOUT COVERAGE BY EPISODE')
 for run in te:
  w,r,i=by[run];print(f'run={run:03d} windows={w:4d} recalls={r:4d} interp={i:4d} rate={(100*r/w if w else 0):6.2f}%')
 print('='*92)
if __name__=='__main__':main()
