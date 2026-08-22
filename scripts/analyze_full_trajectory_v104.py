#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from memory.full_sensor_trajectory_memory import FullSensorTrajectoryMemory
from memory.sensor_trajectory_memory import SensorTrajectoryMemory
from memory.transition_memory import BalanceState, stability_score

def st(row):
 s=row['state']; return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.0)))
def load(path):
 e=defaultdict(list)
 for line in path.open(encoding='utf-8'):
  try:r=json.loads(line)
  except:continue
  if r.get('state'): e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items()))
def wins(rows,c,h):
 for i in range(c-1,len(rows)-h):
  yield [st(x) for x in rows[i-c+1:i+1]],[st(x) for x in rows[i+1:i+h+1]]
def eval_model(m,eps,ids,c,h):
 total=rec=direct=interp=0; dist=[]; mx=[]; cos=[]; pd=[]; ag=[]; dens=[]; fac=[]
 for run in ids:
  for hist,fut in wins(eps[run],c,h):
   total+=1; rr=m.recall(hist)
   if rr is None: continue
   rec+=1; direct+=int(rr.direct); interp+=int(not rr.direct); dist.append(rr.rms_distance); mx.append(rr.max_channel_error if hasattr(rr,'max_channel_error') else rr.max_sensor_error)
   if hasattr(rr,'local_density'): dens.append(rr.local_density)
   if hasattr(rr,'resolution_factor'): fac.append(rr.resolution_factor)
   cur=hist[-1].vector(); scores=np.asarray([stability_score(x,m.target_height) for x in fut]); j=int(np.argmax(scores)); true=fut[j].vector(); now=stability_score(hist[-1],m.target_height)
   x=(rr.target_state-cur)/m.sensor_scale; y=(true-cur)/m.sensor_scale; nx=np.linalg.norm(x); ny=np.linalg.norm(y); cos.append(float(np.dot(x,y)/(nx*ny)) if nx>0 and ny>0 else 0.)
   pred=BalanceState(*[float(v) for v in rr.target_state]); pd.append(stability_score(pred,m.target_height)-now); ag.append(float(scores[j]-now))
 mean=lambda x:float(np.mean(x)) if x else 0.; p95=lambda x:float(np.percentile(x,95)) if x else 0.
 return dict(total=total,rec=rec,direct=direct,interp=interp,dist=mean(dist),dist95=p95(dist),mx=mean(mx),mx95=p95(mx),cos=mean(cos),aligned=sum(x>=.70 for x in cos),ncos=len(cos),pd=mean(pd),ag=mean(ag),dens=mean(dens),fac=mean(fac),fac95=p95(fac))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--trace',type=Path,default=ROOT/'results/bahiart_multi_episode/combined_trace.jsonl');p.add_argument('--train-fraction',type=float,default=.70);p.add_argument('--context',type=int,default=5);p.add_argument('--horizon',type=int,default=12);a=p.parse_args()
 eps=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*a.train_fraction))));tr,te=ids[:split],ids[split:]
 old=SensorTrajectoryMemory(); new=FullSensorTrajectoryMemory(); tw=0
 for run in tr:
  for hist,fut in wins(eps[run],a.context,a.horizon): tw+=1; old.observe_window(hist,fut); new.observe_window(hist,fut)
 fo,fn=old.stats().copy(),new.stats().copy(); eo=eval_model(old,eps,te,a.context,a.horizon); en=eval_model(new,eps,te,a.context,a.horizon)
 print('='*96);print('RoboCOP — V10.4 FULL SENSOR TRAJECTORY / EPISODE HOLDOUT');print('='*96)
 print(f'Episodes total:                    {len(ids)}');print(f'Training episodes:                 {tr}');print(f'Holdout episodes:                  {te}');print(f'Context window:                    {a.context} states x 6 channels');print(f'Training windows:                  {tw}');print()
 for name,s,e in [('V10.2 S+dS+ddS',fo,eo),('V10.4 FULL WINDOW',fn,en)]:
  print(name);print(f'  prototypes / confirmed:          {s["records"]} / {s["confirmed_records"]}');print(f'  merged / max confirmations:      {s["merged"]} / {s["max_confirmations"]}');print(f'  holdout recalls:                 {e["rec"]}/{e["total"]} ({100*e["rec"]/e["total"] if e["total"] else 0:.2f}%)');print(f'  direct / interpolated:           {e["direct"]} / {e["interp"]}');print(f'  RMS distance mean/p95:           {e["dist"]:.4f} / {e["dist95"]:.4f}');print(f'  max channel error mean/p95:      {e["mx"]:.4f} / {e["mx95"]:.4f}');print(f'  recovery direction cosine mean:  {e["cos"]:.4f}');print(f'  direction aligned >=0.70:        {e["aligned"]}/{e["ncos"]} ({100*e["aligned"]/e["ncos"] if e["ncos"] else 0:.2f}%)');print(f'  predicted / observed gain:       {e["pd"]:.4f} / {e["ag"]:.4f}')
  if name.startswith('V10.4'): print(f'  local density mean:              {e["dens"]:.2f}');print(f'  resolution factor mean/p95:      {e["fac"]:.4f} / {e["fac95"]:.4f}')
  print()
 print('='*96)
if __name__=='__main__': main()
