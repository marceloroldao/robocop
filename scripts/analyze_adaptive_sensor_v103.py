#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from memory.sensor_trajectory_memory import SensorTrajectoryMemory
from memory.adaptive_sensor_trajectory_memory import AdaptiveSensorTrajectoryMemory
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
 for i in range(c-1,len(rows)-h):yield [st(x) for x in rows[i-c+1:i+1]],[st(x) for x in rows[i+1:i+h+1]]

def evaluate(m,eps,ids,c,h):
 out={'total':0,'rec':0,'direct':0,'interp':0,'cos':[],'pred':[],'actual':[],'dist':[],'maxerr':[],'density':[],'factor':[]}
 scale=m.sensor_scale
 for run in ids:
  for hist,fut in wins(eps[run],c,h):
   out['total']+=1; rr=m.recall(hist)
   if rr is None:continue
   out['rec']+=1;out['direct']+=int(rr.direct);out['interp']+=int(not rr.direct);out['dist'].append(rr.rms_distance);out['maxerr'].append(rr.max_sensor_error)
   if hasattr(m,'last_local_density'):out['density'].append(m.last_local_density);out['factor'].append(m.last_resolution_factor)
   cur=hist[-1].vector();scores=np.asarray([stability_score(x,m.target_height) for x in fut]);j=int(np.argmax(scores));true=fut[j].vector();now=stability_score(hist[-1],m.target_height)
   x=(rr.target_state-cur)/scale;y=(true-cur)/scale;nx=np.linalg.norm(x);ny=np.linalg.norm(y);out['cos'].append(float(np.dot(x,y)/(nx*ny)) if nx>0 and ny>0 else 0.)
   pred=BalanceState(*[float(v) for v in rr.target_state]);out['pred'].append(stability_score(pred,m.target_height)-now);out['actual'].append(float(scores[j]-now))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,default=ROOT/'results/bahiart_multi_episode/combined_trace.jsonl');p.add_argument('--train-fraction',type=float,default=.70);p.add_argument('--context',type=int,default=5);p.add_argument('--horizon',type=int,default=12);a=p.parse_args()
 eps=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*a.train_fraction))));tr,te=ids[:split],ids[split:]
 fixed=SensorTrajectoryMemory();adapt=AdaptiveSensorTrajectoryMemory();tw=0
 for run in tr:
  for hist,fut in wins(eps[run],a.context,a.horizon):tw+=1;fixed.observe_window(hist,fut);adapt.observe_window(hist,fut)
 ff,fa=fixed.stats().copy(),adapt.stats().copy();rf=evaluate(fixed,eps,te,a.context,a.horizon);ra=evaluate(adapt,eps,te,a.context,a.horizon)
 unchanged=(ff==fixed.stats()) and (fa==adapt.stats())
 mean=lambda x:float(np.mean(x)) if x else 0.;p95=lambda x:float(np.percentile(x,95)) if x else 0.
 print('='*96);print('RoboCOP — V10.3 ADAPTIVE LOCAL RESOLUTION / EPISODE HOLDOUT');print('='*96)
 print(f'Episodes total:                   {len(ids)}');print(f'Training / holdout:               {tr} / {te}');print(f'Training windows:                 {tw}');print(f'Memory unchanged in holdout:      {"PASS" if unchanged else "FAIL"}');print()
 for name,m,s,r in [('V10.2 FIXED',fixed,ff,rf),('V10.3 ADAPTIVE',adapt,fa,ra)]:
  print(name);print(f'  prototypes / confirmed:         {s["records"]} / {s["confirmed_records"]}');print(f'  merged observations:            {s["merged"]}');print(f'  recalls:                        {r["rec"]}/{r["total"]} ({100*r["rec"]/r["total"] if r["total"] else 0:.2f}%)');print(f'  direct / interpolated:          {r["direct"]} / {r["interp"]}');print(f'  address RMS mean/p95:           {mean(r["dist"]):.4f} / {p95(r["dist"]):.4f}');print(f'  max sensor error mean/p95:      {mean(r["maxerr"]):.4f} / {p95(r["maxerr"]):.4f}');print(f'  recovery direction cosine:      {mean(r["cos"]):.4f}');print(f'  aligned >=0.70:                 {sum(x>=.70 for x in r["cos"])}/{len(r["cos"])} ({100*sum(x>=.70 for x in r["cos"])/len(r["cos"]) if r["cos"] else 0:.2f}%)');print(f'  predicted / observed gain:      {mean(r["pred"]):.4f} / {mean(r["actual"]):.4f}')
  if r['density']:print(f'  local density mean/p95:         {mean(r["density"]):.2f} / {p95(r["density"]):.2f}');print(f'  resolution factor mean/p95:     {mean(r["factor"]):.4f} / {p95(r["factor"]):.4f}')
  print()
 print('DELTA ADAPTIVE - FIXED');print(f'  coverage:                       {(100*ra["rec"]/ra["total"])-(100*rf["rec"]/rf["total"]):+.2f} pp');print(f'  max sensor error mean:          {mean(ra["maxerr"])-mean(rf["maxerr"]):+.4f}');print(f'  direction cosine:               {mean(ra["cos"])-mean(rf["cos"]):+.4f}');print(f'  optimism gap |pred-observed|:   {abs(mean(ra["pred"])-mean(ra["actual"]))-abs(mean(rf["pred"])-mean(rf["actual"])):+.4f}');print('='*96)
if __name__=='__main__':main()
