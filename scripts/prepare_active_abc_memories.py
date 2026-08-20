#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pickle,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory.transition_memory import BalanceState,ResolutiveTransitionMemory,stability_score
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory

def load(p):
 e=defaultdict(list)
 for line in p.open(encoding='utf-8'):
  try:r=json.loads(line)
  except Exception:continue
  if r.get('state') and r.get('full_body_state') and r.get('baseline_action') is not None:
   e[int(r.get('run_id',r.get('episode',0)))].append(r)
 return dict(sorted(e.items()))
def bal(r):
 s=r['state'];return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.)))
def vec(r):return np.asarray(r['full_body_state']['vector'],float)
def act(r):return np.asarray(r['baseline_action'],float)
def windows(rows,c,h):
 for i in range(c-1,len(rows)-h):yield rows[i-c+1:i+1],rows[i+1:i+h+1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--horizon',type=int,default=12);a=ap.parse_args()
 eps=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*.70))));tr=ids[:split]
 # Original frozen memory: native one-step transition/action objective.
 original=ResolutiveTransitionMemory(min_gain=.005,target_height=1.0)
 for run in tr:
  rows=eps[run]
  for i in range(len(rows)-1):
   original.observe(bal(rows[i]),act(rows[i]),bal(rows[i+1]),terminal=False)
 # V11.7 fine + coarse; retain demonstrations for target-state -> motor-action decoding.
 train_vectors=[vec(r) for run in tr for r in eps[run]]
 fine=IndexedFullBodyTrajectoryMemory(context=5);fine.fit_scales(train_vectors)
 coarse=IndexedFullBodyTrajectoryMemory(context=1);coarse.fit_scales(train_vectors)
 demo_targets=[];demo_actions=[];corrective=0
 for run in tr:
  for hist,fut in windows(eps[run],5,a.horizon):
   old=stability_score(bal(hist[0]),1.);now=stability_score(bal(hist[-1]),1.)
   if old-now<.01:continue
   scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));gain=float(scores[j]-now)
   if gain<.03:continue
   target=vec(fut[j]);action=act(fut[j]);fine.observe([vec(x) for x in hist],target,gain);coarse.observe([vec(hist[-1])],target,gain)
   demo_targets.append(target);demo_actions.append(action);corrective+=1
 payload={'original':original,'fine':fine,'coarse':coarse,'demo_targets':np.asarray(demo_targets,float),'demo_actions':np.asarray(demo_actions,float),'train_runs':tr,'corrective':corrective,'trace':str(a.trace)}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('wb') as f:pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
 print(f'Prepared active A/B/C memories: original_records={original.size} fine={fine.size} coarse={coarse.size} demos={len(demo_targets)} out={a.out}')
if __name__=='__main__':main()
