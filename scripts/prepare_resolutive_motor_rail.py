#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pickle,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from memory.resolutive_motor_rail import ResolutiveMotorRailMemory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--context',type=int,default=3);a=ap.parse_args()
 eps=defaultdict(list)
 for line in a.trace.open(encoding='utf-8'):
  try:r=json.loads(line)
  except Exception:continue
  if r.get('full_body_state') and r.get('baseline_action') is not None:eps[int(r.get('run_id',r.get('episode',0)))].append(r)
 ids=sorted(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*.70))));train=ids[:split]
 allstates=[np.asarray(r['full_body_state']['vector'],float) for k in train for r in eps[k]]
 m=ResolutiveMotorRailMemory(context=a.context);m.fit_scales(allstates)
 for k in train:
  states=[np.asarray(r['full_body_state']['vector'],float) for r in eps[k]];actions=[np.asarray(r['baseline_action'],float) for r in eps[k]];m.add_episode(states,actions,k)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('wb') as f:pickle.dump({'memory':m,'train_runs':train,'trace':str(a.trace)},f,pickle.HIGHEST_PROTOCOL)
 print(f'Prepared motor rail memory: rails={len(m.rails)} train_runs={train} context={m.context} out={a.out}')
if __name__=='__main__':main()
