#!/usr/bin/env python3
"""Pure resolutive motor-control probe.

BahiaRT supplies robot/server integration only. The Walk skill is NEVER executed.
The frozen original ResolutiveTransitionMemory directly proposes 23 motor targets.
On a memory MISS the controller holds/decays the last resolutive target; it does
not fall back to BahiaRT locomotion.
"""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXTERNAL=ROOT/'.external'/'BahiaRT-MujOCo-base';sys.path[:0]=[str(ROOT),str(EXTERNAL)]
from mujococodebase.agent import Agent
from memory.transition_memory import BalanceState,stability_score
from robocop.integrations.bahiart_full_body_v114 import MOTOR_NAMES

def read_action(agent):return np.asarray([float(agent.robot.motor_targets[m]['target_position']) for m in MOTOR_NAMES],float)
def write_action(agent,a):
 for m,v in zip(MOTOR_NAMES,np.asarray(a,float)):agent.robot.motor_targets[m]['target_position']=float(v)
class BalanceExtractor:
 def __init__(self):self.last_t=None;self.last_h=None;self.prev=None
 def get(self,agent):
  p=np.asarray(agent.world.global_position,float);e=np.asarray(agent.robot.global_orientation_euler,float);g=np.asarray(agent.robot.gyroscope,float);h=float(p[2]);t=getattr(agent.world,'server_time',None)
  try:t=float(t)
  except Exception:t=None
  vz=0.
  if t is not None and self.last_t is not None and self.last_h is not None and t>self.last_t:vz=(h-self.last_h)/(t-self.last_t)
  self.last_t=t;self.last_h=h;s=BalanceState(h,float(np.deg2rad(e[0])),float(np.deg2rad(e[1])),float(np.linalg.norm(np.deg2rad(g[:3]))),float(vz),0.);old=self.prev;self.prev=s;return s,old

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--memory',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=60000);ap.add_argument('--number',type=int,default=2);ap.add_argument('--max-cycles',type=int,default=1000);ap.add_argument('--min-confidence',type=float,default=.65);ap.add_argument('--max-step',type=float,default=2.0);ap.add_argument('--hold-decay',type=float,default=.995);a=ap.parse_args()
 with a.memory.open('rb') as f:payload=pickle.load(f)
 memory=payload['original']
 agent=Agent(team_name='RoboCOPPURE',number=a.number,host=a.host,port=a.port,field='fifa');be=BalanceExtractor();cycles=recalls=misses=0;layers={};stab=[];start_xy=None;reason='MAX_CYCLES';last=None
 def decide():
  nonlocal cycles,recalls,misses,start_xy,reason,last
  if bool(agent.world.is_fallen()):reason='FALL';raise KeyboardInterrupt
  cycles+=1;s,recent=be.get(agent)
  if start_xy is None:start_xy=np.asarray(agent.world.global_position,float)[:2].copy()
  current=read_action(agent)
  rr=memory.recall(s,recent_state=recent,min_confidence=a.min_confidence)
  if rr is not None:
   recalls+=1;layers[rr.layer]=layers.get(rr.layer,0)+1;target=np.asarray(rr.action,float);last=target.copy()
  else:
   misses+=1
   if last is None:target=current.copy()
   else:
    # No external locomotion fallback: retain learned motor pattern while
    # gently relaxing toward neutral to avoid an indefinitely frozen command.
    last=a.hold_decay*last;target=last.copy()
  delta=np.clip(target-current,-a.max_step,a.max_step);chosen=current+delta
  write_action(agent,chosen);agent.robot.commit_motor_targets_pd();stab.append(stability_score(s,1.0))
  if cycles%50==0:print(f'[PURE] cycles={cycles} recalls={recalls} misses={misses} layers={layers} stability={np.mean(stab):.4f}',flush=True)
  if cycles>=a.max_cycles:reason='MAX_CYCLES';raise KeyboardInterrupt
 agent.decision_maker.update_current_behavior=decide
 try:agent.run()
 except KeyboardInterrupt:pass
 except OSError as exc:
  if getattr(exc,'errno',None)!=9:raise
 end_xy=np.asarray(agent.world.global_position,float)[:2] if start_xy is not None else np.zeros(2);disp=float(np.linalg.norm(end_xy-start_xy)) if start_xy is not None else 0.
 out={'controller':'pure_resolutive_original_memory','walk_skill_used':False,'cycles':cycles,'reason':reason,'fallen':reason=='FALL','recalls':recalls,'misses':misses,'recall_rate':recalls/max(1,cycles),'layers':layers,'mean_stability':float(np.mean(stab)) if stab else 0.,'min_stability':float(np.min(stab)) if stab else 0.,'displacement_xy':disp,'min_confidence':a.min_confidence,'max_step':a.max_step,'hold_decay':a.hold_decay}
 a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
