#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXTERNAL=ROOT/'.external'/'BahiaRT-MujOCo-base';sys.path[:0]=[str(ROOT),str(EXTERNAL)]
from mujococodebase.agent import Agent
from memory.transition_memory import BalanceState,stability_score
from robocop.integrations.bahiart_full_body_v114 import full_body_sensor_state,MOTOR_NAMES

def cosine(a,b):
 na=np.linalg.norm(a);nb=np.linalg.norm(b);return float(np.dot(a,b)/(na*nb)) if na>1e-12 and nb>1e-12 else 0.
def velocity(cycle,block):
 return (np.array([.35,0.]),np.array([.15,.18]),np.array([-.20,0.]),np.array([.15,-.18]))[(cycle//block)%4]
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
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['baseline','original','v117'],required=True);ap.add_argument('--memory',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=60000);ap.add_argument('--number',type=int,default=2);ap.add_argument('--block',type=int,default=150);ap.add_argument('--max-cycles',type=int,default=2000);ap.add_argument('--alpha',type=float,default=.15);ap.add_argument('--max-delta',type=float,default=5.0);a=ap.parse_args()
 with a.memory.open('rb') as f:mem=pickle.load(f)
 original,fine,coarse=mem['original'],mem['fine'],mem['coarse'];demo_t=mem['demo_targets'];demo_a=mem['demo_actions']
 agent=Agent(team_name='RoboCOPABC',number=a.number,host=a.host,port=a.port,field='fifa');be=BalanceExtractor();hist=[];cycles=recalls=interventions=0;stab=[];start_xy=None;reason='MAX_CYCLES';decode_cos=[]
 def decide():
  nonlocal cycles,recalls,interventions,start_xy,reason
  if bool(agent.world.is_fallen()):reason='FALL';raise KeyboardInterrupt
  cycles+=1;s,recent=be.get(agent);fb=np.asarray(full_body_sensor_state(agent)['vector'],float);hist.append(fb.copy());hist[:]=hist[-5:]
  if start_xy is None:start_xy=np.asarray(agent.world.global_position,float)[:2].copy()
  cmd=velocity(cycles-1,a.block);agent.skills_manager.execute('Walk',target_2d=cmd,is_target_absolute=False,orientation=0.,is_orientation_absolute=False)
  base=read_action(agent);chosen=base.copy();rr=None
  if a.variant=='original':
   rr=original.recall(s,recent_state=recent,min_confidence=.65)
   if rr is not None:
    recalls+=1;gain=a.alpha*rr.confidence;delta=np.clip(rr.action-base,-a.max_delta,a.max_delta);chosen=base+gain*delta;interventions+=1
  elif a.variant=='v117' and len(hist)==5:
   rr=fine.recall(hist,min_confidence=.40)
   if rr is None:
    cr=coarse.recall([hist[-1]],min_confidence=.40)
    if cr is not None and cr.confidence>=.70:rr=cr
   if rr is not None:
    recalls+=1
    # Decode predicted recovery state to the nearest demonstrated recovery state.
    z=(demo_t-rr.target_state[None,:])/fine.scales[None,:];idx=int(np.argmin(np.mean(z*z,axis=1)));da=demo_a[idx]
    gain=a.alpha*rr.confidence;delta=np.clip(da-base,-a.max_delta,a.max_delta);chosen=base+gain*delta;interventions+=1
  write_action(agent,chosen);agent.robot.commit_motor_targets_pd();stab.append(stability_score(s,1.0))
  if cycles%100==0:print(f'[ABC {a.variant}] cycles={cycles} recalls={recalls} interventions={interventions} stability={np.mean(stab):.4f}',flush=True)
  if cycles>=a.max_cycles:reason='MAX_CYCLES';raise KeyboardInterrupt
 agent.decision_maker.update_current_behavior=decide
 try:agent.run()
 except KeyboardInterrupt:pass
 except OSError as exc:
  if getattr(exc,'errno',None)!=9:raise
 end_xy=np.asarray(agent.world.global_position,float)[:2] if start_xy is not None else np.zeros(2);disp=float(np.linalg.norm(end_xy-start_xy)) if start_xy is not None else 0.
 out={'variant':a.variant,'cycles':cycles,'reason':reason,'fallen':reason=='FALL','recalls':recalls,'interventions':interventions,'recall_rate':recalls/max(1,cycles),'mean_stability':float(np.mean(stab)) if stab else 0.,'min_stability':float(np.min(stab)) if stab else 0.,'displacement_xy':disp,'alpha':a.alpha,'max_delta':a.max_delta}
 a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
