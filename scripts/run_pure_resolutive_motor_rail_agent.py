#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXTERNAL=ROOT/'.external'/'BahiaRT-MujOCo-base';sys.path[:0]=[str(ROOT),str(EXTERNAL)]
from mujococodebase.agent import Agent
from robocop.integrations.bahiart_full_body_v114 import full_body_sensor_state,MOTOR_NAMES

def read_action(a):return np.asarray([float(a.robot.motor_targets[m]['target_position']) for m in MOTOR_NAMES],float)
def write_action(a,x):
 for m,v in zip(MOTOR_NAMES,np.asarray(x,float)):a.robot.motor_targets[m]['target_position']=float(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--memory',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--max-cycles',type=int,default=1000);p.add_argument('--entry-distance',type=float,default=.35);p.add_argument('--rail-distance',type=float,default=.45);p.add_argument('--max-step',type=float,default=2.0);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=60000);a=p.parse_args()
 with a.memory.open('rb') as f:m=pickle.load(f)['memory']
 agent=Agent(team_name='RoboCOPRAIL',number=2,host=a.host,port=a.port,field='fifa');hist=[];active=None;cycles=entries=rail_cycles=misses=releases=0;start=None;reason='MAX_CYCLES';distances=[]
 def decide():
  nonlocal active,cycles,entries,rail_cycles,misses,releases,start,reason
  if bool(agent.world.is_fallen()):reason='FALL';raise KeyboardInterrupt
  cycles+=1;x=np.asarray(full_body_sensor_state(agent)['vector'],float);hist.append(x.copy());hist[:]=hist[-m.context:]
  if start is None:start=np.asarray(agent.world.global_position,float)[:2].copy()
  target=None
  if active is not None:
   ri,j=active;d=m.expected_distance(ri,j,x);distances.append(d)
   if d<=a.rail_distance:
    target=m.action_at(ri,j);active=(ri,j+1);rail_cycles+=1
    if target is None:active=None;releases+=1
   else:active=None;releases+=1
  if active is None and target is None:
   rr=m.recall(hist,max_distance=a.entry_distance)
   if rr is not None:
    entries+=1;target=rr['action'];active=(rr['rail'],rr['index']+1)
   else:misses+=1
  current=read_action(agent)
  if target is None:target=current
  chosen=current+np.clip(np.asarray(target)-current,-a.max_step,a.max_step);write_action(agent,chosen);agent.robot.commit_motor_targets_pd()
  if cycles%50==0:print(f'[RAIL] cycles={cycles} entries={entries} rail_cycles={rail_cycles} releases={releases} misses={misses}',flush=True)
  if cycles>=a.max_cycles:raise KeyboardInterrupt
 agent.decision_maker.update_current_behavior=decide
 try:agent.run()
 except KeyboardInterrupt:pass
 except OSError as e:
  if getattr(e,'errno',None)!=9:raise
 end=np.asarray(agent.world.global_position,float)[:2] if start is not None else np.zeros(2)
 out={'controller':'pure_resolutive_motor_rail','walk_skill_used':False,'cycles':cycles,'reason':reason,'fallen':reason=='FALL','rail_entries':entries,'rail_cycles':rail_cycles,'rail_fraction':rail_cycles/max(1,cycles),'releases':releases,'misses':misses,'displacement_xy':float(np.linalg.norm(end-start)) if start is not None else 0.,'mean_tracking_distance':float(np.mean(distances)) if distances else None,'entry_distance':a.entry_distance,'rail_distance':a.rail_distance}
 a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
