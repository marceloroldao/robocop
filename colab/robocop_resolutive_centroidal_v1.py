# RoboCOP — Resolutive Centroidal/Phase Prototype
# Designed to run directly in Google Colab after installing gymnasium[mujoco].
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

SEED=42
np.random.seed(SEED)

@dataclass
class ResolutiveState:
    z:float; vz:float; tilt:float; angular:float; kinetic:float; phase:float; contact:int

class ResolutiveMemory:
    def __init__(self): self.records=[]
    def add(self,state,target,quality): self.records.append((np.asarray(state,float),np.asarray(target,float),float(quality)))
    def recall(self,state,max_distance=.40):
        if not self.records:return None
        x=np.asarray(state,float);best=None
        for s,target,q in self.records:
            d=float(np.sqrt(np.mean((x-s)**2)))
            if d<=max_distance:
                score=q/(1.+d)
                if best is None or score>best[0]:best=(score,d,target)
        return best

def run(max_steps=3000,seed=SEED):
    env=gym.make('Humanoid-v5');obs,_=env.reset(seed=seed);memory=ResolutiveMemory();previous_z=None
    def extract(phase):
        nonlocal previous_z
        d=env.unwrapped.data;z=float(d.qpos[2]);vz=0. if previous_z is None else z-previous_z;previous_z=z
        quat=np.asarray(d.qpos[3:7]);tilt=float(np.linalg.norm(quat[1:]));angular=float(np.linalg.norm(d.qvel[3:6]));kinetic=float(np.mean(np.square(d.qvel)));contact=1 if d.ncon>0 else 0
        return ResolutiveState(z,vz,tilt,angular,kinetic,phase,contact)
    def vec(s):return np.array([s.z/2.,s.vz*10.,s.tilt,s.angular/10.,s.kinetic/10.,np.sin(2*np.pi*s.phase),np.cos(2*np.pi*s.phase),float(s.contact)])
    def quality(s):return .50*np.exp(-((s.z-1.4)/.35)**2)+.30*np.exp(-(s.tilt/.6)**2)+.20*np.exp(-(s.angular/8.)**2)
    def control(target=None):
        qv=np.asarray(env.unwrapped.data.qvel);nu=env.action_space.shape[0];a=-.03*qv[-nu:]
        if target is not None:a+=.03*target[0]+.015*target[1]
        return np.clip(a,env.action_space.low,env.action_space.high)
    history=[];prev_x=None;prev_q=None;recalls=learned=0
    for t in range(max_steps):
        phase=(t%50)/50.;s=extract(phase);x=vec(s);q=quality(s);rr=memory.recall(x);target=None
        if rr is not None:recalls+=1;target=rr[2]
        _,_,term,trunc,_=env.step(control(target));ns=extract(((t+1)%50)/50.);nx=vec(ns);nq=quality(ns)
        if prev_x is not None and nq-prev_q>.002:memory.add(prev_x,nx,nq-prev_q);learned+=1
        prev_x=x.copy();prev_q=q;history.append({'step':t,'height':s.z,'quality':q,'memory':len(memory.records),'recall':rr is not None,'contact':s.contact})
        if term or trunc:break
    env.close();result={'steps':len(history),'memory':len(memory.records),'learned':learned,'recalls':recalls,'recall_rate':recalls/max(1,len(history))}
    print('\n=== RoboCOP Resolutive Centroidal V1 ===');[print(f'{k:12s}: {v}') for k,v in result.items()]
    for key,title,ylabel in [('height','Altura corporal','altura'),('quality','Coerência / estabilidade','qualidade'),('memory','Crescimento da memória resolutiva','registros')]:
        plt.figure(figsize=(12,4));plt.plot([r['step'] for r in history],[r[key] for r in history]);plt.xlabel('step');plt.ylabel(ylabel);plt.title(title);plt.grid();plt.show()
    return result,history,memory

if __name__=='__main__': RESULT,HISTORY,MEMORY=run()
