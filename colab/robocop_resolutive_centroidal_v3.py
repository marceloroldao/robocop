# RoboCOP — Resolutive Centroidal V3
# Google Colab / Gymnasium Humanoid-v5
# Fixes V2 bootstrap: (1) chooses a stronger PD baseline automatically,
# (2) aligns each memory address with the best future state inside a horizon,
# (3) admits useful/least-degrading futures instead of requiring monotonic gain.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

SEED = 42
np.random.seed(SEED)

@dataclass
class Record:
    state: np.ndarray
    target_q: np.ndarray
    delta_state: np.ndarray
    gain: float
    future_quality: float

class ResolutiveFutureMemory:
    def __init__(self, max_records=10000):
        self.records=[]; self.max_records=max_records; self.scales=None
    def fit_scales(self, states):
        x=np.asarray(states,float); s=np.std(x,axis=0)
        self.scales=np.where(s>1e-5,s,1.0)
    def add(self,state,target_q,delta_state,gain,future_quality):
        self.records.append(Record(np.asarray(state,float).copy(),np.asarray(target_q,float).copy(),np.asarray(delta_state,float).copy(),float(gain),float(future_quality)))
        if len(self.records)>self.max_records:
            self.records.sort(key=lambda r:(r.future_quality,r.gain),reverse=True); del self.records[self.max_records:]
    def recall(self,state,max_distance=1.0):
        if not self.records or self.scales is None:return None
        x=np.asarray(state,float); best=None
        for r in self.records:
            z=(x-r.state)/self.scales; d=float(np.sqrt(np.mean(z*z)))
            if d>max_distance:continue
            score=(0.6+0.8*r.future_quality+0.7*max(r.gain,-0.05))/(1.0+d)
            if best is None or score>best[0]:best=(score,d,r)
        return best

def quaternion_tilt(q):
    q=np.asarray(q,float);q=q/max(np.linalg.norm(q),1e-9)
    return float(2*np.arccos(np.clip(abs(q[0]),0,1)))

def body_quality(data,target_z):
    z=float(data.qpos[2]);tilt=quaternion_tilt(data.qpos[3:7]);ang=float(np.linalg.norm(data.qvel[3:6]));vz=float(abs(data.qvel[2]))
    qz=np.exp(-((z-target_z)/.25)**2);qt=np.exp(-(tilt/.50)**2);qa=np.exp(-(ang/3.5)**2);qv=np.exp(-(vz/1.7)**2)
    return float(.45*qz+.30*qt+.15*qa+.10*qv)

def state_vector(data,q_ref,target_z):
    qj=np.asarray(data.qpos[7:],float);vj=np.asarray(data.qvel[6:],float);qerr=qj-q_ref;torso=np.asarray(data.qvel[:6],float)
    z=float(data.qpos[2]);tilt=quaternion_tilt(data.qpos[3:7])
    return np.concatenate([np.array([(z-target_z)/.25,tilt/.5]),torso/np.array([2,2,2,4,4,4]),qerr,vj/5.0])

def pd_action(env,q_target,kp,kd):
    d=env.unwrapped.data;qj=np.asarray(d.qpos[7:],float);vj=np.asarray(d.qvel[6:],float)
    raw=kp*(np.asarray(q_target)-qj)-kd*vj
    return np.clip(raw,env.action_space.low,env.action_space.high)

def rollout(env,q_ref,target_z,kp,kd,seed,max_steps=600,explore=False,memory=None,learn=False,horizon=8,recall_distance=1.0):
    env.reset(seed=seed); hist=[]; recalls=misses=0; initial=np.asarray(env.unwrapped.data.qpos[:2],float).copy()
    for t in range(max_steps):
        d=env.unwrapped.data;x=state_vector(d,q_ref,target_z);q=body_quality(d,target_z);desired=q_ref.copy();rr=None
        if memory is not None and memory.records:
            rr=memory.recall(x,max_distance=recall_distance)
            if rr is not None: recalls+=1; desired=rr[2].target_q.copy()
            else: misses+=1
        a=pd_action(env,desired,kp,kd)
        if explore:
            phase=2*np.pi*(t%96)/96.; excitation=.02*np.sin(phase+np.arange(a.size)*.41);a=np.clip(a+excitation,env.action_space.low,env.action_space.high)
        hist.append({'x':x.copy(),'q':q,'qj':np.asarray(d.qpos[7:],float).copy(),'step':t,'height':float(d.qpos[2]),'recall':rr is not None})
        _,_,term,trunc,_=env.step(a)
        if term or trunc: break
    final=np.asarray(env.unwrapped.data.qpos[:2],float)
    learned=0
    if learn and memory is not None and len(hist)>horizon+1:
        for i in range(len(hist)-1):
            j2=min(len(hist),i+horizon+1)
            if i+1>=j2:continue
            # Best physically observed future inside the horizon, not merely i+1.
            future=hist[i+1:j2]; best=max(future,key=lambda h:h['q'])
            gain=best['q']-hist[i]['q']
            # Admit stable or least-degrading useful futures; reject collapsed states.
            if best['q']>=0.35 and gain>=-0.03:
                memory.add(hist[i]['x'],best['qj'],best['x']-hist[i]['x'],gain,best['q']);learned+=1
    return {'steps':len(hist),'mean_quality':float(np.mean([h['q'] for h in hist])) if hist else 0.,'min_quality':float(np.min([h['q'] for h in hist])) if hist else 0.,'recalls':recalls,'misses':misses,'recall_rate':recalls/max(1,len(hist)),'learned':learned,'displacement_xy':float(np.linalg.norm(final-initial)),'history':hist}

def main():
    env=gym.make('Humanoid-v5');env.reset(seed=SEED);q_ref=np.asarray(env.unwrapped.model.qpos0[7:],float).copy();target_z=float(env.unwrapped.model.qpos0[2])
    # Small deterministic search for the best local physical stabilizer.
    candidates=[(.6,.08),(.9,.10),(1.2,.12),(1.6,.16),(2.0,.20),(2.5,.25)]
    scores=[]
    for kp,kd in candidates:
        rr=[rollout(env,q_ref,target_z,kp,kd,SEED+200+s,max_steps=400) for s in range(3)]
        mean_steps=float(np.mean([x['steps'] for x in rr]));mean_q=float(np.mean([x['mean_quality'] for x in rr]));score=mean_steps+100*mean_q
        scores.append((score,kp,kd,mean_steps,mean_q))
    scores.sort(reverse=True);_,kp,kd,_,_=scores[0]
    print('PD search:')
    for score,kpi,kdi,st,q in sorted(scores,key=lambda x:x[1]):print(f'  kp={kpi:.2f} kd={kdi:.2f} steps={st:.1f} quality={q:.4f}')
    print(f'Chosen PD: kp={kp:.2f} kd={kd:.2f}')

    baseline=rollout(env,q_ref,target_z,kp,kd,SEED+100,max_steps=1000)
    memory=ResolutiveFutureMemory();all_states=[];boot=[]
    # Collect trajectories first, then learn memory with correctly aligned best-future targets.
    for ep in range(12):
        r=rollout(env,q_ref,target_z,kp,kd,SEED+ep,max_steps=1000,explore=True,memory=None,learn=False)
        boot.append(r);all_states.extend([h['x'] for h in r['history']])
    memory.fit_scales(all_states)
    # Re-process bootstrap episodes into memory without using evaluation data.
    for r in boot:
        h=r['history'];H=8
        for i in range(len(h)-1):
            fut=h[i+1:min(len(h),i+H+1)]
            if not fut:continue
            best=max(fut,key=lambda z:z['q']);gain=best['q']-h[i]['q']
            if best['q']>=0.35 and gain>=-0.03:memory.add(h[i]['x'],best['qj'],best['x']-h[i]['x'],gain,best['q'])
    before=len(memory.records)
    resolutive=rollout(env,q_ref,target_z,kp,kd,SEED+100,max_steps=1000,memory=memory,recall_distance=1.0)
    frozen=(before==len(memory.records));env.close()

    print('\n'+'='*76);print('RoboCOP — RESOLUTIVE CENTROIDAL V3');print('='*76)
    print(f'chosen PD kp/kd       : {kp:.2f} / {kd:.2f}')
    print(f"baseline steps/quality: {baseline['steps']} / {baseline['mean_quality']:.4f}")
    print(f'bootstrap episodes    : {len(boot)}')
    print(f"bootstrap mean steps  : {np.mean([r['steps'] for r in boot]):.1f}")
    print(f'memory records        : {len(memory.records)}')
    print(f"resolutive steps      : {resolutive['steps']}")
    print(f"resolutive quality    : {resolutive['mean_quality']:.4f}")
    print(f"recalls / rate        : {resolutive['recalls']} / {100*resolutive['recall_rate']:.2f}%")
    print(f"misses                : {resolutive['misses']}")
    print(f"displacement          : {resolutive['displacement_xy']:.4f}")
    print(f"memory frozen eval    : {'PASS' if frozen else 'FAIL'}")
    print('='*76)
    plt.figure(figsize=(12,4));plt.plot([h['step'] for h in baseline['history']],[h['q'] for h in baseline['history']],label='PD baseline');plt.plot([h['step'] for h in resolutive['history']],[h['q'] for h in resolutive['history']],label='Resolutive + PD');plt.xlabel('step');plt.ylabel('quality');plt.title('V3 baseline vs resolutive');plt.legend();plt.grid();plt.show()
    return baseline,boot,resolutive,memory,scores

if __name__=='__main__': BASELINE,BOOTSTRAP,RESOLUTIVE,MEMORY,PD_SCORES=main()
