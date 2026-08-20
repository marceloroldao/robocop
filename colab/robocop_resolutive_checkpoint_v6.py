# RoboCOP V6 — causal frozen checkpoint benchmark
# Fixes V5 null intervention: stored targets now include 17 joint positions.
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
SEED=42;MAX_STEPS=500;KP=1.60;KD=.16;CHECKPOINTS=[0,100,250,500,1000,1500,2300];EVAL=[10000+i for i in range(20)];TRAIN=[SEED+i for i in range(1,100)]
@dataclass
class Rec:x:np.ndarray;target_q:np.ndarray;gain:float
class Memory:
 def __init__(self,r=None):self.r=list(r or [])
 def add(self,x,t,g):self.r.append(Rec(np.asarray(x,float),np.asarray(t,float),float(g)))
 def recall(self,x):
  if not self.r:return None
  X=np.asarray([z.x for z in self.r]);d=np.sqrt(np.mean((X-x)**2,axis=1));eps=max(.18,.42/(1+len(self.r)/1500.)**.25);ids=np.where(d<=eps)[0]
  if not len(ids):return None
  score=np.asarray([self.r[i].gain/(1+d[i]) for i in ids]);i=ids[int(np.argmax(score))];return self.r[i].target_q,float(d[i]),eps

def sense(env,pz):
 d=env.unwrapped.data;z=float(d.qpos[2]);vz=0. if pz is None else z-pz
 return np.array([z/2,vz*10,*d.qpos[3:7],*np.clip(d.qvel[:6]/10,-2,2),float(d.ncon>0)]),z

def qlt(env):
 d=env.unwrapped.data;z=float(d.qpos[2]);tilt=np.linalg.norm(d.qpos[4:7]);ang=np.linalg.norm(d.qvel[3:6]);return float(.5*np.exp(-((z-1.4)/.35)**2)+.3*np.exp(-(tilt/.6)**2)+.2*np.exp(-(ang/8)**2))
def joints(env):
 nu=env.action_space.shape[0];return np.asarray(env.unwrapped.data.qpos[-nu:],float).copy()
def act(env,tq=None):
 d=env.unwrapped.data;nu=env.action_space.shape[0];q=np.asarray(d.qpos[-nu:]);v=np.asarray(d.qvel[-nu:]);a=-KP*q-KD*v
 if tq is not None:a+=.12*np.clip(np.asarray(tq)-q,-1,1)
 return np.clip(a,env.action_space.low,env.action_space.high)
def run(mem,seed,learn=False):
 env=gym.make('Humanoid-v5');env.reset(seed=seed);pz=None;hist=[];qs=[];rec=0;ds=[]
 for t in range(MAX_STEPS):
  x,pz=sense(env,pz);q0=qlt(env);rr=None if mem is None else mem.recall(x);tq=None
  if rr is not None:tq=rr[0];rec+=1;ds.append(rr[1])
  env.step(act(env,tq));q1=qlt(env);hist.append((x,joints(env),q1-q0));qs.append(q0)
  if learn and mem is not None and len(hist)>=8:
   # address at start of window -> best future joint posture in that window
   w=hist[-8:];best=max(w,key=lambda z:z[2]);gain=best[2]
   if gain>-.003:mem.add(w[0][0],best[1],gain+.004)
  if env.unwrapped.data.qpos[2]<1.0:break
 env.close();return {'steps':t+1,'q':float(np.mean(qs)),'rate':rec/max(1,t+1),'d':float(np.mean(ds)) if ds else np.nan}
master=Memory()
for s in TRAIN:
 if len(master.r)>=2300:break
 run(master,s,True)
print('training records available:',len(master.r))
pd=[run(None,s) for s in EVAL];pds=np.array([r['steps'] for r in pd]);pdq=np.mean([r['q'] for r in pd]);rng=np.random.default_rng(12345)
print('\nRoboCOP V6 — CAUSAL FROZEN CHECKPOINT BENCHMARK')
print('checkpoint real_steps shuf_steps pd_steps real_q shuf_q recall distance')
last=None
for n in CHECKPOINTS:
 rr=master.r[:min(n,len(master.r))];real=Memory(rr);perm=rng.permutation(len(rr)) if rr else []
 sh=Memory([Rec(r.x.copy(),rr[int(perm[i])].target_q.copy(),r.gain) for i,r in enumerate(rr)]) if rr else Memory()
 R=[run(real,s) for s in EVAL];S=[run(sh,s) for s in EVAL];rs=np.array([r['steps'] for r in R]);ss=np.array([r['steps'] for r in S]);rq=np.mean([r['q'] for r in R]);sq=np.mean([r['q'] for r in S]);rate=np.mean([r['rate'] for r in R]);ds=[r['d'] for r in R];dm=np.nanmean(ds) if np.any(np.isfinite(ds)) else np.nan
 print(f'{len(rr):10d} {rs.mean():10.2f} {ss.mean():10.2f} {pds.mean():8.2f} {rq:7.4f} {sq:7.4f} {100*rate:6.2f}% {dm:8.4f}');last=rs
print('PD quality:',pdq)
d=last-pds;print('largest checkpoint paired delta vs PD: mean=',float(d.mean()),'median=',float(np.median(d)),'wins=',int((d>0).sum()),'ties=',int((d==0).sum()),'losses=',int((d<0).sum()))
print('V6 validity check: target_q dimension=',len(master.r[0].target_q) if master.r else 0,'action dimension=',gym.make('Humanoid-v5').action_space.shape[0])