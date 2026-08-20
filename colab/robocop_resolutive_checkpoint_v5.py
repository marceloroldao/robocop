# RoboCOP V5 — frozen checkpoint benchmark / Colab
# Tests whether larger REAL memory improves behavior on identical seeds,
# with PD-only and shuffled-target controls.
import gymnasium as gym
import numpy as np
from dataclasses import dataclass

SEED=42; MAX_STEPS=500; KP=1.60; KD=.16
CHECKPOINTS=[0,100,250,500,1000,1500,2300]
EVAL_SEEDS=[10000+i for i in range(20)]
TRAIN_SEEDS=[SEED+i for i in range(1,80)]

@dataclass
class Rec:
 x: np.ndarray
 target: np.ndarray
 gain: float

class Memory:
 def __init__(self,records=None):self.r=list(records or [])
 def add(self,x,target,gain):self.r.append(Rec(np.asarray(x,float),np.asarray(target,float),float(gain)))
 def recall(self,x):
  if not self.r:return None
  X=np.asarray([z.x for z in self.r]);d=np.sqrt(np.mean((X-x)**2,axis=1))
  # Global schedule retained only as a controlled V4->V5 baseline.
  eps=max(.18,.42/(1.+len(self.r)/1500.)**.25);ids=np.where(d<=eps)[0]
  if not len(ids):return None
  score=np.asarray([self.r[i].gain/(1.+d[i]) for i in ids]);i=ids[int(np.argmax(score))]
  return self.r[i].target,float(d[i]),eps

def state(env,prevz):
 d=env.unwrapped.data;z=float(d.qpos[2]);vz=0. if prevz is None else z-prevz
 return np.array([z/2.,vz*10.,*d.qpos[3:7],*np.clip(d.qvel[:6]/10.,-2,2),float(d.ncon>0)]),z

def quality(env):
 d=env.unwrapped.data;z=float(d.qpos[2]);tilt=np.linalg.norm(d.qpos[4:7]);ang=np.linalg.norm(d.qvel[3:6])
 return float(.5*np.exp(-((z-1.4)/.35)**2)+.3*np.exp(-(tilt/.6)**2)+.2*np.exp(-(ang/8.)**2))

def action(env,target=None):
 d=env.unwrapped.data;nu=env.action_space.shape[0];q=d.qpos[-nu:];v=d.qvel[-nu:];a=-KP*q-KD*v
 if target is not None and len(target)>=nu:a+=.12*np.clip(np.asarray(target[-nu:])-q,-1,1)
 return np.clip(a,env.action_space.low,env.action_space.high)

def run(mem,seed,learn=False):
 env=gym.make('Humanoid-v5');env.reset(seed=seed);prevz=None;buf=[];qs=[];rec=miss=0;ds=[]
 for t in range(MAX_STEPS):
  x,prevz=state(env,prevz);q0=quality(env);rr=None if mem is None else mem.recall(x);target=None
  if rr is None:miss+=1
  else:target=rr[0];ds.append(rr[1]);rec+=1
  _,_,term,trunc,_=env.step(action(env,target));q1=quality(env);nx,_=state(env,prevz);qs.append(q0);buf.append((x,nx,q1-q0))
  if learn and mem is not None and len(buf)>=8:
   best=max(buf[-8:],key=lambda z:z[2])
   if best[2]>-.003:mem.add(best[0],best[1],best[2]+.004)
  if term or trunc:break
 env.close();return {'steps':t+1,'quality':float(np.mean(qs)),'rate':rec/max(1,t+1),'distance':float(np.mean(ds)) if ds else np.nan}

# Build one ordered training memory; checkpoints are prefixes, so data content is nested.
master=Memory()
for s in TRAIN_SEEDS:
 if len(master.r)>=max(CHECKPOINTS):break
 run(master,s,learn=True)
print('training records available:',len(master.r))
if len(master.r)<max(CHECKPOINTS):print('WARNING: largest checkpoints will use all available records')

# PD-only control once on identical seeds.
pd=[run(None,s,False) for s in EVAL_SEEDS]
pd_steps=np.asarray([x['steps'] for x in pd]);pd_q=np.asarray([x['quality'] for x in pd])
print('\nRoboCOP V5 — FROZEN CHECKPOINT BENCHMARK')
print('checkpoint  real_steps  shuf_steps  pd_steps  real_q  shuf_q  recall  distance')
rows=[]
rng=np.random.default_rng(12345)
for n in CHECKPOINTS:
 records=master.r[:min(n,len(master.r))]
 real=Memory(records)
 # Same addresses/gains, targets permuted: tests whether state->future relation matters.
 if records:
  perm=rng.permutation(len(records));sh=[Rec(r.x.copy(),records[int(perm[i])].target.copy(),r.gain) for i,r in enumerate(records)]
 else:sh=[]
 shuffled=Memory(sh)
 R=[run(real,s,False) for s in EVAL_SEEDS];S=[run(shuffled,s,False) for s in EVAL_SEEDS]
 rs=np.asarray([x['steps'] for x in R]);ss=np.asarray([x['steps'] for x in S]);rq=np.asarray([x['quality'] for x in R]);sq=np.asarray([x['quality'] for x in S]);rates=np.asarray([x['rate'] for x in R]);dist=np.asarray([x['distance'] for x in R])
 row=(len(records),rs.mean(),ss.mean(),pd_steps.mean(),rq.mean(),sq.mean(),rates.mean(),np.nanmean(dist) if np.any(np.isfinite(dist)) else np.nan)
 rows.append(row);print(f'{row[0]:10d} {row[1]:10.2f} {row[2]:10.2f} {row[3]:8.2f} {row[4]:7.4f} {row[5]:7.4f} {100*row[6]:6.2f}% {row[7]:8.4f}')
print('\nPD quality:',float(pd_q.mean()))
# Paired effect at largest checkpoint.
n=rows[-1][0];real=Memory(master.r[:n]);R=[run(real,s,False) for s in EVAL_SEEDS];delta=np.asarray([x['steps'] for x in R])-pd_steps
print('largest checkpoint paired Δsteps vs PD: mean=',float(delta.mean()),'median=',float(np.median(delta)),'wins=',int(np.sum(delta>0)),'ties=',int(np.sum(delta==0)),'losses=',int(np.sum(delta<0)))
print('Interpretation: checkpoint growth is useful only if real memory improves on identical seeds and separates from shuffled targets.')
