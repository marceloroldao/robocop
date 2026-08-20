# RoboCOP V4 — cumulative persistent resolutive memory / Colab
import gymnasium as gym, numpy as np, pickle, os
from pathlib import Path

SEED=42; EPISODES=40; MAX_STEPS=500; KP=1.60; KD=.16
MEMFILE=Path('/content/robocop_v4_memory.pkl')

class Memory:
 def __init__(self): self.r=[]
 def add(self,x,target,gain): self.r.append((np.asarray(x,float),np.asarray(target,float),float(gain)))
 def recall(self,x):
  if not self.r:return None
  X=np.asarray([z[0] for z in self.r]); d=np.sqrt(np.mean((X-x)**2,axis=1));
  # Resolution tightens as memory grows, but only mildly in V4.
  eps=max(.18,.42/(1.+len(self.r)/1500.)**.25); ids=np.where(d<=eps)[0]
  if not len(ids):return None
  scores=np.asarray([self.r[i][2]/(1.+d[i]) for i in ids]);i=ids[int(np.argmax(scores))];return self.r[i][1],float(d[i]),eps

def state(env,prev_z):
 d=env.unwrapped.data;z=float(d.qpos[2]);vz=0 if prev_z is None else z-prev_z;q=d.qpos;v=d.qvel
 return np.array([z/2,vz*10,*q[3:7],*np.clip(v[0:6]/10,-2,2),float(d.ncon>0)]),z

def qual(env):
 d=env.unwrapped.data;z=float(d.qpos[2]);tilt=np.linalg.norm(d.qpos[4:7]);ang=np.linalg.norm(d.qvel[3:6]);return float(.5*np.exp(-((z-1.4)/.35)**2)+.3*np.exp(-(tilt/.6)**2)+.2*np.exp(-(ang/8)**2))
def pd(env,target=None):
 d=env.unwrapped.data;nu=env.action_space.shape[0];q=d.qpos[-nu:];v=d.qvel[-nu:];a=-KP*q-KD*v
 if target is not None:
  # future target influences desired joint posture, while PD remains physical executor
  tq=np.asarray(target[-nu:]) if len(target)>=nu else None
  if tq is not None:a += .12*np.clip(tq-q,-1,1)
 return np.clip(a,env.action_space.low,env.action_space.high)

def episode(mem,seed,learn=True):
 env=gym.make('Humanoid-v5');env.reset(seed=seed);prevz=None;buf=[];rec=miss=0;qs=[]
 for t in range(MAX_STEPS):
  x,prevz=state(env,prevz);q0=qual(env);rr=mem.recall(x);target=None
  if rr is None:miss+=1
  else:target=rr[0];rec+=1
  _,_,term,trunc,_=env.step(pd(env,target));q1=qual(env);nx,_=state(env,prevz);qs.append(q0);buf.append((x,nx,q1-q0))
  if learn and len(buf)>=8:
   # best future in recent local horizon; admit useful or least-degrading experience
   window=buf[-8:];best=max(window,key=lambda z:z[2]);
   if best[2]>-.003:mem.add(best[0],best[1],best[2]+.004)
  if term or trunc:break
 env.close();return {'steps':t+1,'quality':float(np.mean(qs)),'recalls':rec,'misses':miss,'rate':rec/max(1,t+1),'records':len(mem.r)}

if MEMFILE.exists():
 try: mem=pickle.load(open(MEMFILE,'rb'));print('Loaded persistent memory:',len(mem.r),'records')
 except Exception: mem=Memory()
else:mem=Memory()
rows=[]
print('\nRoboCOP V4 — CUMULATIVE MEMORY')
print('ep   records  steps  quality  recall')
for ep in range(1,EPISODES+1):
 r=episode(mem,SEED+ep,learn=True);rows.append(r);pickle.dump(mem,open(MEMFILE,'wb'))
 print(f"{ep:02d} {r['records']:9d} {r['steps']:6d} {r['quality']:.4f} {100*r['rate']:6.2f}%")
# Frozen probe: learning off, fixed seeds
probe=[episode(mem,10000+i,learn=False) for i in range(10)]
print('\n=== FROZEN PROBE AFTER GROWTH ===')
print('records          :',len(mem.r));print('mean steps       :',np.mean([r['steps'] for r in probe]));print('mean quality     :',np.mean([r['quality'] for r in probe]));print('mean recall rate :',np.mean([r['rate'] for r in probe]));print('memory file      :',MEMFILE)
# Growth correlation is descriptive only, not causal.
if len(rows)>2:
 print('corr(records,steps):',float(np.corrcoef([r['records'] for r in rows],[r['steps'] for r in rows])[0,1]))
