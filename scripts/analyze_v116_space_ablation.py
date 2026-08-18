#!/usr/bin/env python3
"""V11.6 ablation: sensory-only vs body-only vs coupled, plus instantaneous coupled state.

Uses the corrected V11.5 dataset. Scales are fit on training episodes only. Memory is
frozen for holdout. This script intentionally reuses the indexed V11.1 memory so the
only experimental variable is the address representation.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.transition_memory import BalanceState, stability_score

def load(path):
    eps=defaultdict(list); schema=None
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:r=json.loads(line)
            except Exception:continue
            fb=r.get('full_body_state'); st=r.get('state')
            if not fb or not st:continue
            names=tuple(fb.get('names',[])); vec=np.asarray(fb.get('vector',[]),float)
            if schema is None:schema=names
            if names!=schema or vec.size!=len(schema):raise SystemExit('schema mismatch')
            eps[int(r.get('run_id',r.get('episode',0)))].append(r)
    return dict(sorted(eps.items())),schema

def bal(r):
    s=r['state'];return BalanceState(float(s['height']),float(s['roll']),float(s['pitch']),float(s['angular_speed']),float(s['vertical_speed']),float(s.get('support_margin',0.)))
def full(r):return np.asarray(r['full_body_state']['vector'],float)
def windows(rows,c,h):
    for i in range(c-1,len(rows)-h):yield rows[i-c+1:i+1],rows[i+1:i+h+1]
def classify(names):
    body=np.asarray([n.startswith('joint_pos:') or n.startswith('joint_vel:') for n in names],bool)
    return np.where(~body)[0],np.where(body)[0]
def project(v,idx):return np.asarray(v,float)[idx]

def evaluate(label,eps,tr,te,idx,context,horizon,progress):
    t0=time.time(); m=IndexedFullBodyTrajectoryMemory(context=context)
    trainvec=[project(full(r),idx) for run in tr for r in eps[run]];m.fit_scales(trainvec)
    tw=cand=corr=0
    for run in tr:
        for hist,fut in windows(eps[run],context,horizon):
            tw+=1; bh=[bal(x) for x in hist]; old=stability_score(bh[0],1.); now=stability_score(bh[-1],1.)
            if old-now<.01:continue
            cand+=1; scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));gain=float(scores[j]-now)
            if gain<.03:continue
            corr+=1;m.observe([project(full(x),idx) for x in hist],project(full(fut[j]),idx),gain)
            if progress and corr%progress==0:print(f'[{label}] train corrective={corr} proto={m.size} elapsed={time.time()-t0:.1f}s',flush=True)
    frozen=m.stats().copy(); hold=rec=0;direction=[];rms=[];mx=[];conf=[];by=defaultdict(lambda:[0,0])
    for run in te:
        for hist,fut in windows(eps[run],context,horizon):
            hold+=1;by[run][0]+=1;rr=m.recall([project(full(x),idx) for x in hist])
            if rr is None:continue
            rec+=1;by[run][1]+=1;conf.append(rr.confidence);rms.append(rr.rms_distance);mx.append(rr.max_channel_error)
            cur=project(full(hist[-1]),idx);scores=np.asarray([stability_score(bal(x),1.) for x in fut]);j=int(np.argmax(scores));true=project(full(fut[j]),idx)
            x=(rr.target_state-cur)/m.scales;y=(true-cur)/m.scales;nx=np.linalg.norm(x);ny=np.linalg.norm(y);direction.append(float(np.dot(x,y)/(nx*ny)) if nx>1e-12 and ny>1e-12 else 0.)
    return dict(label=label,context=context,channels=len(idx),address_dims=len(idx)*context,training_windows=tw,candidate_windows=cand,corrective_windows=corr,prototypes=frozen['records'],confirmed=frozen['confirmed_records'],holdout_windows=hold,recalls=rec,coverage=rec/max(1,hold),direction_mean=float(np.mean(direction)) if direction else 0.,aligned=sum(x>=.70 for x in direction),aligned_rate=sum(x>=.70 for x in direction)/max(1,len(direction)),rms=float(np.mean(rms)) if rms else 0.,maxerr=float(np.mean(mx)) if mx else 0.,confidence=float(np.mean(conf)) if conf else 0.,frozen=(frozen==m.stats()),elapsed=time.time()-t0,by=dict(by),index=m.index_stats())

def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--train-fraction',type=float,default=.70);p.add_argument('--context',type=int,default=5);p.add_argument('--horizon',type=int,default=12);p.add_argument('--progress-every',type=int,default=500);a=p.parse_args()
    eps,names=load(a.trace);ids=list(eps);split=max(1,min(len(ids)-1,int(round(len(ids)*a.train_fraction))));tr,te=ids[:split],ids[split:];sidx,bidx=classify(names);allidx=np.arange(len(names))
    if len(sidx)!=16 or len(bidx)!=46:raise SystemExit(f'expected 16 sensory + 46 body, got {len(sidx)} + {len(bidx)}')
    # Verify body is dynamic before doing expensive work.
    sample=np.asarray([full(r) for run in ids for r in eps[run]]);std=np.std(sample[:,bidx],axis=0)
    if np.count_nonzero(std>1e-8)<40:raise SystemExit('FAIL: corporal channels are not sufficiently dynamic')
    configs=[('S-only trajectory',sidx,a.context),('B-only trajectory',bidx,a.context),('SxB trajectory',allidx,a.context),('SxB instantaneous',allidx,1)]
    results=[]
    print(f'V11.6 dataset episodes={len(ids)} train={tr} holdout={te} rows={len(sample)}',flush=True)
    for label,idx,c in configs:
        print(f'\n=== {label}: channels={len(idx)} context={c} ===',flush=True);results.append(evaluate(label,eps,tr,te,idx,c,a.horizon,a.progress_every))
    print('\n'+'='*112);print('RoboCOP — V11.6 SPACE / TEMPORAL ABLATION');print('='*112)
    print(f"{'variant':24s} {'dims':>6s} {'proto':>7s} {'recall':>12s} {'cos':>8s} {'>=.70':>10s} {'rms':>8s} {'maxerr':>8s} {'sec':>8s}")
    for r in results:
        print(f"{r['label']:24s} {r['address_dims']:6d} {r['prototypes']:7d} {r['recalls']:5d}/{r['holdout_windows']:<6d} {r['direction_mean']:8.4f} {100*r['aligned_rate']:9.2f}% {r['rms']:8.4f} {r['maxerr']:8.4f} {r['elapsed']:8.1f}")
    print('\nCoverage:')
    for r in results:print(f"  {r['label']:24s}: {100*r['coverage']:.2f}% | confidence={r['confidence']:.4f} | frozen={'PASS' if r['frozen'] else 'FAIL'} | candidates/query={r['index']['mean_candidates']:.1f}")
    print('='*112)
if __name__=='__main__':main()
