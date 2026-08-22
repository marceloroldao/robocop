#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--max-rows',type=int,default=3000);a=p.parse_args()
    names=None;rows=[]
    with a.trace.open(encoding='utf-8') as f:
        for line in f:
            try:r=json.loads(line)
            except:continue
            fb=r.get('full_body_state')
            if not fb:continue
            n=tuple(fb.get('names',[]))
            if names is None:names=n
            elif n!=names:raise SystemExit('FAIL: full-body schema changed inside trace')
            rows.append(np.asarray(fb['vector'],dtype=float))
            if len(rows)>=a.max_rows:break
    if len(rows)<20:raise SystemExit('FAIL: not enough full-body samples')
    x=np.stack(rows);std=np.std(x,axis=0)
    body=np.asarray([i for i,n in enumerate(names) if n.startswith('joint_pos_deg:') or n.startswith('joint_speed_deg_s:')],dtype=int)
    sensory=np.asarray([i for i in range(len(names)) if i not in set(body.tolist())],dtype=int)
    if body.size==0:raise SystemExit('FAIL: no corporal channels in schema')
    moving=int(np.count_nonzero(std[body]>1e-6)); frozen=int(body.size-moving)
    print('RoboCOP — V11 FULL-BODY TRACE VALIDATION')
    print(f'rows_checked:       {len(rows)}')
    print(f'total_channels:     {len(names)}')
    print(f'sensory_channels:   {sensory.size}')
    print(f'corporal_channels:  {body.size}')
    print(f'corporal_moving:    {moving}')
    print(f'corporal_frozen:    {frozen}')
    print(f'corporal_std_mean:  {float(np.mean(std[body])):.8f}')
    if moving < max(8,int(0.50*body.size)):
        raise SystemExit('FAIL: corporal sensor channels are frozen or incorrectly mapped')
    print('PASS: corporal sensor state is dynamic')
if __name__=='__main__':main()
