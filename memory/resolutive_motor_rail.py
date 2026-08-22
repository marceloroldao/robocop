from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MotorRail:
    states: np.ndarray
    actions: np.ndarray
    run_id: int

class ResolutiveMotorRailMemory:
    """Trajectory memory for continuous motor replay.

    A recall selects an entry point in a demonstrated state trajectory. Subsequent
    motor actions are replayed while observed state remains close to the expected
    trajectory; divergence releases the rail and requires a new recall.
    """
    def __init__(self, context=3, stride=1):
        self.context=int(context); self.stride=int(stride); self.rails=[]; self.scales=None
    def fit_scales(self, states):
        x=np.asarray(states,float); s=np.std(x,axis=0); self.scales=np.where(s>1e-6,s,1.0)
    def add_episode(self, states, actions, run_id=0):
        s=np.asarray(states,float); a=np.asarray(actions,float)
        if len(s)!=len(a) or len(s)<self.context+2:return
        self.rails.append(MotorRail(s,a,int(run_id)))
    def _context_distance(self, observed, rail, j):
        q=np.asarray(observed[-self.context:],float); r=rail.states[j-self.context+1:j+1]
        z=(q-r)/self.scales
        return float(np.sqrt(np.mean(z*z)))
    def recall(self, observed, max_distance=.35):
        if self.scales is None or len(observed)<self.context:return None
        best=None
        for ri,rail in enumerate(self.rails):
            for j in range(self.context-1,len(rail.states)-1,self.stride):
                d=self._context_distance(observed,rail,j)
                if best is None or d<best[0]:best=(d,ri,j)
        if best is None or best[0]>max_distance:return None
        d,ri,j=best
        return {'distance':d,'confidence':max(0.0,1.0-d/max_distance),'rail':ri,'index':j,'action':self.rails[ri].actions[j].copy()}
    def expected_distance(self, rail_id, index, observed_state):
        rail=self.rails[int(rail_id)]
        if index>=len(rail.states):return float('inf')
        z=(np.asarray(observed_state,float)-rail.states[index])/self.scales
        return float(np.sqrt(np.mean(z*z)))
    def action_at(self, rail_id, index):
        rail=self.rails[int(rail_id)]
        return None if index>=len(rail.actions) else rail.actions[index].copy()
