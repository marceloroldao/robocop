from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.full_body_trajectory_memory import FullBodyRecall


@dataclass(frozen=True)
class DualSpaceDiagnostics:
    sensory_rms: float
    body_rms: float
    sensory_max: float
    body_max: float


class DualSpaceFullBodyMemory(IndexedFullBodyTrajectoryMemory):
    """V11.2: coupled neighborhoods in sensory and corporal spaces.

    Current V11 schema:
      sensory space = global position + quaternion + euler + gyro + accel (16)
      corporal space = 23 joint positions + 23 joint velocities (46)

    The coarse index only proposes candidates. A candidate is valid only when
    BOTH subspaces satisfy their adaptive sensor-by-sensor gates.
    """

    def __init__(self, *args, sensory_channels: int = 16,
                 sensory_gate: float = 2.10, body_gate: float = 1.90,
                 sensory_rms_limit: float = 0.72, body_rms_limit: float = 0.62,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sensory_channels = int(sensory_channels)
        self.sensory_gate = float(sensory_gate)
        self.body_gate = float(body_gate)
        self.sensory_rms_limit = float(sensory_rms_limit)
        self.body_rms_limit = float(body_rms_limit)

    def _dual_metrics(self, reference, query, density: int):
        factor = self._resolution_factor(density, reference.confirmations)
        z = self._z(reference.trajectory, query)
        zs = z[:, :self.sensory_channels]
        zb = z[:, self.sensory_channels:]
        srms = float(np.sqrt(np.mean(zs*zs)))
        brms = float(np.sqrt(np.mean(zb*zb)))
        smax = float(np.max(zs))
        bmax = float(np.max(zb))
        ok = bool(
            np.all(zs <= self.sensory_gate*factor)
            and np.all(zb <= self.body_gate*factor)
            and srms <= self.sensory_rms_limit
            and brms <= self.body_rms_limit
        )
        joint_rms = float(np.sqrt((srms*srms + brms*brms)/2.0))
        return joint_rms, max(smax,bmax), density, factor, ok, DualSpaceDiagnostics(srms,brms,smax,bmax)

    def observe(self, history: Iterable[np.ndarray], target_state: np.ndarray, recovery_gain: float) -> bool:
        traj=self.trajectory(history); target=np.asarray(target_state,dtype=float)
        if target.ndim!=1 or target.shape[0]!=traj.shape[1]: return False
        ids=self._candidate_ids(traj); density=self._local_density_indexed(traj,ids); best=None
        for i in ids:
            rms,_mx,_d,_f,ok,_diag=self._dual_metrics(self._records[i],traj,density)
            if ok and rms<=self.direct_rms and (best is None or rms<best[0]): best=(rms,i)
        self.admitted+=1
        if best is None:
            from memory.full_body_trajectory_memory import FullBodyPrototype
            self._records.append(FullBodyPrototype(traj.copy(),target.copy(),1,float(recovery_gain)))
            self._index[self._key(traj)].append(len(self._records)-1)
        else:
            r=self._records[best[1]]; n=r.confirmations+1
            r.trajectory+=(traj-r.trajectory)/n; r.target_state+=(target-r.target_state)/n
            r.confirmations=n; r.gain_sum+=float(recovery_gain); self.merged+=1; self._rebuild_index()
        return True

    def recall_with_diagnostics(self, history: Iterable[np.ndarray], *, k:int=5, min_confidence:float=0.40):
        query=self.trajectory(history); current=query[-1]; ids=self._candidate_ids(query,confirmed_only=True)
        density=self._local_density_indexed(query,ids); ranked=[]
        for i in ids:
            r=self._records[i]
            rms,mx,den,fac,ok,diag=self._dual_metrics(r,query,density)
            if ok and rms<=self.interpolation_rms: ranked.append((rms,mx,den,fac,r,diag))
        if not ranked:return None,None
        ranked.sort(key=lambda x:x[0]); rms,mx,den,fac,nearest,diag=ranked[0]
        if rms<=self.direct_rms*fac/max(self.resolution_floor,0.5):
            cterm=min(1.0,np.log1p(nearest.confirmations)/np.log(21.0)); conf=float(np.exp(-rms)*(0.72+0.28*cterm))
            if conf>=min_confidence:
                target=nearest.target_state.copy()
                return FullBodyRecall(target,target-current,conf,1.0,1,True,rms,mx,den,fac),diag
        chosen=ranked[:max(2,k)]
        corrections=np.asarray([x[4].target_state-current for x in chosen]); normed=corrections/self.scales[None,:]
        norms=np.linalg.norm(normed,axis=1); valid=norms>1e-9
        if np.count_nonzero(valid)<2:return None,None
        unit=normed[valid]/norms[valid,None]; coherence=float(np.linalg.norm(np.mean(unit,axis=0)))
        if coherence<self.min_coherence:return None,None
        weights=np.asarray([(1+max(0.,x[4].mean_gain))*np.log1p(x[4].confirmations)*np.exp(-x[0]/0.35) for x in chosen]);weights/=np.sum(weights)
        target=np.sum(np.asarray([x[4].target_state for x in chosen])*weights[:,None],axis=0)
        avg=float(np.average([x[0] for x in chosen],weights=weights));mx=float(max(x[1] for x in chosen));fac=float(np.average([x[3] for x in chosen],weights=weights));conf=float(np.exp(-avg)*coherence)
        if conf<min_confidence:return None,None
        ds=DualSpaceDiagnostics(
            float(np.average([x[5].sensory_rms for x in chosen],weights=weights)),
            float(np.average([x[5].body_rms for x in chosen],weights=weights)),
            float(max(x[5].sensory_max for x in chosen)),
            float(max(x[5].body_max for x in chosen)),
        )
        return FullBodyRecall(target,target-current,conf,coherence,len(chosen),False,avg,mx,density,fac),ds

    def recall(self, history, *, k:int=5, min_confidence:float=0.40) -> Optional[FullBodyRecall]:
        r,_=self.recall_with_diagnostics(history,k=k,min_confidence=min_confidence); return r
