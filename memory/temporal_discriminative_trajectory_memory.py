from __future__ import annotations
import numpy as np
from memory.indexed_full_body_trajectory_memory import IndexedFullBodyTrajectoryMemory
from memory.full_body_trajectory_memory import FullBodyRecall

class TemporalDiscriminativeTrajectoryMemory(IndexedFullBodyTrajectoryMemory):
    """Keep the original hard geometry; use learned time×channel weights only to rank survivors."""
    def __init__(self,*args,temporal_weights=None,**kwargs):
        super().__init__(*args,**kwargs)
        w=np.asarray(temporal_weights if temporal_weights is not None else np.ones((self.context,1)),float)
        if w.ndim!=2 or w.shape[0]!=self.context: raise ValueError('temporal_weights must be context x channels')
        self.temporal_weights=w/np.mean(w)
    def _rank_distance(self,a,b):
        z=(np.asarray(a)-np.asarray(b))/self.scales[None,:]
        if self.temporal_weights.shape!=z.shape: raise ValueError(f'weight shape {self.temporal_weights.shape} != trajectory {z.shape}')
        return float(np.sqrt(np.sum(self.temporal_weights*z*z)/np.sum(self.temporal_weights)))
    def recall(self,history,*,k=5,min_confidence=.40):
        query=self.trajectory(history);current=query[-1];ids=self._candidate_ids(query,confirmed_only=True)
        density=self._local_density_indexed(query,ids);ranked=[]
        for i in ids:
            r=self._records[i];rms,mx,den,fac,ok=self._compatible_indexed(r,query,density)
            if ok and rms<=self.interpolation_rms: ranked.append((self._rank_distance(r.trajectory,query),rms,mx,den,fac,r))
        if not ranked:return None
        ranked.sort(key=lambda x:x[0]);_wr,rms,mx,den,fac,nearest=ranked[0]
        if rms<=self.direct_rms*fac/max(self.resolution_floor,.5):
            cterm=min(1.,np.log1p(nearest.confirmations)/np.log(21.));conf=float(np.exp(-rms)*(.72+.28*cterm))
            if conf>=min_confidence:
                target=nearest.target_state.copy();return FullBodyRecall(target,target-current,conf,1.,1,True,rms,mx,den,fac)
        chosen=ranked[:max(2,k)];cor=np.asarray([r.target_state-current for *_x,r in chosen]);normed=cor/self.scales[None,:];norms=np.linalg.norm(normed,axis=1);valid=norms>1e-9
        if np.count_nonzero(valid)<2:return None
        unit=normed[valid]/norms[valid,None];coh=float(np.linalg.norm(np.mean(unit,axis=0)))
        if coh<self.min_coherence:return None
        weights=np.asarray([(1+max(0.,r.mean_gain))*np.log1p(r.confirmations)*np.exp(-wd/.35) for wd,_rms,_mx,_den,_fac,r in chosen]);weights/=np.sum(weights)
        target=np.sum(np.asarray([r.target_state for *_x,r in chosen])*weights[:,None],axis=0);avg=float(np.average([x[1] for x in chosen],weights=weights));mx=float(max(x[2] for x in chosen));fac=float(np.average([x[4] for x in chosen],weights=weights));conf=float(np.exp(-avg)*coh)
        if conf<min_confidence:return None
        return FullBodyRecall(target,target-current,conf,coh,len(chosen),False,avg,mx,density,fac)
