from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

from memory.full_body_trajectory_memory import FullBodyTrajectoryMemory, FullBodyRecall, FullBodyPrototype


class IndexedFullBodyTrajectoryMemory(FullBodyTrajectoryMemory):
    """V11.1 coarse-to-fine index without changing the final sensorwise gate.

    A compact projection of the full trajectory is quantized into buckets. The
    index only proposes candidates; every merge/recall still passes through the
    original FullBodyTrajectoryMemory._compatible() hard gate.
    """

    def __init__(self, *args, bucket_width: float = 0.85, probe_radius: int = 1,
                 min_candidates: int = 48, max_candidates: int = 384, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_width = float(bucket_width)
        self.probe_radius = int(probe_radius)
        self.min_candidates = int(min_candidates)
        self.max_candidates = int(max_candidates)
        self._index: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self.candidate_queries = 0
        self.candidates_examined = 0

    def _descriptor(self, trajectory: np.ndarray) -> np.ndarray:
        if self.scales is None:
            raise RuntimeError("fit_scales() must be called before using V11.1")
        # Preserve body configuration and motion direction while staying cheap:
        # normalized current state + normalized displacement across the window.
        current = trajectory[-1] / self.scales
        delta = (trajectory[-1] - trajectory[0]) / self.scales
        # Aggregate 62+62 dimensions into 16 deterministic groups.
        raw = np.concatenate([current, delta])
        groups = np.array_split(raw, min(16, raw.size))
        return np.asarray([float(np.mean(g)) for g in groups], dtype=float)

    def _key(self, trajectory: np.ndarray) -> tuple[int, ...]:
        d = self._descriptor(trajectory)
        return tuple(np.floor(d / self.bucket_width).astype(np.int32).tolist())

    def _rebuild_index(self) -> None:
        self._index = defaultdict(list)
        for i, r in enumerate(self._records):
            self._index[self._key(r.trajectory)].append(i)

    def _candidate_ids(self, query: np.ndarray, *, confirmed_only: bool = False) -> list[int]:
        self.candidate_queries += 1
        key = np.asarray(self._key(query), dtype=np.int32)
        found: set[int] = set()
        # Exact bucket first. Then one-coordinate neighboring buckets. This avoids
        # the combinatorial 3^16 expansion while covering quantization boundaries.
        keys = [tuple(key.tolist())]
        for radius in range(1, self.probe_radius + 1):
            for j in range(key.size):
                for sign in (-1, 1):
                    k = key.copy(); k[j] += sign * radius; keys.append(tuple(k.tolist()))
        for k in keys:
            for i in self._index.get(k, ()):
                if not confirmed_only or self._records[i].confirmations >= self.min_confirmations:
                    found.add(i)
                    if len(found) >= self.max_candidates:
                        break
            if len(found) >= self.max_candidates:
                break
        # Sparse region fallback: rank compact descriptors globally, but only
        # return a bounded shortlist. The expensive full trajectory gate remains exact.
        if len(found) < self.min_candidates and self._records:
            qd = self._descriptor(query)
            scored = []
            for i, r in enumerate(self._records):
                if confirmed_only and r.confirmations < self.min_confirmations:
                    continue
                rd = self._descriptor(r.trajectory)
                scored.append((float(np.mean((qd-rd)**2)), i))
            scored.sort(key=lambda x: x[0])
            for _, i in scored[:self.min_candidates]: found.add(i)
        ids = list(found)[:self.max_candidates]
        self.candidates_examined += len(ids)
        return ids

    def _local_density_indexed(self, query: np.ndarray, ids: Iterable[int]) -> int:
        return sum(self._rms(self._records[i].trajectory, query) <= self.coarse_rms for i in ids)

    def _compatible_indexed(self, reference: FullBodyPrototype, query: np.ndarray, density: int):
        factor = self._resolution_factor(density, reference.confirmations)
        z = self._z(reference.trajectory, query)
        rms = float(np.sqrt(np.mean(z*z))); mx = float(np.max(z))
        return rms, mx, density, factor, bool(np.all(z <= self.base_gate * factor))

    def observe(self, history, target_state: np.ndarray, recovery_gain: float) -> bool:
        traj = self.trajectory(history); target = np.asarray(target_state, dtype=float)
        if target.ndim != 1 or target.shape[0] != traj.shape[1]: return False
        ids = self._candidate_ids(traj)
        density = self._local_density_indexed(traj, ids)
        best = None
        for i in ids:
            rms, _mx, _d, _f, ok = self._compatible_indexed(self._records[i], traj, density)
            if ok and rms <= self.direct_rms and (best is None or rms < best[0]): best=(rms,i)
        self.admitted += 1
        if best is None:
            self._records.append(FullBodyPrototype(traj.copy(), target.copy(), 1, float(recovery_gain)))
            self._index[self._key(traj)].append(len(self._records)-1)
        else:
            r=self._records[best[1]]; n=r.confirmations+1
            r.trajectory += (traj-r.trajectory)/n; r.target_state += (target-r.target_state)/n
            r.confirmations=n; r.gain_sum += float(recovery_gain); self.merged += 1
            self._rebuild_index()  # prototype centroid moved; correctness over micro-optimization
        if len(self._records)>self.max_records:
            self._records.sort(key=lambda r:(r.confirmations,r.mean_gain)); del self._records[:len(self._records)-self.max_records]; self._rebuild_index()
        return True

    def recall(self, history, *, k:int=5, min_confidence:float=0.40) -> Optional[FullBodyRecall]:
        query=self.trajectory(history); current=query[-1]
        ids=self._candidate_ids(query, confirmed_only=True)
        density=self._local_density_indexed(query, ids); ranked=[]
        for i in ids:
            r=self._records[i]
            rms,mx,den,fac,ok=self._compatible_indexed(r,query,density)
            if ok and rms<=self.interpolation_rms: ranked.append((rms,mx,den,fac,r))
        if not ranked:return None
        ranked.sort(key=lambda x:x[0]); rms,mx,den,fac,nearest=ranked[0]
        if rms<=self.direct_rms*fac/max(self.resolution_floor,0.5):
            cterm=min(1.0,np.log1p(nearest.confirmations)/np.log(21.0)); conf=float(np.exp(-rms)*(0.72+0.28*cterm))
            if conf>=min_confidence:
                target=nearest.target_state.copy(); return FullBodyRecall(target,target-current,conf,1.0,1,True,rms,mx,den,fac)
        chosen=ranked[:max(2,k)]; corrections=np.asarray([r.target_state-current for *_x,r in chosen]); normed=corrections/self.scales[None,:]
        norms=np.linalg.norm(normed,axis=1); valid=norms>1e-9
        if np.count_nonzero(valid)<2:return None
        unit=normed[valid]/norms[valid,None]; coherence=float(np.linalg.norm(np.mean(unit,axis=0)))
        if coherence<self.min_coherence:return None
        weights=np.asarray([(1+max(0.,r.mean_gain))*np.log1p(r.confirmations)*np.exp(-d/0.35) for d,_mx,_den,_fac,r in chosen]); weights/=np.sum(weights)
        target=np.sum(np.asarray([r.target_state for *_x,r in chosen])*weights[:,None],axis=0); avg=float(np.average([x[0] for x in chosen],weights=weights)); mx=float(max(x[1] for x in chosen)); fac=float(np.average([x[3] for x in chosen],weights=weights)); conf=float(np.exp(-avg)*coherence)
        if conf<min_confidence:return None
        return FullBodyRecall(target,target-current,conf,coherence,len(chosen),False,avg,mx,density,fac)

    def index_stats(self) -> dict[str,float|int]:
        return {"buckets":len(self._index),"candidate_queries":self.candidate_queries,"candidates_examined":self.candidates_examined,"mean_candidates":self.candidates_examined/max(1,self.candidate_queries)}
