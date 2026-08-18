from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from memory.corrective_trajectory_memory import CorrectiveTrajectoryMemory
from memory.transition_memory import BalanceState


@dataclass(frozen=True)
class InterpolatedCorrectiveRecall:
    action_sequence: np.ndarray
    confidence: float
    mean_distance: float
    max_distance: float
    neighbors: int
    coherence: float
    mean_gain: float
    mode: str = "INTERPOLATED"


class InterpolatedCorrectiveTrajectoryMemory(CorrectiveTrajectoryMemory):
    """V10 extension: reconstruct a correction from nearby known trajectories.

    Direct V9 recall remains authoritative. Interpolation is attempted only when
    direct recall fails. Neighbor weights combine trajectory proximity, historical
    recovery gain and confirmation count. A correction is returned only when the
    neighboring action sequences are mutually coherent enough.
    """

    def interpolate_recall(
        self,
        history: Iterable[BalanceState],
        *,
        k: int = 5,
        min_neighbors: int = 3,
        max_neighbor_distance: float = 1.25,
        temperature: float = 0.35,
        min_coherence: float = 0.70,
        min_confidence: float = 0.45,
    ) -> Optional[InterpolatedCorrectiveRecall]:
        if not self._records:
            return None

        ctx = self.context(history)
        ranked: list[tuple[float, object]] = []
        for record in self._records:
            if record.confirmations < self.min_confirmations:
                continue
            d = self._context_distance(record.context, ctx)
            if d <= max_neighbor_distance:
                ranked.append((d, record))

        ranked.sort(key=lambda x: x[0])
        ranked = ranked[: max(k, min_neighbors)]
        if len(ranked) < min_neighbors:
            return None

        # All prototypes must represent the same action shape before interpolation.
        shape = ranked[0][1].action_sequence.shape
        ranked = [(d, r) for d, r in ranked if r.action_sequence.shape == shape]
        if len(ranked) < min_neighbors:
            return None

        distances = np.asarray([d for d, _ in ranked], dtype=np.float64)
        raw_weights = []
        for d, record in ranked:
            confirmation = np.log1p(record.confirmations)
            quality = max(1e-6, record.mean_gain) * (1.0 + 0.20 * confirmation)
            raw_weights.append(np.exp(-d / max(temperature, 1e-6)) * quality)
        weights = np.asarray(raw_weights, dtype=np.float64)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        weights /= total

        actions = np.stack([r.action_sequence for _, r in ranked], axis=0)
        flat = actions.reshape(actions.shape[0], -1)

        # Coherence is mean absolute cosine agreement with the weighted consensus.
        consensus = np.sum(weights[:, None] * flat, axis=0)
        consensus_norm = float(np.linalg.norm(consensus))
        cosines = []
        for row in flat:
            norm = float(np.linalg.norm(row))
            if norm <= 1e-12 or consensus_norm <= 1e-12:
                cosines.append(0.0)
            else:
                cosines.append(float(np.dot(row, consensus) / (norm * consensus_norm)))
        coherence = float(np.mean(cosines))
        if coherence < min_coherence:
            return None

        action = np.sum(weights[:, None, None] * actions, axis=0)
        mean_distance = float(np.sum(weights * distances))
        max_distance = float(np.max(distances))
        mean_gain = float(np.sum(weights * np.asarray([r.mean_gain for _, r in ranked], dtype=float)))

        # Confidence rewards close, coherent neighborhoods with repeated evidence.
        evidence = np.asarray([r.confirmations for _, r in ranked], dtype=np.float64)
        confirmation_term = min(1.0, float(np.log1p(np.sum(evidence)) / np.log(51.0)))
        confidence = float(
            np.exp(-mean_distance)
            * (0.55 + 0.25 * coherence + 0.20 * confirmation_term)
        )
        if confidence < min_confidence:
            return None

        return InterpolatedCorrectiveRecall(
            action_sequence=action,
            confidence=confidence,
            mean_distance=mean_distance,
            max_distance=max_distance,
            neighbors=len(ranked),
            coherence=coherence,
            mean_gain=mean_gain,
        )
