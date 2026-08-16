from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .field import FieldState


def _quantize(value: float, step: float) -> int:
    return int(round(value / step))


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < 1e-12 else v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class TrajectoryPrototype:
    gradient: np.ndarray
    visits: int = 1
    mean_energy: float = 0.0
    mean_reward: float = 0.0
    mean_survival: float = 0.0
    confidence: float = 0.35

    def update(self, gradient: np.ndarray, energy: float, reward: float, survival: float) -> None:
        gradient = np.asarray(gradient, dtype=float)
        self.gradient = 0.85 * self.gradient + 0.15 * gradient
        self.mean_energy = 0.90 * self.mean_energy + 0.10 * float(energy)
        self.mean_reward = 0.90 * self.mean_reward + 0.10 * float(reward)
        self.mean_survival = 0.90 * self.mean_survival + 0.10 * float(survival)
        self.visits += 1
        self.confidence = float(np.clip(self.confidence + 0.06, 0.0, 1.0))

    def quality(self, energy_scale: float = 0.01, reward_scale: float = 5.0) -> float:
        """Comparable quality score for realized outcomes.

        Survival and reward are mapped to bounded [0, 1]-like terms while energy
        remains an explicit penalty. This prevents raw reward units from drowning
        the energetic objective.
        """
        survival_term = float(np.clip(self.mean_survival, 0.0, 1.0))
        reward_term = float(np.tanh(max(self.mean_reward, 0.0) / max(reward_scale, 1e-9)))
        energy_penalty = float(np.clip(self.mean_energy / max(energy_scale, 1e-9), 0.0, 3.0))
        return 0.45 * survival_term + 0.35 * reward_term - 0.20 * energy_penalty


@dataclass
class TrajectoryNode:
    prototypes: List[TrajectoryPrototype] = field(default_factory=list)

    def learn(
        self,
        gradient: np.ndarray,
        energy: float,
        reward: float,
        survival: float,
        merge_cosine: float,
        max_prototypes: int,
    ) -> None:
        gradient = np.asarray(gradient, dtype=float)
        if not self.prototypes:
            self.prototypes.append(
                TrajectoryPrototype(
                    gradient=gradient.copy(),
                    mean_energy=float(energy),
                    mean_reward=float(reward),
                    mean_survival=float(survival),
                )
            )
            return

        similarities = [_cosine(p.gradient, gradient) for p in self.prototypes]
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= merge_cosine:
            self.prototypes[best_idx].update(gradient, energy, reward, survival)
            return

        if len(self.prototypes) < max_prototypes:
            self.prototypes.append(
                TrajectoryPrototype(
                    gradient=gradient.copy(),
                    mean_energy=float(energy),
                    mean_reward=float(reward),
                    mean_survival=float(survival),
                )
            )
            return

        # Capacity reached: update the nearest branch rather than averaging all branches.
        self.prototypes[best_idx].update(gradient, energy, reward, survival)

    def ambiguity(self) -> float:
        if len(self.prototypes) <= 1:
            return 0.0
        sims = []
        for i in range(len(self.prototypes)):
            for j in range(i + 1, len(self.prototypes)):
                sims.append(_cosine(self.prototypes[i].gradient, self.prototypes[j].gradient))
        return float(1.0 - np.mean(sims)) if sims else 0.0

    def best(self, min_visits: int, min_confidence: float):
        eligible = [
            p for p in self.prototypes
            if p.visits >= min_visits and p.confidence >= min_confidence
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda p: p.quality())


class TrajectoryMemory:
    """V6 hierarchical memory that preserves multiple trajectories per state region.

    Coarse Z1 memory is used only when the node is not directionally ambiguous.
    Ambiguous coarse nodes force progressive refinement to Z2/Z3.
    """

    def __init__(
        self,
        merge_cosine: float = 0.82,
        max_prototypes: int = 4,
        z1_max_ambiguity: float = 0.25,
        z2_max_ambiguity: float = 0.40,
    ) -> None:
        self.z1: Dict[Tuple[int, ...], TrajectoryNode] = {}
        self.z2: Dict[Tuple[int, ...], TrajectoryNode] = {}
        self.z3: Dict[Tuple[int, ...], TrajectoryNode] = {}
        self.merge_cosine = float(merge_cosine)
        self.max_prototypes = int(max_prototypes)
        self.z1_max_ambiguity = float(z1_max_ambiguity)
        self.z2_max_ambiguity = float(z2_max_ambiguity)
        self.frozen = False

    @staticmethod
    def keys(s: FieldState):
        z1 = (_quantize(s.height, 0.08), _quantize(s.vertical, 0.08))
        z2 = (
            _quantize(s.height, 0.06),
            _quantize(s.vertical, 0.06),
            _quantize(s.omega, 0.30),
        )
        z3 = (
            _quantize(s.height, 0.05),
            _quantize(s.vertical, 0.04),
            _quantize(s.omega, 0.20),
            _quantize(s.vel_z, 0.18),
        )
        return z1, z2, z3

    def freeze(self) -> None:
        self.frozen = True

    def learn(
        self,
        s: FieldState,
        gradient,
        energy: float,
        reward: float,
        survival: float,
    ) -> None:
        if self.frozen:
            return
        gradient = np.asarray(gradient, dtype=float)
        for table, key in zip((self.z1, self.z2, self.z3), self.keys(s)):
            node = table.setdefault(key, TrajectoryNode())
            node.learn(
                gradient,
                energy,
                reward,
                survival,
                merge_cosine=self.merge_cosine,
                max_prototypes=self.max_prototypes,
            )

    def lookup(self, s: FieldState):
        z1, z2, z3 = self.keys(s)
        policies = (
            (self.z1, z1, 5, 0.70, self.z1_max_ambiguity, 1),
            (self.z2, z2, 4, 0.65, self.z2_max_ambiguity, 2),
            (self.z3, z3, 3, 0.60, None, 3),
        )
        for table, key, min_visits, min_conf, max_ambiguity, level in policies:
            node = table.get(key)
            if node is None:
                continue
            if max_ambiguity is not None and node.ambiguity() > max_ambiguity:
                continue
            prototype = node.best(min_visits=min_visits, min_confidence=min_conf)
            if prototype is not None:
                return prototype.gradient.copy(), level, prototype, node.ambiguity()
        return None, 0, None, None

    def stats(self) -> dict:
        def count_prototypes(table):
            return sum(len(node.prototypes) for node in table.values())
        return {
            "z1_nodes": len(self.z1),
            "z2_nodes": len(self.z2),
            "z3_nodes": len(self.z3),
            "z1_prototypes": count_prototypes(self.z1),
            "z2_prototypes": count_prototypes(self.z2),
            "z3_prototypes": count_prototypes(self.z3),
        }
