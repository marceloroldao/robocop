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
class GainOutcome:
    visits: int = 0
    mean_energy: float = 0.0
    mean_reward: float = 0.0
    mean_survival: float = 0.0

    def update(self, energy: float, reward: float, survival: float) -> None:
        energy = float(energy)
        reward = float(reward)
        survival = float(survival)
        if self.visits == 0:
            self.mean_energy = energy
            self.mean_reward = reward
            self.mean_survival = survival
        else:
            alpha = 0.15
            self.mean_energy = (1.0 - alpha) * self.mean_energy + alpha * energy
            self.mean_reward = (1.0 - alpha) * self.mean_reward + alpha * reward
            self.mean_survival = (1.0 - alpha) * self.mean_survival + alpha * survival
        self.visits += 1

    def quality(self, energy_scale: float = 0.01, reward_scale: float = 5.0) -> float:
        survival_term = float(np.clip(self.mean_survival, 0.0, 1.0))
        reward_term = float(np.tanh(max(self.mean_reward, 0.0) / max(reward_scale, 1e-9)))
        energy_penalty = float(np.clip(self.mean_energy / max(energy_scale, 1e-9), 0.0, 3.0))
        return 0.45 * survival_term + 0.35 * reward_term - 0.20 * energy_penalty


@dataclass
class IntensityPrototype:
    gradient: np.ndarray
    visits: int = 1
    confidence: float = 0.35
    gains: Dict[float, GainOutcome] = field(default_factory=dict)

    def update_direction(self, gradient: np.ndarray) -> None:
        gradient = np.asarray(gradient, dtype=float)
        self.gradient = 0.85 * self.gradient + 0.15 * gradient
        self.visits += 1
        self.confidence = float(np.clip(self.confidence + 0.06, 0.0, 1.0))

    def record_gain(self, gain: float, energy: float, reward: float, survival: float) -> None:
        key = round(float(gain), 4)
        outcome = self.gains.setdefault(key, GainOutcome())
        outcome.update(energy, reward, survival)

    def best_gain(self, candidates, min_visits: int = 2, default: float = 0.20) -> float:
        eligible = [
            (gain, self.gains[gain])
            for gain in sorted(self.gains)
            if self.gains[gain].visits >= min_visits
        ]
        if not eligible:
            eligible = [(gain, self.gains[gain]) for gain in sorted(self.gains)]
        if not eligible:
            return float(default)
        return float(max(eligible, key=lambda item: item[1].quality())[0])

    def next_exploration_gain(self, candidates) -> float:
        candidates = tuple(float(g) for g in candidates)
        counts = []
        for gain in candidates:
            outcome = self.gains.get(round(gain, 4))
            counts.append(0 if outcome is None else outcome.visits)
        min_count = min(counts)
        return float(candidates[counts.index(min_count)])


@dataclass
class IntensityNode:
    prototypes: List[IntensityPrototype] = field(default_factory=list)

    def match_or_create(
        self,
        gradient: np.ndarray,
        merge_cosine: float,
        max_prototypes: int,
    ) -> IntensityPrototype:
        gradient = np.asarray(gradient, dtype=float)
        if not self.prototypes:
            proto = IntensityPrototype(gradient=gradient.copy())
            self.prototypes.append(proto)
            return proto
        similarities = [_cosine(p.gradient, gradient) for p in self.prototypes]
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= merge_cosine:
            proto = self.prototypes[best_idx]
            proto.update_direction(gradient)
            return proto
        if len(self.prototypes) < max_prototypes:
            proto = IntensityPrototype(gradient=gradient.copy())
            self.prototypes.append(proto)
            return proto
        proto = self.prototypes[best_idx]
        proto.update_direction(gradient)
        return proto

    def ambiguity(self) -> float:
        if len(self.prototypes) <= 1:
            return 0.0
        sims = []
        for i in range(len(self.prototypes)):
            for j in range(i + 1, len(self.prototypes)):
                sims.append(_cosine(self.prototypes[i].gradient, self.prototypes[j].gradient))
        return float(1.0 - np.mean(sims)) if sims else 0.0

    def best(self, min_visits: int, min_confidence: float):
        eligible = [p for p in self.prototypes if p.visits >= min_visits and p.confidence >= min_confidence]
        if not eligible:
            return None
        # Direction branch quality is derived from its best tested gain.
        return max(
            eligible,
            key=lambda p: max((o.quality() for o in p.gains.values()), default=-1e9),
        )


class IntensityTrajectoryMemory:
    """V6.2 memory: hierarchical trajectory prototypes plus learned field intensity."""

    def __init__(
        self,
        gain_candidates=(0.05, 0.10, 0.15, 0.20),
        merge_cosine: float = 0.82,
        max_prototypes: int = 4,
        z1_max_ambiguity: float = 0.25,
        z2_max_ambiguity: float = 0.40,
    ) -> None:
        self.z1: Dict[Tuple[int, ...], IntensityNode] = {}
        self.z2: Dict[Tuple[int, ...], IntensityNode] = {}
        self.z3: Dict[Tuple[int, ...], IntensityNode] = {}
        self.gain_candidates = tuple(float(g) for g in gain_candidates)
        self.merge_cosine = float(merge_cosine)
        self.max_prototypes = int(max_prototypes)
        self.z1_max_ambiguity = float(z1_max_ambiguity)
        self.z2_max_ambiguity = float(z2_max_ambiguity)
        self.frozen = False

    @staticmethod
    def keys(s: FieldState):
        z1 = (_quantize(s.height, 0.08), _quantize(s.vertical, 0.08))
        z2 = (_quantize(s.height, 0.06), _quantize(s.vertical, 0.06), _quantize(s.omega, 0.30))
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
        gain: float,
        energy: float,
        reward: float,
        survival: float,
    ) -> None:
        if self.frozen:
            return
        gradient = np.asarray(gradient, dtype=float)
        for table, key in zip((self.z1, self.z2, self.z3), self.keys(s)):
            node = table.setdefault(key, IntensityNode())
            proto = node.match_or_create(gradient, self.merge_cosine, self.max_prototypes)
            proto.record_gain(gain, energy, reward, survival)

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
            proto = node.best(min_visits=min_visits, min_confidence=min_conf)
            if proto is not None:
                gain = proto.best_gain(self.gain_candidates)
                return proto.gradient.copy(), gain, level, proto, node.ambiguity()
        return None, 0.20, 0, None, None

    def exploration_gain(self, s: FieldState, gradient: np.ndarray) -> float:
        # Use the deepest region so exploration does not mix distinct local dynamics.
        key = self.keys(s)[2]
        node = self.z3.setdefault(key, IntensityNode())
        proto = node.match_or_create(np.asarray(gradient, dtype=float), self.merge_cosine, self.max_prototypes)
        return proto.next_exploration_gain(self.gain_candidates)

    def stats(self) -> dict:
        def count_prototypes(table):
            return sum(len(node.prototypes) for node in table.values())
        def count_gain_records(table):
            return sum(sum(len(p.gains) for p in node.prototypes) for node in table.values())
        return {
            "z1_nodes": len(self.z1),
            "z2_nodes": len(self.z2),
            "z3_nodes": len(self.z3),
            "z1_prototypes": count_prototypes(self.z1),
            "z2_prototypes": count_prototypes(self.z2),
            "z3_prototypes": count_prototypes(self.z3),
            "z1_gain_records": count_gain_records(self.z1),
            "z2_gain_records": count_gain_records(self.z2),
            "z3_gain_records": count_gain_records(self.z3),
        }
