from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .field import FieldState


def _quantize(value: float, step: float) -> int:
    return int(round(value / step))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


@dataclass
class MemoryNode:
    gradient: np.ndarray
    visits: int = 1
    confidence: float = 0.35
    # A single observation has no observed angular disagreement yet.
    dispersion: float = 0.0
    mean_energy: float = 0.0
    mean_reward: float = 0.0

    def update(self, gradient: np.ndarray, energy: float, reward: float) -> None:
        gradient = np.asarray(gradient, dtype=float)
        cos = _cosine(self.gradient, gradient)
        self.dispersion = 0.90 * self.dispersion + 0.10 * (1.0 - cos)
        self.gradient = 0.85 * self.gradient + 0.15 * gradient
        self.mean_energy = 0.90 * self.mean_energy + 0.10 * energy
        self.mean_reward = 0.90 * self.mean_reward + 0.10 * reward
        self.visits += 1
        if cos > 0.75:
            self.confidence += 0.08
        elif cos > 0.40:
            self.confidence += 0.02
        else:
            self.confidence *= 0.75
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


class DescriptiveMemory:
    """Three-resolution memory inspired by the V5 Colab experiments."""

    def __init__(self) -> None:
        self.z1: Dict[Tuple[int, ...], MemoryNode] = {}
        self.z2: Dict[Tuple[int, ...], MemoryNode] = {}
        self.z3: Dict[Tuple[int, ...], MemoryNode] = {}
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

    def learn(self, s: FieldState, gradient, energy: float, reward: float) -> None:
        if self.frozen:
            return
        gradient = np.asarray(gradient, dtype=float)
        for table, key in zip((self.z1, self.z2, self.z3), self.keys(s)):
            node = table.get(key)
            if node is None:
                table[key] = MemoryNode(
                    gradient=gradient.copy(),
                    mean_energy=float(energy),
                    mean_reward=float(reward),
                )
            else:
                node.update(gradient, float(energy), float(reward))

    def lookup(self, s: FieldState):
        z1, z2, z3 = self.keys(s)
        policies = (
            (self.z1, z1, 5, 0.75, 0.18, 1),
            (self.z2, z2, 4, 0.70, 0.25, 2),
            (self.z3, z3, 3, 0.60, 0.35, 3),
        )
        for table, key, min_visits, min_conf, max_disp, level in policies:
            node = table.get(key)
            if (
                node is not None
                and node.visits >= min_visits
                and node.confidence >= min_conf
                and node.dispersion <= max_disp
            ):
                return node.gradient.copy(), level, node
        return None, 0, None

    def stats(self) -> dict:
        return {"z1_nodes": len(self.z1), "z2_nodes": len(self.z2), "z3_nodes": len(self.z3)}
