#!/usr/bin/env python2
# -*- coding: utf-8 -*-

r""" Lifetime distributions for Monte Carlo aging.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

__all__ = ["LifetimeDist", "Weibull", "weibull_eta_from_lambda"]


class LifetimeDist(Protocol):
    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        ...

    def survival(self, t: np.ndarray):
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def sample_remaining(
        self, rng: np.random.Generator, age: float, size: int = 1
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Weibull(LifetimeDist):
    beta: float
    eta: float

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        return self.eta * rng.weibull(self.beta, size=size)

    def survival(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-((t / self.eta) ** self.beta))

    def mean(self) -> float:
        return self.eta * math.gamma(1.0 + 1.0 / self.beta)

    def sample_remaining(
        self, rng: np.random.Generator, age: float, size: int = 1
    ) -> np.ndarray:
        u = rng.random(size=size)
        term = (age / self.eta) ** self.beta - np.log(u)
        return self.eta * (term ** (1.0 / self.beta)) - age


def weibull_eta_from_lambda(lam: float, beta: float) -> float:
    return (1.0 / lam) / math.gamma(1.0 + 1.0 / beta)
