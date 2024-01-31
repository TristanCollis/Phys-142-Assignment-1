from constants import ALPHA, X_INITIAL
from constants import X


import numpy as np


from typing import Any


def psi_initial(x: np.ndarray[float, Any]) -> np.ndarray[complex, Any]:
    return (ALPHA / np.pi) ** 0.25 * np.exp(-ALPHA / 2 * (x - X_INITIAL) ** 2)


def propagator_HO(t_b: float, t_a: float) -> np.ndarray[complex, Any]:
    x_a = X
    x_b = X

    return (2j * np.pi * np.sin(t_b - t_a)) ** -0.5 * np.exp(
        ((x_a**2 + x_b.T**2) * np.cos(t_b - t_a) - 2 * (x_a @ x_b.T))
        / (-2j * np.sin(t_b - t_a))
    )


def integrate(integrand: np.ndarray[complex, Any], dx: float) -> complex:
    return np.sum(integrand) * dx
