from constants import EPSILON
from helper_functions import propagator_HO
from constants import DX

import numpy as np

from typing import Any


def compute(time_steps: int) -> np.ndarray[complex, Any]:
    K_eps = propagator_HO(EPSILON, 0)
    K_8eps = DX ** (time_steps - 1) * np.linalg.matrix_power(K_eps, time_steps)

    return K_8eps


def display(K_8eps: np.ndarray[complex, Any]):
    print("K_8eps:")
    print(K_8eps)
