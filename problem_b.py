from matplotlib import pyplot as plt
from constants import DX, PERIOD
from helper_functions import integrate
from constants import X

import numpy as np


from typing import Any


def compute(
    psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int
) -> np.ndarray[float, Any]:
    result = np.zeros(steps)
    psi_current = psi

    for i in range(steps):
        result[i] = integrate(X * np.abs(psi_current) ** 2, DX)
        psi_current = K @ psi_current * DX

    return result


def display(x_expectation, steps):
    plt.plot(np.linspace(0, PERIOD, steps), x_expectation)
    plt.xlabel("Time")
    plt.ylabel("<x>")
    plt.savefig("problem_b.png")
    plt.clf()
