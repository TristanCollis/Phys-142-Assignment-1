from matplotlib import pyplot as plt
from constants import DX, T, TIME_STEPS
from helper_functions import integrate
from constants import X

import numpy as np

from typing import Any


def compute(
    psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
    potential_energy = np.zeros(steps)
    kinetic_energy = np.zeros(steps)
    total_energy = np.zeros(steps)
    psi_current = psi

    for i in range(steps):
        potential_energy[i] = integrate(0.5 * X**2 * np.abs(psi_current) ** 2, DX)

        dpsi_dx = (psi_current[2:] - psi_current[:-2]) / (2 * DX)
        dpsi_dx2 = np.abs(dpsi_dx) ** 2
        kinetic_energy[i] = integrate(0.5 * dpsi_dx2, DX)

        psi_current = K @ psi_current * DX

    total_energy = potential_energy + kinetic_energy

    return potential_energy, kinetic_energy, total_energy


def display(
    potential_energy: np.ndarray[float, Any],
    kinetic_energy: np.ndarray[float, Any],
    total_energy: np.ndarray[float, Any],
    steps: int,
) -> None:
    stride = int(TIME_STEPS / steps)
    t_axis = T[::stride]

    for quantity, label in (
        (potential_energy, "V"),
        (kinetic_energy, "K"),
        (total_energy, "E"),
    ):
        plt.plot(t_axis, quantity, label=label)
    plt.xlabel(r"$t$")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig("problem_c.png")
    plt.clf()
