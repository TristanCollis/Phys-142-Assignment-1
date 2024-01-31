from matplotlib import pyplot as plt
from constants import DX, X

import numpy as np

from typing import Any


def compute(
    psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int
) -> np.ndarray[float, Any]:
    pdf_vs_time = np.zeros((steps, psi.shape[0]))
    psi_current = psi

    for i in range(steps):
        pdf_vs_time[i] = (np.abs(psi_current) ** 2).flatten()
        psi_current = K @ psi_current * DX

    return pdf_vs_time


def display(pdf_vs_time: np.ndarray[float, Any]) -> None:
    for row in pdf_vs_time:
        plt.plot(X, row)
    plt.savefig("problem_d.png")
    plt.clf()
