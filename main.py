from matplotlib import pyplot as plt
import numpy as np
from typing import Any, Callable

from animate import animate


PERIOD = 2 * np.pi
EPSILON = PERIOD/128
TIME_STEPS = 128
T = np.linspace(0, PERIOD, TIME_STEPS)

ALPHA = 2

X_INITIAL = 3/4

X_MIN = -4
X_MAX = 4
PARTITIONS = 600
DX = (X_MAX - X_MIN) / PARTITIONS
X = np.linspace(X_MIN, X_MAX, PARTITIONS + 1)[..., np.newaxis]


def psi_initial(x: np.ndarray[float, Any]) -> np.ndarray[complex, Any]:
    return (ALPHA / np.pi) ** 0.25 * np.exp(-ALPHA / 2 * (x - X_INITIAL)**2)


def propagator_HO(t_b: float, t_a: float) -> np.ndarray[complex, Any]:
    x_a = X
    x_b = X

    return (
        (2j * np.pi * np.sin(t_b - t_a)) ** -0.5
        * np.exp(
            ((x_a ** 2 + x_b.T ** 2) * np.cos(t_b - t_a) - 2 * (x_a @ x_b.T))
            / (-2j * np.sin(t_b - t_a))
        )
    )


def integrate(integrand: np.ndarray[complex, Any], dx: float) -> complex:
    return np.sum(integrand) * dx


def problem_a(time_steps: int) -> np.ndarray[complex, Any]:
    K_eps = propagator_HO(EPSILON, 0)
    K_8eps = DX**(time_steps - 1) * np.linalg.matrix_power(K_eps, time_steps)

    return K_8eps


def problem_b(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> np.ndarray[float, Any]:
    result = np.zeros(steps)
    psi_current = psi

    for i in range(steps):
        result[i] = integrate(X * np.abs(psi_current) ** 2, DX)
        psi_current = K @ psi_current * DX

    return result


def problem_c(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
    potential_energy = np.zeros(steps)
    kinetic_energy = np.zeros(steps)
    total_energy = np.zeros(steps)
    psi_current = psi

    for i in range(steps):
        potential_energy[i] = integrate(0.5 * X ** 2 * np.abs(psi_current) ** 2, DX)

        dpsi_dx = (psi_current[2:] - psi_current[:-2]) / (2 * DX)
        dpsi_dx2 = np.abs(dpsi_dx) ** 2
        kinetic_energy[i] = integrate(0.5 * dpsi_dx2, DX)

        psi_current = K @ psi_current * DX

    total_energy = potential_energy + kinetic_energy

    return potential_energy, kinetic_energy, total_energy


def problem_d(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> np.ndarray[float, Any]:
    pdf_vs_time = np.zeros((steps, psi.shape[0]))
    psi_current = psi

    for i in range(steps):
        pdf_vs_time[i] = (np.abs(psi_current)**2).flatten()
        psi_current = K @ psi_current * DX

    return pdf_vs_time


def problem_e(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> None:
    pdf_vs_time = problem_d(psi, K, steps)
    animate(pdf_vs_time, X)


def main() -> None:
    psi_0 = psi_initial(X)

    problem_a_steps = 8
    K_8eps = problem_a(problem_a_steps)

    print("K_8eps:")
    print(K_8eps)

    problem_b_steps = 16
    x_expectation = problem_b(psi_0, K_8eps, problem_b_steps)

    plt.plot(np.linspace(0, PERIOD, problem_b_steps), x_expectation)
    plt.xlabel("Time")
    plt.ylabel("<x>")
    plt.savefig("problem_b.png")
    plt.clf()

    problem_c_steps = 16
    potential_energy, kinetic_energy, total_energy = problem_c(psi_0, K_8eps, problem_c_steps)

    stride = int(TIME_STEPS / problem_c_steps)
    t_axis = T[::stride]

    for quantity, label in (
        (potential_energy, "V"),
        (kinetic_energy, "K"),
        (total_energy, "E"),
    ):
        plt.plot(t_axis, quantity, label=label)

    plt.legend()
    plt.savefig("problem_c.png")
    plt.clf()

    problem_d_steps = 8
    pdf_vs_time = problem_d(psi_0, K_8eps, problem_d_steps)

    for row in pdf_vs_time:
        plt.plot(X, row)
    plt.savefig("problem_d.png")
    plt.clf()

    problem_e_steps = TIME_STEPS
    problem_e(psi_0, propagator_HO(EPSILON, 0), problem_e_steps)


if __name__ == "__main__":
    main()
