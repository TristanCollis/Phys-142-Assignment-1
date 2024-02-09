from matplotlib import pyplot as plt
import numpy as np
from typing import Any

from animate import animate


PERIOD = 2 * np.pi
EPSILON = PERIOD/128
T_AXIS = np.linspace(0, PERIOD, 128)

ALPHA = 2

X_INITIAL = 3/4

X_MIN = -4
X_MAX = 4
PARTITIONS = 600
DX = (X_MAX - X_MIN) / PARTITIONS
X_AXIS = np.linspace(X_MIN, X_MAX, PARTITIONS + 1)[..., np.newaxis]


def psi_initial(x: np.ndarray[float, Any]) -> np.ndarray[complex, Any]:
    return (
        (ALPHA / np.pi) ** (1/4) 
        * np.exp(-ALPHA / 2 * (x - X_INITIAL)**2)
    )


def propagator_HO(t_b: float, t_a: float) -> np.ndarray[complex, Any]:
    x_a = X_AXIS
    x_b = X_AXIS

    return (
        (2j * np.pi  * np.sin(t_b - t_a)) ** -0.5
        * np.exp(
            ((x_a ** 2 + x_b.T ** 2) * np.cos(t_b - t_a) - 2 * (x_a @ x_b.T))
            / (-2j * np.sin(t_b - t_a))
        )
    )


def expectation_value(operator: np.ndarray[float, Any] | float, psi: np.ndarray[complex, Any]) -> float:
    psi_pdf = np.abs(psi * np.conjugate(psi)) ** 2
    return np.sum(operator * psi_pdf * DX)  # type: ignore


def problem_a(time_steps: int) -> np.ndarray[complex, Any]:
    K_eps = propagator_HO(EPSILON, 0)
    K_8eps = DX**(time_steps - 1) * np.linalg.matrix_power(K_eps, time_steps)

    return K_8eps


def problem_b(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> np.ndarray[float, Any]:
    result = np.zeros(steps)
    psi_current = psi

    for i in range(steps):
        result[i] = expectation_value(X_AXIS, psi)
        psi_current = (K @ psi_current) * DX

    return result


def problem_c(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
    potential = np.zeros(steps)
    kinetic = np.zeros(steps)
    energy = np.zeros(steps)
    psi_current = psi

    for i in range(steps):
        potential[i] = expectation_value(0.5 * X_AXIS**2, psi_current)
        kinetic[i] = expectation_value(0.5, psi_current)

        psi_current = K @ psi_current * DX

    energy = potential + kinetic

    return potential, kinetic, energy


def problem_d(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> np.ndarray[float, Any]:
    pdf_vs_time = np.zeros((steps, psi.shape[0]))
    psi_current = psi

    for i in range(steps):
        pdf_vs_time[i] = (np.abs(psi_current)**2).flatten()
        psi_current = K @ psi_current * DX

    return pdf_vs_time


def problem_e(psi: np.ndarray[complex, Any], K: np.ndarray[complex, Any], steps: int) -> None:
    pdf_vs_time = problem_d(psi, K, steps)
    animate(pdf_vs_time, X_AXIS)

def main() -> None:
    psi_0 = psi_initial(X_AXIS)

    K_8eps = problem_a(8)

    print("K_8eps:")
    print(K_8eps)

    x_expectation = problem_b(psi_0, K_8eps, 16)

    plt.plot(np.linspace(0, PERIOD, 16), x_expectation)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\langle x \rangle$")
    plt.show()
    plt.savefig("problem_b.png")
    plt.clf()

    potential, kinetic, energy = problem_c(psi_0, K_8eps, 16)

    plt.plot(T_AXIS[::8], potential, label="V")
    plt.plot(T_AXIS[::8], kinetic, label="K")
    plt.plot(T_AXIS[::8], energy, label="E")
    plt.legend()
    plt.savefig("problem_c.png")
    plt.clf()
    

    pdf_vs_time = problem_d(psi_0, K_8eps, 8)

    for row in pdf_vs_time:
        plt.plot(X_AXIS, row)
    plt.savefig("problem_d.png")
    plt.clf()

    problem_e(psi_0, propagator_HO(EPSILON, 0), 128)






if __name__ == "__main__":
    main()
