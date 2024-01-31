from constants import X

from helper_functions import psi_initial
import problem_a
import problem_b
import problem_c
import problem_d
import problem_e


def main() -> None:
    psi_0 = psi_initial(X)

    problem_a_steps = 8
    K_8eps = problem_a.compute(problem_a_steps)
    problem_a.display(K_8eps)

    problem_b_steps = 16
    x_expectation = problem_b.compute(psi_0, K_8eps, problem_b_steps)
    problem_b.display(x_expectation, problem_b_steps)

    problem_c_steps = 16
    potential_energy, kinetic_energy, total_energy = problem_c.compute(
        psi_0, K_8eps, problem_c_steps
    )
    problem_c.display(potential_energy, kinetic_energy, total_energy, problem_c_steps)

    problem_d_steps = 8
    pdf_vs_time = problem_d.compute(psi_0, K_8eps, problem_d_steps)
    problem_d.display(pdf_vs_time)

    problem_e.display(pdf_vs_time)


if __name__ == "__main__":
    main()
