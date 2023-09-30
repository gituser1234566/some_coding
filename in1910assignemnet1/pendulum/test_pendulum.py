from pendulum import Pendulum
import numpy as np
import pytest


def test_derivative():
    """
    unit test that verifies that your Pendulum class finds the right answer
    """

    pendulum = Pendulum(L=2.7)

    realval_dtheta = 0.15
    realval_domega = -9.81 / 2.7 * np.sin(np.pi / 6)

    assert pendulum(None, (np.pi / 6, 0.15))[0] == realval_dtheta
    assert pendulum(None, (np.pi / 6, 0.15))[1] == realval_domega


def test_Pendulum_rest():
    """
    unit test that checks that a  object of type Pendulum
     at rest in the equilibrium position (θ=0,ω=0) stays at rest
    """
    T = 10
    dt = 0.01
    pendulum = Pendulum()
    u0 = (0, 0)
    pendulum.solve(u0, T, dt)

    assumed_vector = np.zeros(len(pendulum.theta))

    assert np.linalg.norm(pendulum.theta) == np.linalg.norm(assumed_vector)
    assert np.linalg.norm(pendulum.omega) == np.linalg.norm(assumed_vector)


def test_Pendulum_call_solve_first():
    """
    unit test that verifies that all three properties raise exceptions
    if the solve-method has not been called
    """
    pendulum = Pendulum()

    pendulum.t
    pendulum.theta
    pendulum.omega


pendulum = Pendulum()
T = 2 * np.pi
dt = 0.01
u0 = [1, 0]
pendulum.solve(u0, T, dt)


@pytest.mark.parametrize(
    "arg, expected_output", [[i, (pendulum.L) ** 2] for i in range(len(pendulum.x))]
)
def test_coordinates_Pendulum(arg, expected_output):
    """
    unit test that solves some motion of the pendulum
    and verifies that the radius r^2=x^2+y^2 is almost equal to L^2 at all times
    """
    epsilon = np.finfo(float).eps

    assert (pendulum.x[arg]) ** 2 + (pendulum.y[arg]) ** 2 + epsilon or (
        pendulum.x[arg]
    ) ** 2 + (pendulum.y[arg]) ** 2 - epsilon == expected_output
