from double_pendulum import DoublePendulum
import numpy as np
import pytest

G = 9.81
M1 = 1
M2 = 1
L1 = 1
L2 = 1
omega1 = 0.15
omega2 = 0.15


def delta(theta1, theta2):
    return theta2 - theta1


def domega1_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2):
    return (
        (
            (
                M2
                * L1
                * omega1 ** 2
                * np.sin(delta(theta1, theta2))
                * np.cos(delta(theta1, theta2))
            )
            + M2 * G * np.sin(theta2) * np.cos(delta(theta1, theta2))
            + L2 * M2 * omega2 ** 2 * np.sin(delta(theta1, theta2))
            - (M1 + M2) * G * np.sin(theta1)
        )
    ) / ((M1 + M2) * L2 - M2 * L1 * (np.cos(delta(theta1, theta2))) ** 2)


def domega2_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2):
    return (
        -M2
        * L2
        * omega2 ** 2
        * np.sin(delta(theta1, theta2))
        * np.cos(delta(theta1, theta2))
        + (M1 + M2) * G * np.sin(theta1) * np.cos(delta(theta1, theta2))
        - (M1 + M2) * L1 * omega1 ** 2 * np.sin(delta(theta1, theta2))
        - (M1 + M2) * G * np.sin(theta2)
    ) / ((M1 + M2) * L2 - M2 * L2 * (np.cos(delta(theta1, theta2))) ** 2)


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0),
        (0, 0.5235987755982988, 0.5235987755982988),
        (0.5235987755982988, 0, -0.5235987755982988),
        (0.5235987755982988, 0.5235987755982988, 0.0),
    ],
)
def test_delta(theta1, theta2, expected):
    assert abs(delta(theta1, theta2) - expected) < 1e-10


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5235987755982988, 3.4150779130841977),
        (0.5235987755982988, 0, -7.864794228634059),
        (0.5235987755982988, 0.5235987755982988, -4.904999999999999),
    ],
)
def test_domega1_dt(theta1, theta2, expected):
    assert (
        abs(domega1_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2) - expected)
        < 1e-10
    )


@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (0, 0, 0.0),
        (0, 0.5235987755982988, -7.8737942286340585),
        (0.5235987755982988, 0, 6.822361597534335),
        (0.5235987755982988, 0.5235987755982988, 0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    assert (
        abs(domega2_dt(M1, M2, L1, L2, theta1, theta2, omega1, omega2) - expected)
        < 1e-10
    )


def test_DoublePendulum_rest():
    """
    unit test that checks that a object of type DoublePendulum
    at rest in the equilibrium position (θ=0,ω=0) stays at rest
    """
    T = 10
    dt = 0.01
    pendulum = DoublePendulum()
    u0 = [0, 0, 0, 0]
    pendulum.solve(u0, T, dt)

    assumed_vector = np.zeros(len(pendulum.theta1))

    assert np.linalg.norm(pendulum.theta1) == np.linalg.norm(assumed_vector)
    assert np.linalg.norm(pendulum.omega1) == np.linalg.norm(assumed_vector)
    assert np.linalg.norm(pendulum.theta2) == np.linalg.norm(assumed_vector)
    assert np.linalg.norm(pendulum.omega2) == np.linalg.norm(assumed_vector)


def test_DoublePendulum_call_solve_first():
    """
    unit test that verifies that all five properties raise exceptions
    if the solve-method has not been called
    """
    pendulum = DoublePendulum()

    pendulum.t
    pendulum.theta1
    pendulum.omega1
    pendulum.theta2
    pendulum.omega2


pendulum = DoublePendulum()
T = 2 * np.pi
dt = 0.01
u0 = [1, 0, 1, 0]
pendulum.solve(u0, T, dt)


@pytest.mark.parametrize(
    "arg, expected_output",
    [[i, [(pendulum.L1) ** 2, (pendulum.L2) ** 2]] for i in range(len(pendulum.x1))],
)
def test_coordinates_DoublePendulum(arg, expected_output):

    """
    unit test that solves some motion of the pendulum
    and verifies that the radius r1^2=x1^2+y1^2 is almost equal to L1^2 at all times
    and r2^2=x2^2+y2^2 is almost equal to L2^2 at all times
    """
    epsilon = np.finfo(float).eps

    assert (pendulum.x1[arg]) ** 2 + (pendulum.y1[arg]) ** 2 + epsilon or (
        pendulum.x1[arg]
    ) ** 2 + (pendulum.y1[arg]) ** 2 - epsilon == expected_output[0]

    assert (pendulum.x2[arg]) ** 2 + (pendulum.y2[arg]) ** 2 + epsilon or (
        pendulum.x2[arg]
    ) ** 2 + (pendulum.y2[arg]) ** 2 - epsilon == expected_output[1]
