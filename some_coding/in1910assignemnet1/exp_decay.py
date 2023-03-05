from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np


class ExponentialDecay:

    """
    A class that calculates the solution of the differentialequation du/dt=-a*u

    Arguments
    ---------------------------
    a:float
      Constant
    T:float
       Stopping at Time T

    dt:float
       Distanse between two points in time

    u0: array or list or tupple
        initial values

    Returns
    ----------------------------
    two array of timepoints and the solution

    """

    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        return -self.a * u

    def solve(self, u0, T, dt):
        N = int(T / dt) + 1
        sol = solve_ivp(
            self.__call__, (0, T), u0, method="Radau", t_eval=np.linspace(0, T, N)
        )
        return sol.t, sol.y[0]


if __name__ == "__main__":

    a = 5
    u0 = [0.1]
    T = 2 * np.pi
    dt = 0.001
    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)

    plt.plot(t, u)
    plt.show()