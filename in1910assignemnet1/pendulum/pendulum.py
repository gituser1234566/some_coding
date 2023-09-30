from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np


class Pendulum:

    """
    Class that can be used to creat an object of type Pendulum.Calculates the motion
    of a pendulum through two differentialequation.
    Namly:
    dtheta/dt= omega
    domega/dt=-gravity/L*sin(theta)

    Arguments:
    --------------------
    self.M:any number type
        Default to 1

    self.L:any number type
        Default to 1

    self.gravity: float
                Default to 9.81

     T:float
       Stopping at Time T

    dt:float
       Distanse between two points in time

    u0: array or list or tupple
        initial values

    Returns
    -----------------------------------
    tupple of properties: Theta,omega,t,x,y,vx,vy,kinetic energy,potential energy

    """

    def __init__(self, M=1, L=1, gravity=9.81):
        self.M = M
        self.L = L
        self.gravity = gravity

    def __call__(self, t, u):
        theta, omega = u
        dvelocity = -self.gravity / self.L * np.sin(theta)
        dtheta = omega
        return (dtheta, dvelocity)

    def solve(self, u0, T, dt, angle="rad"):
        N = int(T / dt) + 1

        if angle == "deg":

            u0 = np.radians(u0)
        sol = solve_ivp(
            self.__call__, (0, T), u0, method="Radau", t_eval=np.linspace(0, T, N)
        )

        self._theta = sol.y[0]
        self._omega = sol.y[1]
        self._t = sol.t

    @property
    def t(self):
        try:
            return self._t

        except AttributeError:
            print("solve not been called ")

    @property
    def omega(self):
        try:
            return self._omega

        except AttributeError:
            print("solve not been called")

    @property
    def theta(self):
        try:
            return self._theta

        except AttributeError:
            print("solve not been called")

    @property
    def x(self):
        return self.L * np.sin(self.theta)

    @property
    def y(self):
        return -self.L * np.cos(self.theta)

    @property
    def potential(self):
        return self.M * self.gravity * (self.y + self.L)

    @property
    def vx(self):
        return np.gradient(self.x, self.t)

    @property
    def vy(self):
        return np.gradient(self.y, self.t)

    @property
    def kinetic(self):
        return 1 / 2 * self.M * (self.vx ** 2 + self.vy ** 2)


class DampenedPendulum(Pendulum):

    """
    Class that can be used to creat an object of type DampenedPendulum.Calculates the motion
    of a DampenedPendulum through two differentialequation.
    Namly:
    dtheta/dt= omega
    domega/dt=-gravity/L*sin(theta)-B/M*omega

    Arguments:
    --------------------
    self.M:any number type
        Default to 1

    self.L:any number type
        Default to 1

    self.gravity: float
                Default to 9.81

    self.B: float
            Dampened parameter

    Returns
    -----------------------------------
    inherets properties from Pendulum that returns
    tupple of: Theta,omega,t,x,y,vx,vy,kinetic enrgy,potential energy

    """

    def __init__(
        self,
        B,
        M=1,
        L=1,
        gravity=9.81,
    ):

        self.M = M
        self.L = L
        self.gravity = gravity
        self.B = B

    def __call__(self, t, u):

        theta, omega = u
        dvelocity = -self.gravity / self.L * np.sin(theta) - self.B / self.M * omega
        dtheta = omega
        return (dtheta, dvelocity)

    def solve(self, u0, T, dt, angle="rad"):
        super().solve(u0, T, dt, angle="rad")


if __name__ == "__main__":

    T = 10
    dt = 0.01
    pendulum = Pendulum()
    u0 = (1, 0)
    pendulum.solve(u0, T, dt)

    plt.plot(pendulum.t, pendulum.theta)
    plt.show()
    plt.plot(pendulum.t, pendulum.kinetic)

    plt.plot(pendulum.t, pendulum.potential)

    plt.plot(pendulum.t, pendulum.kinetic + pendulum.potential)
    plt.show()

    T = 10
    dt = 0.01
    pendulum = DampenedPendulum(0.5)
    u0 = (1, 0)
    pendulum.solve(u0, T, dt)

    plt.plot(pendulum.t, pendulum.kinetic + pendulum.potential)
    plt.show()
