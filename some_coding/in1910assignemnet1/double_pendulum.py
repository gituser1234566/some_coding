import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib import animation as animation


class DoublePendulum:
    """
    Class that can be used to creat an object of type DoublePendulum.Calculates the motion
    of a DoublePendulum through four differentialequation.
    Namly:
    Too long formula

    Arguments:
    --------------------
    self.M1:any number type
        Default to 1

    self.L1:any number type
        Default to 1

     self.M2:any number type
        Default to 1

    self.L2:any number type
        Default to 1

    self.G: float
                Default to 9.81

     T:float
       Stopping at Time T

    dt:float
       Distanse between two points in time

    u0: array or list or tupple
        initial values



    Returns
    -----------------------------------

    tupple of properties: Theta1,Theta2,omega1,omega2,t,x1,x2,y1,y2,
    vx1,vy1,vx2,vy2,kinetic enrgy,potential energy

    """

    def __init__(self, G=9.81, M1=1, M2=1, L1=1, L2=1):
        self.M1 = M1
        self.M2 = M2
        self.L1 = L1
        self.L2 = L2
        self.G = G

    def __call__(self, t, u):

        theta1, omega1, theta2, omega2 = u
        delta = theta2 - theta1

        dtheta1 = omega1
        domega1 = (
            (
                (self.M2 * self.L1 * omega1 ** 2 * np.sin(delta) * np.cos(delta))
                + self.M2 * self.G * np.sin(theta2) * np.cos(delta)
                + self.L2 * self.M2 * omega2 ** 2 * np.sin(delta)
                - (self.M1 + self.M2) * self.G * np.sin(theta1)
            )
        ) / ((self.M1 + self.M2) * self.L2 - self.M2 * self.L1 * (np.cos(delta)) ** 2)

        dtheta2 = omega2
        domega2 = (
            -self.M2 * self.L2 * omega2 ** 2 * np.sin(delta) * np.cos(delta)
            + (self.M1 + self.M2) * self.G * np.sin(theta1) * np.cos(delta)
            - (self.M1 + self.M2) * self.L1 * omega1 ** 2 * np.sin(delta)
            - (self.M1 + self.M2) * self.G * np.sin(theta2)
        ) / ((self.M1 + self.M2) * self.L2 - self.M2 * self.L2 * (np.cos(delta) ** 2))
        return (dtheta1, domega1, dtheta2, domega2)

    def solve(self, u0, T, dt, angle="rad"):
        self.dt = dt
        N = int(T / dt) + 1

        if angle == "deg":

            u0 = np.radians(u0)
        sol = solve_ivp(
            self.__call__, (0, T), u0, method="Radau", t_eval=np.linspace(0, T, N)
        )

        self._theta1 = sol.y[0]
        self._omega1 = sol.y[1]
        self._theta2 = sol.y[2]
        self._omega2 = sol.y[3]
        self._t = sol.t

    @property
    def t(self):
        try:
            return self._t

        except AttributeError:
            print("call solve first")

    @property
    def omega1(self):

        try:
            return self._omega1

        except AttributeError:
            print("call solve first")

    @property
    def omega2(self):

        try:
            return self._omega2

        except AttributeError:
            print("call solve first")

    @property
    def theta1(self):
        try:
            return self._theta1

        except AttributeError:
            print("call solve first")

    @property
    def theta2(self):
        try:
            return self._theta2

        except AttributeError:
            print("call solve first")

    @property
    def x1(self):
        return self.L1 * np.sin(self.theta1)

    @property
    def y1(self):
        return -self.L1 * np.cos(self.theta1)

    @property
    def x2(self):
        return self.x1 + self.L2 * np.sin(self.theta2)

    @property
    def y2(self):
        return self.y1 - self.L2 * np.cos(self.theta2)

    @property
    def potential(self):
        return self.M1 * self.G * (self.y1 + self.L1) + self.M2 * self.G * (
            self.y2 + self.L1 + self.L2
        )

    @property
    def vx1(self):
        return np.gradient(self.x1, self.t)

    @property
    def vy1(self):
        return np.gradient(self.y1, self.t)

    @property
    def vx2(self):
        return np.gradient(self.x2, self.t)

    @property
    def vy2(self):
        return np.gradient(self.y2, self.t)

    @property
    def kinetic(self):
        return 1 / 2 * self.M1 * (self.vx1 ** 2 + self.vy1 ** 2) + 1 / 2 * self.M2 * (
            self.vx2 ** 2 + self.vy2 ** 2
        )

    def show_animation(self):
        plt.show()

    def save_animation(self, name):
        self.animation.save(name, fps=100)

    def create_animation(self):

        # Create empty figure
        fig = plt.figure()

        # Configure figure
        plt.axis("equal")
        plt.axis("off")
        plt.axis((-3, 3, -3, 3))

        # Make an "empty" plot object to be updated throughout the animation
        (self.pendulums,) = plt.plot([], [], "o-", lw=2)
        (self.tracependulum1,) = plt.plot([], [], "o-", lw=0.001)
        (self.tracependulum2,) = plt.plot([], [], "o-", lw=0.001)
        # Call FuncAnimation
        self.animation = animation.FuncAnimation(
            fig,
            self._next_frame,
            frames=range(len(self.x1)),
            repeat=None,
            interval=1000 * self.dt,
            blit=True,
        )

    def _next_frame(self, i):
        self.pendulums.set_data(
            (0, self.x1[i], self.x2[i]), (0, self.y1[i], self.y2[i])
        )

        self.tracependulum1.set_data((self.x1[:i]), (self.y1[:i]))
        self.tracependulum2.set_data((self.x2[:i]), (self.y2[:i]))

        return self.pendulums, self.tracependulum1, self.tracependulum2


if __name__ == "__main__":
    T = 10
    dt = 0.01
    pendulum = DoublePendulum()
    u01 = (np.pi, 0, np.pi, 5)
    pendulum.solve(u01, T, dt)

    pendulum.create_animation()
    pendulum.save_animation("example_simulation.mp4")

    pendulum2 = DoublePendulum()
    u01 = (np.pi, 0, np.pi, 5)
    u02 = (np.pi - 0.03, 0.04, np.pi + 0.01, 4.99)
    u03 = (np.pi - 0.02, -0.02, np.pi - 0.02, 5.01)
    a = [u01, u02, u03]

    for i in range(len(a)):
        pendulum2.solve(a[i], T, dt)
        pendulum2.create_animation()
        pendulum2.show_animation()
