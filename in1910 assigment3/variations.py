import numpy as np
from matplotlib import pyplot as plt

import chaos_game as chaos


class variation:

    """
    Class that can be used to creat an object of type variation.
    This object contains the ability to generate different transformations of coordinates.


    Arguments:
    --------------------
    x: list,array, any type number


    y: list,array, any type number



    """

    def __init__(self, x, y, name):

        self.x = x
        self.y = y
        self.name = name

    @staticmethod
    def linear(self, x, y):
        """
        Static method that defines a linear tranformation of input
        
        Arguments:
        --------------------
        x: list,array, any type number


        y: list,array, any type number


        """
        return x, y

    @staticmethod
    def swirl(self, x, y):
        """
        Static method that defines a swirl tranformation of input
        
        Arguments:
        --------------------
        x: list,array, any type number


        y: list,array, any type number


        """
        r = np.sqrt(x ** 2 + y ** 2)
        x_t = x * np.sin(r**2) - y * np.cos(r**2)
        y_t = x * np.cos(r**2) + y * np.sin(r**2)
        return x_t, y_t

    @staticmethod
    def disc(self, x, y):
        """
        Static method that defines a disc tranformation of input
        
        Arguments:
        --------------------
        x: list,array, any type number


        y: list,array, any type number



        """
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(x, y)
        x_t = theta / np.pi * np.sin(np.pi * r)
        y_t = theta / np.pi * np.cos(np.pi * r)
        return x_t, y_t

    @staticmethod
    def handkerchief(self, x, y):
        """
        Static method that defines a handkerchief tranformation of input
        
        Arguments:
        --------------------
        x: list,array, any type number


        y: list,array, any type number



        """
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(x, y)

        x_t = r * np.sin(theta + r)
        y_t = r * np.cos(theta - r)
        return x_t, y_t

    @staticmethod
    def hyperbolic(self, x, y):
        """
        Static method that defines hyperbolic tranformation of input

        Arguments:
        --------------------
        x: list,array, any type number


        y: list,array, any type number


        """
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(x, y)

        x_t = np.sin(theta) / r
        y_t = r * np.cos(theta)

        return x_t, y_t

    @staticmethod
    def fisheye(self, x, y):
        """
        Static method that defines fisheye tranformation of input
        
        Arguments:
        --------------------
        x: list,array, any type number


        y: list,array, any type number

        """
        r = np.sqrt(x ** 2 + y ** 2)
        x_t = 2 / (1 + r) * x
        y_t = 2 / (1 + r) * y
        return y_t, x_t

    def transform(self):
        """
        Method uses the Static methods to tranform x and y coordinates

        Returns
        -----------------------------------

        returns transformed coordinates

        """

        x = self.x
        y = self.y
        _func = getattr(variation, self.name)
        func_value = _func(self, x, y)

        return func_value

    @classmethod
    def from_chaos_game(self, Chaos_game, name):
        """
        Method takes in an instance of type ChaosGame,and a name
        of the transformation,returns an instance of type variation
        with the x an y values from the ChaosGame instance.


        """
        Chaos_game.iterate(10000)

        x, y = Chaos_game.X[:, 0], Chaos_game.X[:, 1]
        x_values = x.flatten()
        y_values = y.flatten()

        return variation(x_values, -y_values, name)


def linear_combination_wrap(var_1, var_2):
    """
    Function takes in two instances of varaition,and returns a function, this function
    transforms the coordinates of bought instances and returns a linear combination of
    the tranformed points
    """

    def function(w):
        u, v = var_1.transform()
        e, q = var_2.transform()
        return w * u + (1 - w) * e, w * v + (1 - w) * q

    return function


if __name__ == "__main__":

    # 4.b

    # generating x and y values
    N = 100
    grid_values = np.linspace(-1, 1, N)
    x, y = np.meshgrid(grid_values, grid_values)
    x_values = x.flatten()
    y_values = y.flatten()

    # defining a list of names of transformations
    transformations = ["linear", "handkerchief", "swirl", "disc"]

    # creating instances for each name in tranformation
    variations = [variation(x_values, y_values, version) for version in transformations]

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i, (ax, variationss) in enumerate(zip(axs.flatten(), variations)):

        # transforming
        u, v = variationss.transform()

        # plotting
        ax.plot(u, -v, markersize=1, marker=".", linestyle="", color="black")

        ax.set_title(variationss.name)
        ax.axis("off")

    fig.savefig("variations_4b.png")
    plt.show()

    # 4.c

    # creating an instance of ChaosGame
    chaos_game = chaos.ChaosGame(4, 1 / 3)

    # creating instances for each name in tranformation,using from_chaos_game
    variations = [
        variation.from_chaos_game(chaos_game, version) for version in transformations
    ]
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i, (ax, variationss) in enumerate(zip(axs.flatten(), variations)):

        # transforming
        u, v = variationss.transform()

        # plotting
        ax.scatter(u, -v, s=0.1, c=chaos_game.gradient_color(10000))
        ax.axis("equal")
        ax.axis("off")

    plt.show()

    # 4.d

    coeffs = np.array([0, 0.33, 0.67, 1])

    variation1 = variation.from_chaos_game(chaos_game, "linear")

    variation2 = variation.from_chaos_game(chaos_game, "disc")

    variation12 = linear_combination_wrap(variation1, variation2)

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for ax, w in zip(axs.flatten(), coeffs):
        u, v = variation12(w)

        ax.scatter(u, -v, s=0.2, marker=".", c=chaos_game.gradient_color(10000))

        ax.axis("off")
    plt.show()
