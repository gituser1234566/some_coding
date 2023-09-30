import numpy as np
from matplotlib import pyplot as plt


class ChaosGame:
    def __init__(self, n, r=1 / 2):
        """
        Class that can be used to creat an object of type ChaosGame.
        This object contains the ability to genrate n-gon structure.Filling this structure with points
        ,giving them color,and plotting them.

        Arguments:
        --------------------
        n:any number type
            Default to 1

        r: float
        Default to 1/2

        """

        self.n = n
        self.r = r
        self._X = None
        self.X = None
        self._color = None

        # error raised if n<3
        if n < 3:

            raise ValueError("n is smaller then 3")

        # error raised if r not in interval (0,1)
        if r > 1 or r < 0:

            raise ValueError("r is not between 0 and 1")

        self._generate_ngon()

    def _generate_ngon(self):

        """
        Private method which creates n points, where the angle between a point and its
        two nabours are equal,for all points

         Returns
        -----------------------------------

        returns a list of these points
        """
        one_segment = np.pi * 2 / self.n

        points = [
            (np.sin(one_segment * i) * self.r, np.cos(one_segment * i) * self.r)
            for i in range(self.n)
        ]
        return points

    def _starting_point(self):
        """
        Private method that generates starting point

         Returns
        -----------------------------------

        returns the starting point
        """
        w_hat = np.random.random(self.n)
        w = w_hat / (w_hat).sum()
        x = w.dot(self._generate_ngon())
        return x

    def iterate(self, steps, discard=5):
        """
        Method that creats steps or nr of points within the n-gon boundary.
        Discard n number of iterations.



        Arguments:
        --------------------
        steps: integer

        discard: interger
        Default to 5


        """
        self._X = np.zeros(shape=(steps, 2))
        self.X = np.zeros(shape=(steps, 2))
        self._color = np.zeros(shape=steps)

        x = self._starting_point()

        for i in range(steps + discard):
            rcorner = np.random.randint(self.n)

            corner = np.asarray(self._generate_ngon()[rcorner])

            x_next = self.r * x + (1 - self.r) * (corner)
            x = x_next

            if i > discard:

                self._X[i - 5] = x
                self.X[i - 5] = self._X[i - 5]
                self._color[i - 5] = rcorner

    def gradient_color(self, steps):
        """
        Method giving the points genrated in the method iteration. Steps must
        equal to the steps used in iteration.


        Arguments:
        --------------------
        steps: integer

         Returns
        -----------------------------------

        returns color array
        """

        self.colors = np.zeros(shape=steps)

        C = self._color[0]

        for i in range(1, steps):

            C_next = (C + self._color[i]) / 2
            C = C_next
            self.colors[i] = C

        return self.colors

    def plot(self, color=False, cmap="jet"):

        """

        Method creating plot of n-gon

        no need to call the method, since method plot is called in in bought method show and
        savepng
        
        Arguments:
        --------------------
        color: Boolean
        if true method gradient color is used,to give the plot color
        if false color is default black

        cmap: choose a cmap from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


        """
        if color == True:
            plt.scatter(
                *zip(*self._X), s=0.2, c=self.gradient_color(len(self._X)), cmap=cmap
            )

        if color == False:
            plt.scatter(*zip(*self._X), s=0.2, color="black", cmap=cmap)
        plt.scatter(*zip(*self._generate_ngon()))
        plt.axis("equal")
        plt.axis("off")

    def show(self, color=False, cmap="jet"):

        """

        Method Showing the plot


        Arguments:
        --------------------
        color: Boolean
        if true method gradient color is used,to give the plot color
        if false color is default black

        cmap: choose a cmap from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


        """

        self.plot(color=color, cmap=cmap)
        plt.show()

    def savepng(self, outfile, color=False, cmap="jet"):

        """

        Method saving the plot in png format, if outfile
        does not spesify file type, figure is saved in
        png format. Any other type will raise an error


        Arguments:
        --------------------
        outfile: string

        color: Boolean
        if true method gradient color is used,to give the plot color
        if false color is default black

        cmap: choose a cmap from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        """

        self.plot(color=color, cmap=cmap)

        s2 = "."
        if "." not in outfile:
            outfile = outfile + ".png"
            plt.savefig(outfile, format=None)
        elif outfile[outfile.index(s2) + len(s2) :] != "png":

            raise ValueError("wrong file type")

        elif "png" in outfile:
            plt.savefig(outfile, format=None)

    def plot_ngon(self):

        """
        Method that plots n-gon without color
        """
        plt.scatter(*zip(*self._starting_point()))
        plt.scatter(*zip(*self._generate_ngon()))
        plt.axis("equal")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":

    # 2.i

    # 5 instances created of type ChaosGame
    instance1 = ChaosGame(3)
    instance2 = ChaosGame(4, 1 / 3)
    instance3 = ChaosGame(5, 1 / 3)
    instance4 = ChaosGame(5, 3 / 8)
    instance5 = ChaosGame(6, 1 / 3)

    # putting these instances in a list

    instance = [instance1, instance2, instance3, instance4, instance5]

    # plotting them
    for i in range(0, 5):
        instance[i].iterate(10000)

        # instance[i].savepng(f"chaos{i}", color=True, cmap="nipy_spectral")
        instance[i].show(color=True, cmap="nipy_spectral")
