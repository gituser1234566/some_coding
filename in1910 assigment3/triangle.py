import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# 1.a

# creating weights which sum up to 1
w_hat = np.random.random(3)
w = w_hat / (w_hat).sum()

# defining the corners of the triangle
a = np.asarray([0, 0])
b = np.asarray([1, 0])

c = np.asarray([1 / 2, np.sin(np.pi / 3)])

# putting them into a list,and using this list creating an array
corners = [a, b, c]
corners_array = np.asarray(corners)

# starting point
x = w.dot(corners_array)


# 1.b

# testing if 1000 points ends up within the triangular boundary
storage_array = np.zeros(shape=(1000, 2))

for i in range(1000):
    w_hat = np.random.random(3)
    w = w_hat / (w_hat).sum()
    x = w.dot(corners_array)
    storage_array[i] = x

plt.scatter(*zip(*corners))
plt.scatter(*zip(*storage_array))

plt.show()

# iterating within the n-gon
storage_array = np.zeros(shape=(10000, 2))
for i in range(10005):
    rcorner = np.random.randint(0, 3)

    corner = corners_array[rcorner]

    x_next = (x + corner) / 2
    x = x_next

    if i > 4:

        storage_array[i - 5] = x


# 1.d

# plotting the storage array containing the points generated,through iterating within n-gon
plt.scatter(*zip(*storage_array), s=0.1)
plt.scatter(*zip(*corners))
plt.axis("equal")
plt.axis("off")
plt.show()


# 1.e

# creating a function that add color to the points
def adding_color(corner, x):
    """
    Function that takes in a starting point, and a list of corner points
    calculates the iterating points within the n-gon, storing this in an array.
    Stores another array with randomly generated integers between 0 and 2 bougth included.
    This array has the same lenght as iteration array.Then stores three sub-arrays of the iteration array
    ,depending on wetherthe random array is 0,1 or 2 on the same index.
    Finally it plots these three subarrays in one plot.

    Arguments:
        --------------------
        corner: list of corner points of 3-gon

        x:starting point
    Returns
        -----------------------------------
        returns: plot with colored points, (red ,blue ,green)
    """
    X = np.zeros(shape=(10000, 2))
    colors = np.zeros(shape=10000)

    for i in range(10005):
        rcorner = np.random.randint(0, 3)

        corner = corners_array[rcorner]

        x_next = (x + corner) / 2
        x = x_next

        if i > 4:

            X[i - 5] = x
            colors[i - 5] = rcorner

    red = X[colors == 0]
    green = X[colors == 1]
    blue = X[colors == 2]

    plt.scatter(*zip(*red), s=0.1, color="red")
    plt.scatter(*zip(*blue), s=0.1, color="blue")
    plt.scatter(*zip(*green), s=0.1, color="green")
    plt.scatter(*zip(*corners))
    plt.axis("equal")
    plt.axis("off")
    plt.show()


adding_color(corner, x)


# 1.f


def alternativ_color(corner, x):

    """

    Function that takes in a starting point, and a list of corner points.
    Uses an alternative way of generating color.


    Arguments:
        --------------------
        corner: list of corner points of 3-gon

        x:starting point
    Returns
        -----------------------------------
        returns: plot with colored points

    """
    X = np.zeros(shape=(100000, 2))

    colors = np.zeros(shape=(100000, 3))

    r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    C = [0, 0, 0]
    for i in range(100005):
        rcorner = np.random.randint(0, 3)

        corner = corners_array[rcorner]
        C_next = (C + r[rcorner]) / 2
        C = C_next

        x_next = (x + corner) / 2
        x = x_next

        if i > 4:

            X[i - 5] = x
            colors[i - 5] = C

    plt.scatter(*zip(*X), s=0.2, c=colors)
    plt.scatter(*zip(*corners))
    plt.axis("equal")
    plt.axis("off")
    plt.show()


alternativ_color(corners_array, x)
