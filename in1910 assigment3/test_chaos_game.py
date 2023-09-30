import numpy as np
from matplotlib import pyplot as plt
import portion as p
import chaos_game as chaos
import pytest


@pytest.mark.parametrize("arg", [i for i in range(1, 3)])
def test_n_smaller_then_tree(arg):
    with pytest.raises(ValueError):
        """
        test whether ChaosGame raises error if n<3 or not.
        """
        chaos.ChaosGame(arg)


# creating two intervals which are to the right and left of (0,1)
interval1 = p.open(0, 1)
interval2 = p.open(-1, 2)
interval3 = interval2 - interval1


@pytest.mark.parametrize(
    "arg",
    [
        np.linspace(interval3[i].lower, interval3[i].upper, 10)
        for i in range(len(interval3))
    ],
)
def test_r_not_between_0_and_1(arg):
    """
    testing whether ChaosGame raises an error or not if r is not in (0,1).
    Using the two intervals to genrate 10 points in each of the intervals
    Checking if all of them raises an error.If one of them does not
    the test fails.

    """
    with pytest.raises(ValueError):

        chaos.ChaosGame(3, r=arg[0])
        chaos.ChaosGame(3, r=arg[1])


def test_if_savepng_raises_error_if_file_not_png():
    """
    Testing if method savepng raises an error if one tries
    to save the figue in another format.
    """
    instance = chaos.ChaosGame(3)
    instance.iterate(1000)
    with pytest.raises(ValueError):
        instance.savepng("chaos.jpg", color=True, cmap="nipy_spectral")


def test_if_savepng_does_not_raise_error_if_file_png():

    """
    Testing if method savepng  does not rais an error if one tries
    to save the figue in png,or without a file spesification.
    """
    instance = chaos.ChaosGame(3)
    instance.iterate(1000)

    instance.savepng("chaos.png", color=True, cmap="nipy_spectral")
    instance.savepng("chaos1", color=True, cmap="nipy_spectral")
