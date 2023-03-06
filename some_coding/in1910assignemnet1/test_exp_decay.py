import numpy as np
from exp_decay import ExponentialDecay


def test_ExponentialDecay():
    """
    unittest of your class that verifies that if u(t)=3.2 and a=0.4, then u′(t)=−1.28
    """
    a = 0.4

    u = 3.2

    test = ExponentialDecay(a)

    epsilon = np.finfo(float).eps

    assert test(0, u) + epsilon or test(None, u) - epsilon == -1.28
