import numpy as np

from goat.test_functions.multi_objective import ZDT1
from pyDOE2 import lhs


def test_zdt1():
    x_init = lhs(n=30, samples=64).astype(np.float32)
    print(x_init.shape)

    zdt1 = ZDT1()
    print(zdt1.evaluate(x_init).shape)


if __name__ == '__main__':
    test_zdt1()
