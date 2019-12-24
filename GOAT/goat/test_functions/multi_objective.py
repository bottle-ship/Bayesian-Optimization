import numpy as np


class ZDT1(object):

    def __init__(self):
        self._xi_min = 0
        self._xi_max = 1
        self._dim_var_min = 1
        self._dim_var_max = 30

    def evaluate(self, x):
        if type(x) is not np.ndarray:
            x = np.array(x)

        if np.any(x < self._xi_min) or np.any(x > self._xi_max):
            raise ValueError('%d <= xi <= %d is required' % (self._xi_min, self._xi_max))

        n_dim_var = x.shape[1]

        f1 = x[:, 0]
        g = 1 + 9.0 / (n_dim_var - 1) * np.sum(x[:, 1:], axis=1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h

        return np.array([f1, f2]).T

    def pareto_front(self, n_pareto_points=100):
        x = np.linspace(self._xi_min, self._xi_max, n_pareto_points)

        return np.array([x, 1 - np.sqrt(x)]).T
