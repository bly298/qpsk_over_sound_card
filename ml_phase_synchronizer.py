import numpy as np
from filter import Filter


class PhaseTracker:
    def __init__(self, gamma, rho):
        b = [(1+rho)*gamma, -gamma]
        a = [1, -2, 1]
        self._filter = Filter(b, a)
        self.theta_k = float(0)
        return

    def step(self, x, c):
        err_k = (x * np.exp(self.theta_k * -1.0j) * c.conjugate()).imag
        self.theta_k = self._filter.step(err_k)
        return self.theta_k, err_k

    def step_on_y(self, y, c):
        err_k = (y * c.conjugate()).imag
        self.theta_k = self._filter.step(err_k)
        return self.theta_k, err_k