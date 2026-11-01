import numpy as np
from filter import Filter

class SqrtRaisedCosineFilter(Filter):
    def __init__(self, sample_rate, symbol_rate, N, alpha):
        timestamp = (np.arange(N) - float(N - 1) / 2) / float(sample_rate)
        gain = 1.0 / (sample_rate / symbol_rate)**0.5
        h = [float(0)] * N

        for i in np.arange(N):
            t = timestamp[i]
            if abs(t) <= np.spacing(1):
                h[i] = 1.0 - alpha + (4 * alpha / np.pi)
            else:
                t = t * symbol_rate
                if abs((4 * alpha * t) ** 2 - 1.0) <= np.spacing(1):
                    h[i] = (alpha / np.sqrt(2)) * \
                           (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha)))) +
                            ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
                else:
                    h[i] = (np.sin(np.pi * t * (1 - alpha)) +
                            4 * alpha * t * np.cos(np.pi * t * (1 + alpha))) / \
                           (np.pi * t * (1 - (4 * alpha * t) ** 2))
            h[i] *= gain

        Filter.__init__(self, h, [1])

        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.N = N
        self.alpha = alpha
        return

class DsrrcFilter(Filter):
    def __init__(self, sample_rate, symbol_rate, N, alpha):
        timestamp = (np.arange(N) - float(N - 1) / 2) / float(sample_rate)
        gain = 1.0 / (sample_rate / symbol_rate) ** 0.5
        h = [float(0)] * N

        a = np.pi*(1-alpha)
        b = np.pi*(1+alpha)

        for i in np.arange(N):
            t = timestamp[i]
            if abs(t) <= np.spacing(1):
                h[i] = float(0)
            else:
                x = t * symbol_rate
                if abs(abs(x) - (float(1) / (4 * alpha))) <= np.spacing(1):
                    h[i] = symbol_rate * (DsrrcFilter.__d2N(x, alpha, a, b) * DsrrcFilter.__dD(x, alpha) - \
                            DsrrcFilter.__dN(x, alpha, a, b) * DsrrcFilter.__d2D(x, alpha)) / \
                           (8*(np.pi**2))
                else:
                    h[i] = symbol_rate * (DsrrcFilter.__dN(x, alpha, a, b) * DsrrcFilter.__D(x, alpha) - \
                            DsrrcFilter.__N(x, alpha, a, b) * DsrrcFilter.__dD(x, alpha)) / \
                           (DsrrcFilter.__D(x, alpha)**2)
            h[i] *= gain

        Filter.__init__(self, h, [1])

        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.N = N
        self.alpha = alpha
        return

    @staticmethod
    def __N(x, alpha, a, b):
        return np.sin(a*x) + 4*alpha*x*np.cos(b*x)

    @staticmethod
    def __dN(x, alpha, a, b):
        return a*np.cos(a*x) + 4*alpha*np.cos(b*x) - 4*alpha*b*x*np.sin(b*x)

    @staticmethod
    def __d2N(x, alpha, a, b):
        return -(a**2)*np.sin(a*x) - 8*alpha*b*np.sin(b*x) - 4*alpha*(b**2)*x*np.cos(b*x)

    @staticmethod
    def __D(x, alpha):
        return np.pi*x*(1-(4*alpha*x)**2)

    @staticmethod
    def __dD(x, alpha):
        return np.pi - 48*np.pi*((alpha*x)**2)

    @staticmethod
    def __d2D(x, alpha):
        return -96*np.pi*(alpha**2)*x

class PulseShapingFilter:
    def __init__(self, pulse, symbol_size):
        self._filter = pulse
        self._symbol_size = symbol_size
        return

    def step(self, x):
        y = [float(0)] * self._symbol_size
        y[0] = self._filter.step(x)
        for i in range(1, self._symbol_size):
            y[i] = self._filter.step(float(0))
        return y