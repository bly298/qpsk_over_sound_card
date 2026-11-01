import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np


class Filter :
    def __init__(self, b, a):
        self._b = b
        self._a = a
        self._x = [0.0] * len(b)
        self._y = [0.0] * len(a)
        return

    def __del__(self) :
        return

    def show_freqz(self, num, title=''):
        w,h = signal.freqz(self._b, self._a, num)
        amplitude = abs(h)
        phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
        plt.subplot(211)
        plt.grid()
        if (title != '') :
            plt.title(title)
        plt.plot(w, amplitude)
        plt.subplot(212)
        plt.grid()
        plt.plot(w, phase)
        plt.show()
        return

    def show_coeff(self):
        for i in range(len(self._b)):
            print(f"b[{i}] {self._b[i]}")

        for i in range(len(self._a)):
            print(f"a[{i}] {self._a[i]}")
        return

    def update_coeff(self, b):
        assert len(self._b) == len(b)
        self._b = b
        return

    def get_coeff(self):
        return self._b, self._a

    def step(self, x):
        self._x.insert(0, x)
        self._x.pop()
        self._y.insert(0, 0.0)
        self._y.pop()

        for i in range(0, len(self._b)) :
            self._y[0] += self._b[i] * self._x[i]

        for i in range(1, len(self._a)) :
            self._y[0] -= self._a[i] * self._y[i]

        self._y[0] = self._y[0] / self._a[0]
        return self._y[0]