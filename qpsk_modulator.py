import numpy as np
from raised_cosine_filter import SqrtRaisedCosineFilter, PulseShapingFilter, DsrrcFilter
from settings import *


class Modulator:
    def __init__(self, freq_offset=float(0), init_phase=float(0)):
        self.symbol_size = int(sample_rate / symbol_rate)
        self.symbol_dur = 1.0 / float(symbol_rate)
        self.freq_offset = freq_offset
        self.init_phase = init_phase
        self.pulse = SqrtRaisedCosineFilter(sample_rate, symbol_rate, N, alpha)
        self.shaper = PulseShapingFilter(self.pulse, self.symbol_size)
        self.tick = int(0)
        return

    def __del__(self):
        return

    def step(self, symbol):
        samples = self.shaper.step(symbol)
        signals = [float(0)] * self.symbol_size
        for i in range(self.symbol_size):
            phase = self.init_phase + 2 * np.pi * (carrier_rate + self.freq_offset) * self.tick / sample_rate
            signals[i] = (samples[i] * np.exp(1j * phase)).real
            self.tick += 1
        return signals, samples

class CombDemodulator:
    def __init__(self, freq_offset=float(0), init_phase=float(0)):
        self.tick = int(0)
        self.mf = SqrtRaisedCosineFilter(sample_rate, symbol_rate, N, alpha)
        self.dmf = DsrrcFilter(sample_rate, symbol_rate, N, alpha)
        self.freq_offset = freq_offset
        self.init_phase = init_phase
        return

    def __del__(self):
        return

    def step(self, x):
        phase = self.init_phase + 2 * np.pi * (carrier_rate + self.freq_offset) * self.tick / sample_rate
        mixed_x = x * 2.0 * np.exp(-1j * phase)
        y = self.mf.step(mixed_x)
        dy = self.dmf.step(mixed_x)
        self.tick += 1
        return y, dy