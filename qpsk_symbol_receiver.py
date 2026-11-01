import numpy as np
from qpsk_codec import qpsk_decode, qpsk_encode
from settings import *

class SymbReceiver:
    def __init__(self, symb_sz, phase_tracker=None, timing_tracker=None):
        self.symb_sz = symb_sz
        self.phase_tracker = phase_tracker
        self.timing_tracker = timing_tracker
        self.theta = float(0)
        self.rx_symb = []
        self.base = int(0)
        self.target = int(0)
        self.smpl_idx = int(0)
        return

    def set_base(self, base):
        self.base = int(base)
        self.target = self.base
        self.smpl_idx = int(0)
        if self.timing_tracker is not None:
            self.timing_tracker.set_base(base / float(sample_rate))
        return

    def step(self, y_smpl, dy_smpl):
        done = False
        y = complex(0, 0)
        if self.timing_tracker is not None:
            revised_y = y_smpl * np.exp(self.theta * -1j)
            revised_dy = dy_smpl * np.exp(self.theta * -1j)
            done, t, y, err, c = self.timing_tracker.step(revised_y, revised_dy)
            if done:
                self.rx_symb.append(y)
                self.theta, err = self.phase_tracker.step_on_y(y, c)
        elif self.phase_tracker is not None:
            if self.smpl_idx == self.target:
                revised_y = y_smpl * np.exp(self.theta * -1j)
                self.rx_symb.append(revised_y)
                c = qpsk_encode(qpsk_decode(revised_y)) * 0.70710678118654752440084436210485
                self.theta, err = self.phase_tracker.step_on_y(revised_y, c)
                self.target += self.symb_sz
                y = revised_y
                done = True
            self.smpl_idx += 1
        else:
            if self.smpl_idx == self.target:
                self.rx_symb.append(y_smpl)
                self.target += self.symb_sz
                y = y_smpl
                done = True
            self.smpl_idx += 1
        return done, y