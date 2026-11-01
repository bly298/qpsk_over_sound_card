import numpy as np
from cubic_interpolator import CubicInterpolator
from filter import Filter
from qpsk_codec import qpsk_decode, qpsk_encode
from raised_cosine_filter import DsrrcFilter, SqrtRaisedCosineFilter
from settings import *


class TimingError:
    def __init__(self, dh, c_ref, dy, D, idx):
        self.dh = dh
        self.c_ref = c_ref
        self.c = c_ref[-1]
        self.dy = dy
        self.subject_c_idx = len(c_ref) - 1
        self.idx = idx
        self.err = 0
        self.done = False
        self.D = D

        if len(c_ref) - 1 >= self.D:
            self.preceding_c_num = self.D
        else:
            self.preceding_c_num = len(c_ref) - 1

        center = int((len(dh) - 1) / 2)
        self.first_c_idx = self.subject_c_idx - self.preceding_c_num
        self.last_dh_idx = center + self.preceding_c_num
        self.expected_c_num = len(c_ref) + self.D
        return

    def calc(self, show_detail):
        if len(self.c_ref) < self.expected_c_num:
            self.expected_c_num = len(self.c_ref)

        cross_sum = 0
        dh_idx = self.last_dh_idx

        for i in range(self.first_c_idx, self.expected_c_num):
            cross_sum += self.c_ref[i] * self.dh[dh_idx]
            dh_idx -= 1

        self.err =  (self.c.conjugate() * (self.dy - cross_sum)).real

        if show_detail:
            subsequent_c_num = self.expected_c_num - self.first_c_idx - 1 - self.preceding_c_num
            print(f"{[self.idx]} {self.preceding_c_num} {subsequent_c_num}, {self.first_c_idx} {self.expected_c_num}: {self.err}")
        return

    def update(self, show_detail):
        if len(self.c_ref) >= self.expected_c_num:
            self.calc(show_detail)
            self.done = True
        return

class TimingTracker:
    def __init__(self, revised_gain, D=2, show_err=True):
        self.group_delay = int((N - 1) / 2) * 2
        symbol_size = int(sample_rate / symbol_rate)
        self.y_intp = CubicInterpolator(sample_rate, symbol_size)
        self.dy_intp = CubicInterpolator(sample_rate, symbol_size)
        self.dh = TimingTracker.build_dh()
        self.D = D
        self.y = []
        self.c = []
        self.te = []
        self.base = None
        self.target = None
        self.filter = Filter([revised_gain], [1, -1])
        self.show_err = show_err
        return

    @staticmethod
    def build_dh():
        dmf_filter = DsrrcFilter(sample_rate, symbol_rate, N, alpha)
        dmf_b, a = dmf_filter.get_coeff()
        mf_filter = SqrtRaisedCosineFilter(sample_rate, symbol_rate, N, alpha)
        mf_b, a = mf_filter.get_coeff()
        dh_b = np.convolve(dmf_b, mf_b)

        center = int((len(dh_b) - 1) / 2)
        symbol_size = int(sample_rate / symbol_rate)
        dh = [dh_b[center - symbol_size * 5],
              dh_b[center - symbol_size * 4],
              dh_b[center - symbol_size * 3],
              dh_b[center - symbol_size * 2],
              dh_b[center - symbol_size * 1],
              dh_b[center                  ],
              dh_b[center + symbol_size * 1],
              dh_b[center + symbol_size * 2],
              dh_b[center + symbol_size * 3],
              dh_b[center + symbol_size * 4],
              dh_b[center + symbol_size * 5]]
        return dh

    def set_base(self, target):
        self.base = target
        self.target = target
        self.y_intp.set_target(target)
        self.dy_intp.set_target(target)
        return

    def step(self, y_smpl, dy_smpl):
        done, y = self.y_intp.step(y_smpl)
        done, dy = self.dy_intp.step(dy_smpl)
        if done:
            c = qpsk_encode(qpsk_decode(y)) * 0.70710678118654752440084436210485
            #print(f"{c}, {y}")
            self.c.append(c)
            self.y.append(y)
            self.te.append(TimingError(self.dh, self.c, dy, self.D, len(self.y)))

            err = float(0)
            for i in range(len(self.te)):
                self.te[i].update(False)
                if self.te[i].done:
                    err = self.te[i].err
                    self.te.pop(0)
                    break

            err = self.filter.step(err)
            if self.show_err:
                print(f"relative-err [{len(self.y)}] {err*symbol_rate}")
            timing = self.target
            self.target = self.base + float(len(self.y)) / symbol_rate + err
            #print(f"{len(self.y)}: {err}")
            self.y_intp.set_target(self.target)
            self.dy_intp.set_target(self.target)
            return True, timing, y, err, c
        else:
            return False, 0, 0, 0, None

    def update_gain(self, gain):
        self.filter.update_coeff([gain])
        return