import numpy as np
from qpsk_codec import qpsk_evm
from settings import *

class AcPump:
    def __init__(self, depth):
        self.mixed_signal = []
        self.mixed_sum = complex(0, 0)
        self.signal_num = 0
        self.depth = depth
        self.updated = False
        self.overflowed = False
        self.out_signal = None
        return

    def push_mixed_signal(self, signal):
        self.mixed_signal.append(signal)
        self.mixed_sum += signal
        self.signal_num += 1

        if self.updated:
            self.mixed_sum -= self.mixed_signal[0]
            self.out_signal = self.mixed_signal.pop(0)
            self.overflowed = True
            ac = abs(self.mixed_sum)
            return True, ac
        else:
            if self.signal_num >= self.depth:
                self.updated = True
                ac = abs(self.mixed_sum)
                return True, ac
            else:
                return False, 0

class AcWorker:
    def __init__(self, depth, pump_num=1, id=0):
        self.id = id
        self.pump_num = pump_num
        self.pumps = []
        for i in range(pump_num):
            self.pumps.append(AcPump(depth))
        return

    def add_mixed_signal(self, signal, ref_pwr):
        any_updated = False
        total_ac = float(0)
        in_signal = signal

        for i in range(self.pump_num):
            updated, ac = self.pumps[i].push_mixed_signal(in_signal)
            any_updated |= updated
            total_ac += ac
            if self.pumps[i].overflowed:
                in_signal = self.pumps[i].out_signal
            else:
                break

        normalized_ac = total_ac / ref_pwr
        return any_updated, normalized_ac

class BarkerCodeStart:
    def __init__(self, idx, ac, avg_diff, fo, cp, a, ref_pwr):
        self.idx = idx
        self.ac = ac
        self.avg_diff = avg_diff
        self.fo = fo
        self.cp = cp
        self.a = a
        self.pwr = ref_pwr
        return
    def show(self, task_id):
        print(f"Task[{task_id}]: Evaluate: idx = {self.idx}, ac = {self.ac}, avg_diff = {self.avg_diff}, fo = {self.fo}, cp = {self.cp}, a = {self.a}")
        return

class PreambleLocator:
    def __init__(self, L, T, c, bc_len):
        self.L = L
        self.T = T
        self.c = c
        self.bc_len = bc_len
        self.symb_sz = int(sample_rate / symbol_rate)

        self.smpl_buf = []
        self.lead_smpl_num = (self.L - 1) * self.symb_sz
        self.smpl_buf_sz = self.lead_smpl_num + 1
        self.smpl_num = int(0)
        self.delay = bc_len * self.symb_sz

        self.smpl_pwr = []
        self.pwr_ma_win = self.symb_sz * bc_len
        self.energy = float(0)

        depth = bc_len
        pump_num = int(self.L / depth) - 1
        self.worker_idx = 0
        self.workers = []
        for i in range(self.symb_sz):
            self.workers.append(AcWorker(depth, pump_num, i))

        self.bcs_win_sz = 5
        self.bcs = [None] * self.bcs_win_sz
        self.dummy_bcs_num = self.bcs_win_sz
        self.subject = None
        return

    def update_ref_pwr(self, x):
        smpl_pwr = abs(x) ** 2
        self.energy += smpl_pwr
        self.smpl_pwr.append(smpl_pwr)
        if len(self.smpl_pwr) > self.pwr_ma_win:
            self.energy -= self.smpl_pwr[0]
            self.smpl_pwr.pop(0)
        avg_pwr = self.energy / self.pwr_ma_win
        return avg_pwr

    def fitz_fo_estimate(self, z):
        total = float(0)
        for k in range(0, self.L):
            R = float(0)
            for i in range(k, self.L):
                R += z[i] * z[i - k].conjugate()
            total += k * np.angle(R)
        fo = 3.0 * total / (self.L * (self.L - 1) * (2 * self.L - 1) * self.T * np.pi)
        return fo

    def kay_fo_estimate(self, z):
        total = float(0)
        for k in range(1, self.L):
            gama = 1.5 * self.L / (self.L ** 2 - 1) * (1.0 - (2.0 * k / self.L - 1.0) ** 2)
            total += gama * np.angle(z[k] * z[k - 1].conjugate())
        fo = total / (2.0 * np.pi * self.T)
        return fo

    def phase_estimate(self, z):
        total = complex(0)
        for k in range(0, self.L):
            total += z[k]
        theta = np.angle(total)
        return theta

    def amplitude_estimate(self, z):
        total = complex(0)
        for k in range(0, self.L):
            total += z[k]
        a = abs(total) / float(self.L)
        return a

    def verify_preamble(self, y):
        # derive z[k]
        z = [0] * self.L
        for k in range(self.L):
            z[k] = y[k] * self.c[k].conjugate()

        # estimate fo
        fo = self.kay_fo_estimate(z)
        for i in range(len(z)):
            phase = 2.0 * np.pi * fo * i / symbol_rate
            z[i] = z[i] * np.exp(-1j * phase)

        # estimate phase
        cp = self.phase_estimate(z)
        for i in range(len(z)):
            z[i] = z[i] * np.exp(-1j * cp)

        # estimate amplitude
        a = self.amplitude_estimate(z)

        # extract rx_symb
        rx_symb = []
        for i in range(self.L):
            phase = 2.0 * np.pi * fo * i / symbol_rate + cp
            y[i] = y[i] * np.exp(-1j * phase)
            rx_symb.append(y[i] / a)

        err_num, avg_diff, max_diff = qpsk_evm(self.c, rx_symb, self.L)
        return err_num, avg_diff, fo, cp, a

    def evaluate_bcs(self):
        j = int((self.bcs_win_sz - 1) / 2)
        subject = self.bcs[j]
        for i in range(self.bcs_win_sz):
            if i != j:
                if self.bcs[i].avg_diff < subject.avg_diff:
                    subject = None
                    break
        return subject

    def step(self, x):
        ref_pwr = self.update_ref_pwr(x)
        updated = False
        ac = 0

        if len(self.smpl_buf) >= self.delay:
            prev_x = self.smpl_buf[-self.delay]
            mixed_sig = x * prev_x.conjugate()
            worker = self.workers[self.worker_idx]
            updated, ac = worker.add_mixed_signal(mixed_sig, ref_pwr)
            self.worker_idx = (self.worker_idx + 1) % len(self.workers)

        self.smpl_buf.append(x)
        if len(self.smpl_buf) > self.smpl_buf_sz:
            self.smpl_buf.pop(0)
        smpl_idx = self.smpl_num
        self.smpl_num = (self.smpl_num + 1) & 0xFFFF

        bcs = None
        idx_diff = 0
        if updated and (ac > self.bc_len - 3) and (len(self.smpl_buf) >= self.lead_smpl_num + 1):
            y = [0] * self.L
            base = -1 - self.lead_smpl_num
            for k in range(self.L):
                idx = base + k * self.symb_sz
                y[k] = self.smpl_buf[idx]

            err_num, avg_diff, fo, cp, a = self.verify_preamble(y)
            if err_num == 0:
                start = (smpl_idx - self.lead_smpl_num) & 0xFFFF
                bcs = BarkerCodeStart(start, ac, avg_diff, fo, cp, a, ref_pwr)

        self.bcs.append(bcs)
        if bcs is None:
            self.dummy_bcs_num += 1

        if len(self.bcs) > self.bcs_win_sz:
            bcs = self.bcs.pop(0)
            if bcs is None:
                self.dummy_bcs_num -= 1

        if self.dummy_bcs_num == 0 and self.subject is None:
            self.subject = self.evaluate_bcs()
            if self.subject is not None:
                idx_diff = (smpl_idx - self.subject.idx) & 0xFFFF

        return ac, self.subject, idx_diff