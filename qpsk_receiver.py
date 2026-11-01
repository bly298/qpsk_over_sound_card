import copy
import time
import numba
import numpy as np
import pyaudio
from constellation import refined_constellation_show
from energy_detector import EnergyDetector
from ml_phase_synchronizer import PhaseTracker
from ml_symbol_synchronizer import TimingTracker
from preamble_locator import PreambleLocator
from qpsk_symbol_decoder import QpskDecoder
from raised_cosine_filter import SqrtRaisedCosineFilter, DsrrcFilter
from wav_file import wav_file_read, wav_file_write
from settings import *
from qpsk_symbol_receiver import SymbReceiver
import matplotlib.pyplot as plt


def load_input(filename):
    delta_freq = 0
    delta_phase = 0
    a = 1
    tx_symb = []
    signal = []
    state = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            while True:
                if state is None or state == "retry":
                    if s == "[delta_freq]":
                        state = "delta_freq"
                    elif s == "[delta_phase]":
                        state = "delta_phase"
                    elif s == "[a]":
                        state = "a"
                    elif s == "[tx_symb]":
                        state = "tx_symb"
                    elif s == "[signal]":
                        state = "signal"
                elif state == "delta_freq":
                    delta_freq = float(s)
                    state = None
                elif state == "delta_phase":
                    delta_phase = float(s)
                    state = None
                elif state == "a":
                    a = float(s)
                    state = None
                elif state == "tx_symb":
                    if s[0] != '[':
                        tx_symb.append(complex(s))
                    else:
                        state = "retry"
                elif state == "signal":
                    if s[0] != '[':
                        signal.append(float(s))
                    else:
                        state = "retry"
                if state != "retry":
                    break
    return delta_freq, delta_phase, a, tx_symb, signal

@numba.jit(nopython=True)
def sum_products(a, b):
    sum = complex(0, 0)
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

@numba.jit(nopython=True)
def filter_mixed_sig(coeff, prev_sig, cur_sig, output):
    idx = 0
    prev_sz = len(coeff) - 1
    while prev_sz > 0:
        output[idx] = sum_products(coeff[:prev_sz], prev_sig[-prev_sz:])
        next_sz = len(coeff) - prev_sz
        output[idx] += sum_products(coeff[-next_sz:], cur_sig[:next_sz])
        prev_sz -= 1
        idx += 1
    start = 0
    while start + len(coeff) <= len(cur_sig):
        output[idx] = sum_products(coeff, cur_sig[start:start + len(coeff)])
        start += 1
        idx += 1
    return

class QpskReceiver:
    def __init__(self, L, T, c, bc_len, process_frame):
        self.L = L
        self.T = T
        self.c = c
        self.bc_len = bc_len
        assert len(c) == L
        self.symb_sz = int(sample_rate / symbol_rate)
        self.detector = EnergyDetector(self.symb_sz, self.symb_sz*self.bc_len)
        self.found = False
        self.bcs = None
        self.buf_sz = self.L * 2 * self.symb_sz
        self.y_buf = np.zeros(self.buf_sz, dtype=complex)
        self.dy_buf = np.zeros(self.buf_sz, dtype=complex)
        self.smpl_idx = 0
        self.smpl_num = 0
        self.smpl_num_up_to_bcs = 0
        self.buf_overflowed = False
        self.phase = 0
        self.reset()

        self.interval = self.symb_sz * 1000
        self.weight = np.zeros(self.interval, dtype=complex)
        for i in range(len(self.weight)):
            phase = 2 * np.pi * carrier_rate * i / sample_rate
            self.weight[i] = 2.0 * np.exp(-1j * phase)

        mf = SqrtRaisedCosineFilter(sample_rate, symbol_rate, N, alpha)
        b, a = mf.get_coeff()
        self.mf_b = np.array(b)[::-1]
        dmf = DsrrcFilter(sample_rate, symbol_rate, N, alpha)
        b, a = dmf.get_coeff()
        self.dmf_b = np.array(b)[::-1]
        self.prev_sig = np.zeros(self.interval, dtype=complex)

        self.prev_det_st = 0
        self.process_frame = process_frame
        return

    def reset(self):
        self.found = False
        self.bcs = None
        self.locator = PreambleLocator(self.L, self.T, self.c, self.bc_len)
        self.phase_tracker = PhaseTracker(gamma=0.05, rho=0.01)
        self.timing_tracker = TimingTracker(0.00000003, D=2, show_err=False)
        self.receiver = SymbReceiver(self.symb_sz, self.phase_tracker, self.timing_tracker)
        self.decoder = QpskDecoder(self.c)
        return

    def rx(self, signal):
        start = 0
        while (start < len(signal)):
            size = min(self.interval, len(signal) - start)
            if size < self.interval:
                rx_sig = np.zeros(self.interval, dtype=complex)
                rx_sig[:size] = np.array(signal[start:], dtype=complex)
                size = self.interval
            else:
                rx_sig = np.array(signal[start:start+size], dtype=complex)

            mixed_sig = rx_sig * self.weight[:size]
            y_smpl = np.zeros(len(mixed_sig), dtype=complex)
            filter_mixed_sig(self.mf_b, self.prev_sig, mixed_sig, y_smpl)
            dy_smpl = np.zeros(len(mixed_sig), dtype=complex)
            filter_mixed_sig(self.dmf_b, self.prev_sig, mixed_sig, dy_smpl)
            self.prev_sig = mixed_sig

            for i in range(len(mixed_sig)):
                self.step(y_smpl[i], dy_smpl[i])

            start += size
        return

    def step(self, y, dy):
        decoder_done = False
        self.detector.step(y)
        if self.prev_det_st != self.detector.state:
            print(f"detector {self.prev_det_st}->{self.detector.state}")
            self.prev_det_st = self.detector.state

        self.y_buf[self.smpl_idx] = y
        self.dy_buf[self.smpl_idx] = dy
        cur_idx = self.smpl_idx
        self.smpl_idx += 1
        if self.smpl_idx >= self.buf_sz:
            self.smpl_idx = 0

        if not self.buf_overflowed:
            self.smpl_num += 1
            if self.smpl_num == self.buf_sz:
                self.buf_overflowed = True
        else:
            if (not self.found) and (self.bcs is not None):
                self.smpl_num_up_to_bcs -= 1
                if self.smpl_num_up_to_bcs < self.symb_sz:
                    self.reset()

        if not self.found:
            if self.bcs is None:
                ac, bcs, idx_diff = self.locator.step(y)
                if bcs is not None:
                    self.smpl_num_up_to_bcs = self.smpl_num - idx_diff
                    assert self.smpl_num_up_to_bcs > 0
                    self.bcs = bcs
                    bcs.show(0)

            if self.bcs is not None:
                if self.detector.state == 3:
                    self.found = True
                    self.receiver.set_base(self.smpl_num_up_to_bcs - 1)
                    idx = (cur_idx - (self.smpl_num - 1) + self.buf_sz) % self.buf_sz
                    for n in range(self.smpl_num):
                        diff = n + 1 - self.smpl_num_up_to_bcs
                        self.phase = 2.0 * np.pi * self.bcs.fo * diff / sample_rate + self.bcs.cp
                        self.y_buf[idx] *= np.exp(-1j * self.phase) / self.bcs.a
                        self.dy_buf[idx] *= np.exp(-1j * self.phase) / self.bcs.a
                        done, y = self.receiver.step(self.y_buf[idx], self.dy_buf[idx])
                        if done and self.decoder.step(y):
                            decoder_done = True
                            break
                        idx = (idx + 1 + self.buf_sz) % self.buf_sz
                elif self.detector.state == 0:
                    print("bcs is found but det-st goes down")
                    self.reset()
        else:
            if self.detector.state == 0:
                print("preamble is found but det-st goes down")
                self.reset()

            self.phase += 2.0 * np.pi * self.bcs.fo / sample_rate
            while self.phase > 2.0 * np.pi:
                self.phase -= 2.0 * np.pi
            y *= np.exp(-1j * self.phase) / self.bcs.a
            dy *= np.exp(-1j * self.phase) / self.bcs.a
            done, y = self.receiver.step(y, dy)
            if done and self.decoder.step(y):
                decoder_done = True

        if decoder_done:
            if self.decoder.rx_err == 0:
                rx_symb = copy.deepcopy(self.receiver.rx_symb)
                data = copy.deepcopy(self.decoder.data)
                print("decoding is done")
                self.reset()
                self.process_frame(data, rx_symb)
            else:
                print(f"rx-err {self.decoder.rx_err}")
                self.reset()
        return

def dump_rx_process_frame(data, rx_symb):
    content = data.decode('utf-8')
    print(f"content = {content}")
    refined_constellation_show(rx_symb, len(rx_symb), "IQ with timing-sync")
    return

def test_dump_rx():
    preamble = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
                -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]

    '''
    path = "dump\\raw_signal.txt"
    delta_freq, delta_phase, a, tx_symb, signal = load_input(path)
    print(f"load {path}: delta_freq = {delta_freq}, delta_phase = {delta_phase}, a = {a}")
    L = len(preamble)
    T = float(1) / symbol_rate
    c = tx_symb[0:L]
    '''
    raw_data = wav_file_read('dump\\recording.wav')
    signal = raw_data[:,0]
    n = np.arange(0, len(signal))
    plt.plot(n, signal)
    plt.title('Signal')
    plt.grid()
    plt.show()
    L = len(preamble)
    T = float(1) / symbol_rate
    c = []
    for i in range(len(preamble)):
        symb = complex(preamble[i], preamble[i]) * 0.70710678118654752440084436210485
        c.append(symb)

    receiver = QpskReceiver(L, T, c, 13, dump_rx_process_frame)
    start = time.time()
    receiver.rx(signal)
    stop = time.time()
    print("cost %s" % (stop - start))
    return

def audio_rx_process_frame(data, rx_symb):
    content = data.decode('utf-8')
    print(f"content = {content}")
    return

def test_audio_rx():
    preamble = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
                -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]

    L = len(preamble)
    T = float(1) / symbol_rate
    c = []
    for i in range(len(preamble)):
        symb = complex(preamble[i], preamble[i]) * 0.70710678118654752440084436210485
        c.append(symb)

    receiver = QpskReceiver(L, T, c, 13, audio_rx_process_frame)

    CHUNK = receiver.interval
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RECORDING = True
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)
    print('receiving')
    while RECORDING:
        try:
            str_data = stream.read(CHUNK)
            raw_data = np.frombuffer(str_data, dtype=np.int16)
            start = time.time()
            receiver.rx(raw_data)
            stop = time.time()
            print("cost %s" % (stop - start))
        except Exception as e:
            print(e)
            RECORDING = False
    print('end')
    stream.stop_stream()
    stream.close()
    p.terminate()
    return

if __name__ == '__main__':
    #test_dump_rx()
    test_audio_rx()