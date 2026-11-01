import numpy as np
import random as random
from awgn import awgn_add
from qpsk_codec import qpsk_encode
from qpsk_modulator import Modulator
from settings import *


def tx_sample_generate(a, delta_freq, delta_phase, SNR,
                       preamble, pre_data_num, post_data_num):
    symbol_size = int(sample_rate / symbol_rate)
    symbol_span = int(N / symbol_size)
    modulator = Modulator(freq_offset=delta_freq, init_phase=delta_phase)

    symbol_count = pre_data_num + len(preamble) + post_data_num
    total_count = symbol_count + symbol_span * 2

    sqrt_2 = np.sqrt(2)
    tx_symb = [complex(0, 0j)] * total_count

    for i in range(symbol_count):
        x = random.randint(0, 3)
        tx_symb[i] = qpsk_encode(x) / sqrt_2

    for i in range(len(preamble)):
        tx_symb[pre_data_num + i] = complex(preamble[i], preamble[i]) / sqrt_2

    signal = []
    for i in range(total_count):
        sig, smpl = modulator.step(tx_symb[i] * a)
        signal.extend(sig)

    # add noise
    if SNR >= 0:
        awgn_add(signal, SNR)

    tx_symb = tx_symb[0:symbol_count]
    return signal, tx_symb