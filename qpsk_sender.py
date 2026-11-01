import random
import struct
import numpy as np
import pyaudio
from awgn import awgn_add
from qpsk_codec import qpsk_encode
from qpsk_modulator import Modulator
from wav_file import wav_file_write
from settings import *
import zlib


def save_input(filename, delta_freq, delta_phase, a, tx_symb, signal):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"[delta_freq]\n")
        f.write(f"{delta_freq}\n")
        f.write(f"[delta_phase]\n")
        f.write(f"{delta_phase}\n")
        f.write(f"[a]\n")
        f.write(f"{a}\n")
        f.write(f"[tx_symb]\n")
        for z in tx_symb:
            f.write(f"{z}\n")
        f.write(f"[signal]\n")
        for s in signal:
            f.write(f"{s}\n")
    return

class QpskSender:
    def __init__(self):
        self.preamble = [ 1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1,
                    -1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1]
        return

    def encode_uint8(self, uint8, tx_symb, symb_idx):
        x = uint8 & 0x3
        tx_symb[symb_idx] = qpsk_encode(x) * 0.70710678118654752440084436210485
        x = (uint8 >> 2) & 0x3
        tx_symb[symb_idx + 1] = qpsk_encode(x) * 0.70710678118654752440084436210485
        x = (uint8 >> 4) & 0x3
        tx_symb[symb_idx + 2] = qpsk_encode(x) * 0.70710678118654752440084436210485
        x = (uint8 >> 6) & 0x3
        tx_symb[symb_idx + 3] = qpsk_encode(x) * 0.70710678118654752440084436210485
        return symb_idx + 4

    def encode_uint32(self, uint32, tx_symb, symb_idx):
        self.encode_uint8(uint32 & 0xFF, tx_symb, symb_idx)
        self.encode_uint8((uint32 >> 8) & 0xFF, tx_symb, symb_idx + 4)
        self.encode_uint8((uint32 >> 16) & 0xFF, tx_symb, symb_idx + 8)
        self.encode_uint8((uint32 >> 24) & 0xFF, tx_symb, symb_idx + 12)
        return symb_idx + 16

    def send(self, sn, type, data):
        delta_freq = random.randint(0, 10)
        delta_phase = np.pi * random.random()
        a = 5 #float(random.randint(1, 10))
        SNR = -1
        symbol_size = int(sample_rate / symbol_rate)
        symbol_span = int(N / symbol_size)
        modulator = Modulator(freq_offset=delta_freq, init_phase=delta_phase)

        content = bytearray(data, 'utf-8')
        length = len(content)
        assert length < 0x10000
        header = bytearray(6)
        header[0] = sn & 0xFF
        header[1] = (sn >> 8) & 0xFF
        header[2] = type & 0xFF
        header[3] = (type >> 8) & 0xFF
        header[4] = length & 0xFF
        header[5] = (length >> 8) & 0xFF
        hdr_cs = zlib.crc32(header)
        frm_cs = zlib.crc32(content)

        # header: 6 bytes
        # header checksum: 4 bytes
        # content: n bytes
        # frame checksum: 4 bytes
        # frame extension: 16 bytes
        # total 30 + n bytes
        symb_num = len(self.preamble) + (30 + len(content)) * 4
        total_num = symb_num + symbol_span * 2
        tx_symb = [complex(0, 0j)] * total_num

        # preamble
        for i in range(len(self.preamble)):
            tx_symb[i] = complex(self.preamble[i], self.preamble[i]) * 0.70710678118654752440084436210485

        symb_idx = len(self.preamble)

        # header
        for i in range(len(header)):
            symb_idx = self.encode_uint8(header[i], tx_symb, symb_idx)

        # header checksum
        symb_idx = self.encode_uint32(hdr_cs, tx_symb, symb_idx)

        # content
        for i in range(len(content)):
            symb_idx = self.encode_uint8(content[i], tx_symb, symb_idx)

        # frame checksum
        symb_idx = self.encode_uint32(frm_cs, tx_symb, symb_idx)

        # frame extension
        symb_idx = self.encode_uint32(frm_cs, tx_symb, symb_idx)
        symb_idx = self.encode_uint32(frm_cs, tx_symb, symb_idx)
        symb_idx = self.encode_uint32(frm_cs, tx_symb, symb_idx)
        symb_idx = self.encode_uint32(frm_cs, tx_symb, symb_idx)

        signal = []
        for i in range(total_num):
            sig, smpl = modulator.step(tx_symb[i] * a)
            signal.extend(sig)

        # mix signal with noise
        if SNR >= 0:
            awgn_add(signal, SNR)

        # dump signal to a wav file
        wav_file_write('dump\\raw_signal.wav', sample_rate, signal)

        # dump all
        tx_symb = tx_symb[0:symb_num]
        save_input("dump\\raw_signal.txt", delta_freq, delta_phase, a, tx_symb, signal)

        # send signal over sound card
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, output=True)
        for i in range(0, len(signal)) :
            val = int(signal[i] * 500)
            output = struct.pack('hh', val, val)
            stream.write(output)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return signal

def test_qpsk_sender():
    sender = QpskSender()
    content = 'This paper is devoted to the comparison of hard and soft decision directed feedforward phase estimators based on the maximum likelihood principle. The particular structure of these estimators is taken into account to derive a new lower bound on the estimation variance for a constant phase error. Equivalent noise bandwidths for decision directed estimators are also studied. They allow us to characterize the estima-tor behaviour for the tracking of a time varying phase error. Simulation results are finally proposed, comparing the root mean square errors of the different estimators, for a constant phase error and for a time varying phase error respectively.'
    #content = 'Hello, this is Ted from Saibotan'
    sender.send(0x55,0, content)
    return

if __name__ == '__main__':
    test_qpsk_sender()