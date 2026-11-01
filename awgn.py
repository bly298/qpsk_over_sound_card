import numpy as np


def awgn_add(sig, snr):
    snr = 10**(snr/10.0)
    total = 0.0

    for i in np.arange(len(sig)):
        total += sig[i]**2

    s_power = total / len(sig)
    n_power = s_power / snr
    #print(f'sig power {s_power}, noise power {n_power}')
    a = np.sqrt(n_power)
    noise = np.random.randn(len(sig)) * a

    for i in np.arange(len(sig)):
        sig[i] += noise[i]

    return noise, a