
__code_to_symb = [complex(-1, -1),
           complex(1, -1),
           complex(1, 1),
           complex(-1, 1)]

def qpsk_encode(data):
    c = int(data) & 0x3
    return __code_to_symb[c]

def qpsk_decode(symb):
    if symb.real <= 0:
        if symb.imag <= 0:
            return 0
        else:
            return 3
    else:
        if symb.imag <= 0:
            return 1
        else:
            return 2

def qpsk_evm(tx_symb, rx_symb, symb_num):
    total_diff = float(0)
    max_diff = float(0)
    avg_diff = float(0)
    err_num = int(0)

    if symb_num != 0:
        for i in range(symb_num):
            if qpsk_decode(tx_symb[i]) != qpsk_decode(rx_symb[i]):
                err_num += 1
            else:
                diff = abs(rx_symb[i] - tx_symb[i])
                max_diff = max(max_diff, diff)
                total_diff += diff

        if symb_num != err_num:
            avg_diff = total_diff / float(symb_num - err_num)

        ref = abs(tx_symb[0])
        return err_num, avg_diff / ref, max_diff / ref
    else:
        return len(tx_symb), 0, 0