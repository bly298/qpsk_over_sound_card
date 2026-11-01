import zlib
from qpsk_codec import qpsk_decode

class QpskDecoder:
    def __init__(self, c):
        self.byte_val = 0
        self.bit_offset = 0
        self.state = 0
        self.symb_num = 0
        self.byte_num = 0
        self.header = bytearray()
        self.hdr_cs_received = 0
        self.hdr_cs_rebuilt = 0
        self.cs_offset = 0
        self.length = 0
        self.data = bytearray()
        self.frm_cs_received = 0
        self.frm_cs_rebuilt = 0
        self.rx_err = -99
        self.c = c
        return

    def step(self, y):
        done = False
        self.symb_num += 1
        bit_val = qpsk_decode(y)
        if self.state == 0:
            if qpsk_decode(self.c[self.symb_num - 1]) == bit_val:
                if self.symb_num == len(self.c):
                    print("preamble found")
                    self.state = 1
            else:
                self. rx_err = -1
                done = True
        else:
            self.byte_val |= ((bit_val & 0x3) << self.bit_offset)
            self.bit_offset += 2
            if self.bit_offset == 8:
                #print(hex(self.byte_val))
                self.byte_num += 1
                if self.state == 1:
                    if self.byte_num <= 2:
                        self.header.append(self.byte_val)
                        if self.byte_num == 2:
                            self.sn = int(self.header[0]) + (int(self.header[1]) << 8)
                            print(f"sn = {self.sn}")
                    elif self.byte_num <= 4:
                        self.header.append(self.byte_val)
                        if self.byte_num == 4:
                            self.type = int(self.header[2]) + (int(self.header[3]) << 8)
                            print(f"type = {self.type}")
                    elif self.byte_num <= 6:
                        self.header.append(self.byte_val)
                        if self.byte_num == 6:
                            self.length = int(self.header[4]) + (int(self.header[5]) << 8)
                            print(f"length = {self.length}")
                            self.hdr_cs_rebuilt = zlib.crc32(self.header)
                            print(f"hdr_cs_rebuilt = {self.hdr_cs_rebuilt:08X}")
                    elif self.byte_num <= 10:
                        self.hdr_cs_received |= (self.byte_val << self.cs_offset)
                        self.cs_offset += 8
                        if self.byte_num == 10:
                            print(f"hdr_cs_received = {self.hdr_cs_received:08X}")
                            if self.hdr_cs_received != self.hdr_cs_rebuilt:
                                self.rx_err = -2
                                done = True
                    elif self.byte_num <= 10 + self.length:
                        self.data.append(self.byte_val)
                        if self.byte_num == 10 + self.length:
                            self.frm_cs_rebuilt = zlib.crc32(self.data)
                            print(f"frm_cs_rebuilt = {self.frm_cs_rebuilt:08X}")
                            self.cs_offset = 0
                    elif self.byte_num <= 14 + self.length:
                        self.frm_cs_received |= (self.byte_val << self.cs_offset)
                        self.cs_offset += 8
                        if self.byte_num == 14 + self.length:
                            print(f"frm_cs_received = {self.frm_cs_received:08X}")
                            if self.frm_cs_received != self.frm_cs_rebuilt:
                                self.rx_err = -3
                            else:
                                self.rx_err = 0
                            done = True
                self.byte_val = 0
                self.bit_offset = 0
        return done