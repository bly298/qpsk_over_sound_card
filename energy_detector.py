
class PwrWin:
    def __init__(self, win_sz):
        self.smpl_pwr = [0.01] * win_sz
        self.level = 0.01 * win_sz
        return

    def add_smpl_pwr(self, in_pwr):
        self.level += in_pwr
        self.smpl_pwr.append(in_pwr)
        out_pwr = self.smpl_pwr.pop(0)
        self.level -= out_pwr
        return out_pwr

class EnergyDetector:
    def __init__(self, symb_sz, win_sz):
        self.pwr_win = []
        self.pwr_win.append(PwrWin(win_sz))
        self.pwr_win.append(PwrWin(win_sz))
        self.state = 0
        self.prev_ratio = 0
        self.rising_num = 0
        self.max_ratio = 0
        self.symb_sz = symb_sz
        return

    def step(self, x):
        smpl_pwr = abs(x)**2
        for i in range(len(self.pwr_win)):
            smpl_pwr = self.pwr_win[i].add_smpl_pwr(smpl_pwr)
        ratio = self.pwr_win[0].level / self.pwr_win[1].level
        if self.state == 0: #low
            if ratio > float(10):
                self.max_ratio = ratio
                self.rising_num = 0
                self.state = 1 #crossing
        elif self.state == 1: #crossing
            self.max_ratio = max(ratio, self.max_ratio)
            if ratio >= self.prev_ratio:
                self.rising_num += 1
            else:
                self.rising_num -= 1
            if self.rising_num > self.symb_sz: # found the rising edge
                self.state = 2 #rising
            else:
                if ratio < 5:
                    self.state = 0 #low
        elif self.state == 2: #rising
            self.max_ratio = max(ratio, self.max_ratio)
            if ratio < self.max_ratio * 0.75:
                self.state = 3 #high
        else: #high
            if ratio < 0.3:
                self.state = 0 #low
        self.prev_ratio = ratio
        return ratio, self.state