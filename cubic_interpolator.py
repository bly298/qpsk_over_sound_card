
class CubicInterpolator:
    def __init__(self, sample_rate, depth):
        self.sample_rate = int(sample_rate)
        self.depth = depth
        self.now = int(0)
        self.alarm = int(0)
        self.reset()
        return

    def reset(self):
        self.now = int(0)
        self.alarm = int(0)
        self.x = [float(0)] * self.depth
        self.t = float(0)
        self.m = int(0)
        self.u = float(0)
        return

    def interpolate(self, m, u):
        x = 0.0
        x += self.x[m + 2] * (1.0/6 * u**3 - 1.0/6 * u)
        x += self.x[m + 1] * (-1.0/2 * u**3 + 1.0/2 * u**2 + u)
        x += self.x[m + 0] * (1.0/2 * u**3 - u**2 - 1.0/2 * u + 1)
        x += self.x[m - 1] * (-1.0/6 * u**3 + 1.0/2 * u**2 - 1.0/3 * u)
        return x

    def step(self, x):
        if self.now >= len(self.x):
            self.x.pop(0)
            self.x.append(x)
            base = self.now - (len(self.x) - 1)
        else:
            self.x[self.now] = x
            base = 0

        if self.now == self.alarm:
            m = self.m - base
            assert (0 <= m - 1) and (m + 2 < len(self.x))
            revised_x = self.interpolate(m, self.u)
            self.now += 1
            return True, revised_x
        else:
            self.now += 1
            return False, 0

    def set_target(self, t):
        self.t = t
        self.m = int(t * self.sample_rate)
        self.u = t * self.sample_rate - self.m
        self.alarm = self.m + 2
        return