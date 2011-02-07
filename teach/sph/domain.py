

class Domain:
    def __init__(self, dmin, dmax):
        self.dmin = dmin
        self.dmax = dmax
        self.width = dmax.x - dmin.x
        self.height = dmax.y - dmin.y
        self.depth = dmax.z - dmin.z
        if self.depth == 0.: self.depth = 1.

        self.V = self.width * self.height * self.depth
