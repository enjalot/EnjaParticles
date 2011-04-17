import math
from vector import Vec

class Domain(object):
    def __init__(self, bnd_min, bnd_max):#, surface):

        #the boundary we want particles to stay within
        self.bnd_min = bnd_min
        self.bnd_max = bnd_max

        #get the volume of the domain
        self.width = bnd_max.x - bnd_min.x
        self.height = bnd_max.y - bnd_min.y
        self.depth = bnd_max.z - bnd_min.z
        if self.depth == 0.: self.depth = 1.

        self.V = self.width * self.height * self.depth

        #self.surface = surface


    def setup(self, cell_size):
        #we create 2 cells of padding around the bounds
        s2 = 2.*cell_size;
        self.min = self.bnd_min - Vec([s2, s2, s2])
        self.max = self.bnd_max + Vec([s2, s2, s2])

        self.size = self.max - self.min
        self.res = Vec([    math.ceil(self.size.x / cell_size),
                            math.ceil(self.size.y / cell_size),
                            math.ceil(self.size.z / cell_size) ])

        self.size = self.res * cell_size
        self.max = self.min + self.size

        self.delta = Vec([  self.res.x / self.size.x,
                            self.res.y / self.size.y,
                            self.res.z / self.size.z ])

        self.nb_cells = int(self.res.x * self.res.y * self.res.z)

       
    def calc_cell(self, v):
        """Calculate the grid cell from a vertex"""
        vv  = (v - self.min)*self.delta

        ii = Vec([0,0,0])
        ii.x = int(vv.x)
        ii.y = int(vv.y)
        ii.z = int(vv.z)
        return ii

    def calc_hash(self, p):
        """Calculate the grid hash from a grid position"""
        #tODO: wrapEdges boolean
        gx = p.x
        gy = p.y
        gz = p.z
        return (gz*self.res.y + gy) * self.res.x + gx; 


    def draw(self):
        """draw the lines of the grid"""
        pass

    def __str__(self):
        s = "min: %s\n" % self.min
        s += "max: %s\n" % self.max
        s += "bnd min: %s\n" % self.bnd_min
        s += "bnd max: %s\n" % self.bnd_max
        s += "size: %s\n" % self.size
        s += "res: %s\n" % self.res
        s += "delta: %s\n" % self.delta
        s += "number of cells: %s\n" % self.nb_cells
        return s

if __name__ == "__main__":

    #doing some testing on hashing out of bounds
    dmin = Vec([0,0,0])
    dmax = Vec([5,5,5])
    domain = Domain(dmin, dmax)




