import math
from vector import Vec

class Domain(object):
    def __init__(self, bnd_min, bnd_max, surface):

        #the boundary we want particles to stay within
        self.bnd_min = bnd_min
        self.bnd_max = bnd_max

        #get the volume of the domain
        self.width = bnd_max.x - bnd_min.x
        self.height = bnd_max.y - bnd_min.y
        self.depth = bnd_max.z - bnd_min.z
        if self.depth == 0.: self.depth = 1.

        self.V = self.width * self.height * self.depth

        self.surface = surface


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

        self.delta = Vec([  self.res.x / self.size.x,
                            self.res.y / self.size.y,
                            self.res.z / self.size.z ])


       
    def calc_pos(self, v):
        """Calculate the grid position from a vertex"""
        vv  = (v - self.grid_min)*self.grid_delta

        ii = Vec([0,0,0])
        ii.x = int(vv.x)
        ii.y = int(vv.y)
        ii.z = int(vv.z)
        return ii

    def calc_cell(self, p):
        """Calculate the grid cell from a grid position"""
        #tODO: wrapEdges boolean
        gx = p.x
        gy = p.y
        gz = p.z
        return (gz*grid_res.y + gy) * self.grid_res.x + gx; 


    def draw(self):
        """draw the lines of the grid"""
        pass

