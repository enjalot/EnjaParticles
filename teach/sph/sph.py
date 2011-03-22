import pygame
from pygame.locals import *
import numpy as np

from vector import Vec
from hash import Domain
from kernels import Kernel

screen_scale = 160

class SPH:
    def __init__(self, max_num, domain):
        self.max_num = max_num

        rho0 = 1000.                #rest density [ kg/m^3 ]
        VF = .0262144               #simulation volume [ m^3 ]
        VP = VF / max_num           #particle volume [ m^3 ]
        m = rho0 * VP               #particle mass [ kg ]
        re = (VP)**(1/3.)           #particle radius [ m ]
        print "re, m, VP", re, m, VP
        rest_distance = .87 * re    #rest distance between particles [ m ]

        smoothing_radius = 2.0 * rest_distance      #smoothing radius for SPH Kernels
        boundary_distance = .5 * rest_distance      #for calculating collision with boundary
        
        #the ratio between the particle radius in simulation space and world space
        print "VF", VF
        print "domain.V: ", domain.V
        print "VF/domain.V", VF/domain.V
        print "scale calc", (VF/domain.V)**(1/3.)
        sim_scale = (VF / domain.V)**(1/3.)     #[m^3 / world m^3 ]

        self.rho0 = rho0
        self.VF = VF
        self.m = m
        self.VP = VP
        self.re = re
        self.rest_distance = rest_distance
        self.smoothing_radius = smoothing_radius
        self.boundary_distance = boundary_distance
        self.sim_scale = sim_scale

        print "=====================================================" 
        print "particle mass:", self.m
        print "Fluid Volume VF:", self.VF
        print "simulation scale:", self.sim_scale
        print "smoothing radius:", self.smoothing_radius
        print "rest distance:", self.rest_distance
        print "=====================================================" 

        #Other parameters
        self.K = 2.    #Gas constant
        self.boundary_stiffness = 20000.
        self.boundary_dampening = 256.

        self.velocity_limit = 600.
        self.xsph_factor = .05

        self.domain = domain
        self.domain.setup(self.smoothing_radius / self.sim_scale)

        self.kernels = Kernel(self.smoothing_radius)


def toscreen(p, surface, screen_scale):
    translate = Vec([0,0])
    p.x = translate.x + p.x*screen_scale
    p.y = surface.get_height() - (translate.y + p.y*screen_scale)
    return p

class Particle:
    def __init__(self, pos, sphp, color, surface):
        #physics stuff
        self.pos = pos
        self.h = sphp.smoothing_radius
        self.scale = sphp.sim_scale
        self.mass = sphp.m

        self.vel = Vec([0.,0.])
        self.veleval = Vec([0.,0.])

        #pygame stuff
        self.col = color
        self.surface = surface
        self.screen_scale = self.surface.get_width() / sphp.domain.width

    def move(self, pos):
        self.pos = pos * self.scale

        #print "dens", self.dens

    def draw(self):
        #draw circle representing particle smoothing radius

        dp = toscreen(self.pos / self.scale, self.surface, self.screen_scale)
        pygame.draw.circle(self.surface, self.col, dp, self.screen_scale * self.h / self.scale, 1)
        #draw filled circle representing density
        pygame.draw.circle(self.surface, self.col, dp, self.dens / 40., 0)

        #TODO draw force vector (make optional)
        #vec = [self.x - f[0]*fdraw/fscale, self.y - f[1]*fdraw/fscale]
        #pygame.draw.line(self.surface, pj.col, self.pos, vec)

#std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
def addRect(num, pmin, pmax, sphp):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addRect ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    spacing = 1.5 * sphp.rest_distance / sphp.sim_scale;
    print "spacing", spacing

    xmin = pmin.x# * scale
    xmax = pmax.x# * scale
    ymin = pmin.y# * scale
    ymax = pmax.y# * scale

    print "min, max", xmin, xmax, ymin, ymax
    rvec = [];
    i=0;
    for y in np.arange(ymin, ymax, spacing):
        for x in np.arange(xmin, xmax, spacing):
            if i >= num: break
            print "x, y", x, y
            rvec += [ Vec([x,y]) * sphp.sim_scale];
            i+=1;
    print "%d particles added" % i
    return rvec;



def init_particles(num, sphp, domain, surface):
    particles = []
    p1 = Vec([.5, 2.]) * sphp.sim_scale
    particles += [ Particle(p1, sphp, [255,0,0], surface) ] 

    pmin = Vec([.5, .5])
    pmax = Vec([2., 3.])
    ps = addRect(num, pmin, pmax, sphp)
    for p in ps:
        particles += [ Particle(p, sphp, [0,0,255], surface) ] 

    """
    p2 = Vec([400., 400.]) * sphp.sim_scale
    p3 = Vec([415., 400.]) * sphp.sim_scale
    p4 = Vec([400., 415.]) * sphp.sim_scale
    p5 = Vec([415., 415.]) * sphp.sim_scale
    p6 = Vec([430., 430.]) * sphp.sim_scale
    particles += [ Particle(p1, sphp, [255,0,0], surface) ] 
    particles += [ Particle(p2, sphp, [0,0,255], surface) ] 
    particles += [ Particle(p3, sphp, [0,205,0], surface) ] 
    particles += [ Particle(p4, sphp, [0,205,205], surface) ] 
    particles += [ Particle(p5, sphp, [0,205,205], surface) ] 
    particles += [ Particle(p6, sphp, [0,205,205], surface) ] 
    """
    return particles



if __name__ == "__main__":
    dmin = Vec([0.,0.,0.])
    dmax = Vec([500.,500.,1.])
    domain = Domain(dmin, dmax)
    system = SPH(2**14, domain)     #16384
    #system = SPH(2**12, domain)     #4096

