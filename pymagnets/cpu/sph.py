import pygame
from pygame.locals import *
import numpy as np

from vector import Vec
from hash import Domain
from kernels import Kernel


class SPH:
    def __init__(self, max_num, domain):
        self.max_num = max_num

        rho0 = 1000.                #rest density [ kg/m^3 ]
        VF = .0262144               #simulation volume [ m^3 ]
        VP = VF / max_num           #particle volume [ m^3 ]
        m = rho0 * VP               #particle mass [ kg ]
        re = (VP)**(1/3.)           #particle radius [ m ]
        #re = (VP)**(1/2.)           #particle radius [ m ]
        print "re, m, VP", re, m, VP
        rest_distance = .87 * re    #rest distance between particles [ m ]

        smoothing_radius = 2.0 * rest_distance      #smoothing radius for SPH Kernels
        boundary_distance = .5 * rest_distance      #for calculating collision with boundary
        
        #the ratio between the particle radius in simulation space and world space
        print "VF", VF
        print "domain.V: ", domain.V
        print "VF/domain.V", VF/domain.V
        print "scale calc", (VF/domain.V)**(1/3.)
        #print "scale calc", (VF/domain.V)**(1/2.)
        sim_scale = (VF / domain.V)**(1/3.)     #[m^3 / world m^3 ]
        #sim_scale = (VF / domain.V)**(1/2.)     #[m^2 / world m^2 ]

        self.rho0 = rho0
        self.VF = VF
        self.mass = m
        self.VP = VP
        self.re = re
        self.rest_distance = rest_distance
        self.smoothing_radius = smoothing_radius
        self.boundary_distance = boundary_distance
        self.sim_scale = sim_scale

        print "=====================================================" 
        print "particle mass:", self.mass
        print "Fluid Volume VF:", self.VF
        print "simulation scale:", self.sim_scale
        print "smoothing radius:", self.smoothing_radius
        print "rest distance:", self.rest_distance
        print "=====================================================" 

        #Other parameters
        self.K = 15.    #Gas constant
        self.boundary_stiffness = 20000.
        self.boundary_dampening = 256.
        #friction
        self.friction_coef = 0.
        self.restitution_coef = 0.
        #not used yet
        self.shear = 0.
        self.attraction = 0.
        self.spring = 0.

        self.velocity_limit = 600.
        self.xsph_factor = .05

        self.viscosity = .01
        self.gravity = -9.8

        self.EPSILON = 10E-6

        #Domain
        self.domain = domain
        self.domain.setup(self.smoothing_radius / self.sim_scale)

        #Kernels
        self.kernels = Kernel(self.smoothing_radius)

    def make_struct(self, num):
        import struct
        sphstruct = struct.pack('ffff'+
                    'ffff'+
                    'ffff'+
                    'ffff'+
                    'ffff'+
                    'ffff'+
                    'ffff'+
                    'iiii',
                    self.mass, self.rest_distance, self.smoothing_radius, self.sim_scale,
                    self.boundary_stiffness, self.boundary_dampening, self.boundary_distance, self.K,
                    self.viscosity, self.velocity_limit, self.xsph_factor, self.gravity,
                    self.friction_coef, self.restitution_coef, self.shear, self.attraction,
                    self.spring, self.EPSILON, np.pi, self.kernels.coeffs["poly6"],
                    0., 0., 0., self.kernels.coeffs["dspiky"],  #some kernels not used so coeffs
                    0., 0., 0., self.kernels.coeffs["ddvisc"],  #not calculated right now
                    num, 0, 0, self.max_num                         #nb_vars and coice not used
                    )
        return sphstruct




if __name__ == "__main__":
    dmin = Vec([0.,0.,0.])
    dmax = Vec([500.,500.,1.])
    domain = Domain(dmin, dmax)
    system = SPH(2**14, domain)     #16384
    #system = SPH(2**12, domain)     #4096

