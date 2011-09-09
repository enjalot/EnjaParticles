import pygame
from pygame.locals import *
import numpy as np

from vector import Vec
from hash import Domain
from kernels import Kernel



import numpy as np

def addRect_old(num, pmin, pmax, sphp):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addRect ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    spacing = 1.0 * sphp.rest_distance / sphp.sim_scale;
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
            #rvec += [[x, y, 0., 1.]]
            i+=1;
    print "%d particles added" % i
    #rvecnp = np.array(rvec, dtype=np.float32)
    #return rvecnp;
    return rvec



def init_particles(num, sphp, domain, surface):
    particles = []
    p1 = Vec([.5, 2.]) * sphp.sim_scale
    particles += [ Particle(p1, sphp, [0,0,255], surface) ] 

    pmin = Vec([.5, .5])
    pmax = Vec([2., 3.])
    ps = addRect_old(num, pmin, pmax, sphp)
    for p in ps:
        particles += [ Particle(p, sphp, [255,0,0], surface) ] 

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

   
