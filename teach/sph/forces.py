import pygame
from pygame.locals import *

from kernels import *
from util import Vec


class Particle:
    def __init__(self, pos, radius, scale, color, surface):
        #physics stuff
        self.pos = pos
        self.radius = radius
        self.scale = scale
        self.mass = 1

        self.dens = 0

        #pygame stuff
        self.col = color
        self.surface = surface

    def move(self, pos):
        self.pos = pos

        #print "dens", self.dens

    def draw(self):
        #draw circle representing particle smoothing radius
        pygame.draw.circle(self.surface, self.col, self.pos, self.radius*self.scale, 1)
        #draw filled circle representing density
        pygame.draw.circle(self.surface, self.col, self.pos, self.dens*5, 0)


def density_update(particles):
    #brute force
    for pi in particles:
        pi.dens = 0.
        for pj in particles:
            r = dist(pi.pos, pj.pos)
            r = [i/(pi.scale) for i in r]
            #print r
            pi.dens += pj.mass*Wpoly6(pi.radius, r)

def force_update(particles):
    #brute force
    rest_dens = 1000.
    K = 20.
    fscale = 100000 #arbitrary. about how big the force gets in this example
    fdraw = 100     #how big we scale the vector to draw

    for pi in particles:
        pi.force = [0,0]
        di = pi.dens
        Pi = K*(di - rest_dens)
        for pj in particles:
            if pj == pi:
                continue
            r = dist(pi.pos, pj.pos)
            r = [i/(pi.scale) for i in r]

            dj = pj.dens
            Pj = K*(dj - rest_dens)

            kern = -.5 * (Pi + Pj) * dWspiky(pi.radius, r)
            #f = [r[0]*kern, r[1]*kern]
            f = [i*kern for i in r]    #i*kern is physical force
            pi.force[0] += f[0]
            pi.force[1] += f[1]
            """
            vec = [self.x - f[0]*fdraw/fscale, self.y - f[1]*fdraw/fscale]
            pygame.draw.line(self.surface, pj.col, self.pos, vec)
            tot[0] += f[0]*fdraw/fscale
            tot[1] += f[1]*fdraw/fscale
            """

        #tot[0] = self.x - tot[0]
        #tot[1] = self.y - tot[1]
  
def euler_update(particles):
    dt = .001




