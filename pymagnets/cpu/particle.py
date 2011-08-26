import pygame
from pygame.locals import *
import numpy as np

from vector import Vec
from hash import Domain
from kernels import Kernel


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
        self.mass = sphp.mass
        #defaults
        self.dens = 500
        self.force = Vec([0., 0.])
        self.xsph = Vec([0.,0.])

        self.lock = False

        self.vel = Vec([0.,0.])
        self.veleval = Vec([0.,0.])

        #pygame stuff
        self.col = color
        self.surface = surface
        self.screen_scale = self.surface.get_width() / sphp.domain.width

    def move(self, pos):
        self.pos = pos * self.scale / self.screen_scale

        #print "dens", self.dens

    def draw(self, show_dense = False):
        #draw circle representing particle smoothing radius

        dp = toscreen(self.pos / self.scale, self.surface, self.screen_scale)
        pygame.draw.circle(self.surface, self.col, dp, self.screen_scale * self.h / self.scale, 1)
        #draw filled circle representing density
        #pygame.draw.circle(self.surface, self.col, dp, self.dens / 40., 0)
        if show_dense:
            pygame.draw.circle(self.surface, self.col, dp, self.dens / 10., 0)
        else:
            pygame.draw.circle(self.surface, self.col, dp, 30., 0)

        #TODO draw force vector (make optional)
        #vec = [self.x - f[0]*fdraw/fscale, self.y - f[1]*fdraw/fscale]
        #pygame.draw.line(self.surface, pj.col, self.pos, vec)


