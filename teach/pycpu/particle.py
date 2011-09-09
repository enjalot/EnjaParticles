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
    def __init__(self, pos, np, system, color, surface):
        #physics stuff
        #position
        self.pos = pos
        #north pole (vector with origin at position)
        #self.np = np
        #south pole (vector with origin at position)
        #self.sp = -1 * np

        #self.r = system.radius
        self.h = system.smoothing_radius
        self.scale = system.sim_scale
        self.mass = system.mass
        self.dens = system.rho0

        self.force = Vec([0., 0.])
        self.vel = Vec([0.,0.])
        self.veleval = Vec([0.,0.])

        #lock a particle in place (force updates don't affect it)
        self.lock = False

        #pygame stuff
        self.col = color
        self.surface = surface
        self.screen_scale = self.surface.get_width() / system.domain.width

    def move(self, pos):
        self.pos = pos * self.scale / self.screen_scale

        #print "dens", self.dens

    def draw(self, show_dense = False):
        #draw circle representing particle smoothing radius

        dp = toscreen(self.pos / self.scale, self.surface, self.screen_scale)
        dp = [int(dp.x), int(dp.y)]
        r = self.screen_scale * self.h / self.scale
        #print dp, r
        #pygame.draw.circle(self.surface, self.col, dp, r, 1)
        #pygame.draw.circle(self.surface, self.col, dp, int(r), 1)
        #draw filled circle representing particle size 
        pygame.draw.circle(self.surface, self.col, dp, int(self.dens / 40.), 0)
        
        #dnp = toscreen(self.pos + self.np / self.scale, self.surface, self.screen_scale)
        #dsp = toscreen(self.pos + self.sp / self.scale, self.surface, self.screen_scale)
        #pygame.draw.circle(self.surface, [200,0,0], dnp, 5)
        #pygame.draw.circle(self.surface, [0,0,200], dsp, 5)


        #TODO draw force vector (make optional)
        #vec = [self.x - f[0]*fdraw/fscale, self.y - f[1]*fdraw/fscale]
        #pygame.draw.line(self.surface, pj.col, self.pos, vec)


