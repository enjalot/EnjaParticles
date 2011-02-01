import os, sys
import pygame
from pygame.locals import *

from kernels import *
class Particle:
    def __init__(self, x, y, radius, scale, color, surface):
        #physics stuff
        self.x = x
        self.y = y
        self.pos = [self.x, self.y]
        self.radius = radius
        self.scale = scale

        #pygame stuff
        self.col = color
        self.surface = surface

    def move(self, x,y):
        self.x = x
        self.y = y
        self.pos = [self.x, self.y]

    def density(self, particles):
        self.dens = 0

        mj = 1 #mass = 1 for now
        for pj in particles:
            r = dist(self.pos, pj.pos)
            r = [i/(self.scale) for i in r]
            #print r
            self.dens += mj*Wpoly6(self.radius, r)

    def force(self, particles):
        rest_dens = 1000.
        K = 20.
        fscale = 100000 #arbitrary. about how big the force gets in this example
        fdraw = 100     #how big we scale the vector to draw
        tot = [0,0] #total force vector
        for pj in particles:
            if pj == self:
                continue
            r = dist(self.pos, pj.pos)
            r = [i/(self.scale) for i in r]

            di = self.dens
            dj = pj.dens
            Pi = K*(di - rest_dens)
            Pj = K*(dj - rest_dens)

            kern = -.5 * (Pi + Pj) * dWspiky(self.radius, r)
            #f = [r[0]*kern, r[1]*kern]
            f = [i*kern for i in r]    #i*kern is physical force
            vec = [self.x - f[0]*fdraw/fscale, self.y - f[1]*fdraw/fscale]
            pygame.draw.line(self.surface, pj.col, self.pos, vec)
            tot[0] += f[0]*fdraw/fscale
            tot[1] += f[1]*fdraw/fscale

        tot[0] = self.x - tot[0]
        tot[1] = self.y - tot[1]
        pygame.draw.line(self.surface, self.col, self.pos, tot)





        #print "dens", self.dens

    def draw(self):
        #draw circle representing particle smoothing radius
        pygame.draw.circle(self.surface, self.col, self.pos, self.radius*self.scale, 1)
        #draw filled circle representing density
        pygame.draw.circle(self.surface, self.col, self.pos, self.dens*5, 0)


def init_particles(surface):
    particles = []
    radius = 1.
    scale = 80.
    particles += [ Particle(100,100, radius, scale, [255,0,0], surface) ] 
    particles += [ Particle(400,400, radius, scale, [0,0,255], surface) ] 
    particles += [ Particle(479,400, radius, scale, [0,205,0], surface) ] 
    particles += [ Particle(400,479, radius, scale, [0,205,205], surface) ] 
    return particles

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption('SPH Forces')

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    clock = pygame.time.Clock()

    particles = init_particles(screen)


    mouse_down = False
    while 1:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and (event.key == K_ESCAPE or event.key == 113): #q
                return
            elif event.type == MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == MOUSEMOTION:
                if(mouse_down):
                    particles[0].move(event.pos[0], event.pos[1])
            elif event.type == MOUSEBUTTONUP:
                mouse_down = False

        
        screen.blit(background, (0, 0))
        for p in particles:
            p.density(particles)
        particles[0].force(particles);
        for p in particles:
            p.draw()
        pygame.display.flip()

if __name__ == "__main__":
    main()
