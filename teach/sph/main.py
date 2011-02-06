import pygame
from pygame.locals import *

from forces import *
from vector import Vec

def init_particles(surface):
    particles = []
    radius = 1.
    scale = 80.
    particles += [ Particle(Vec([100,100]), radius, scale, [255,0,0], surface) ] 
    particles += [ Particle(Vec([400,400]), radius, scale, [0,0,255], surface) ] 
    particles += [ Particle(Vec([479,400]), radius, scale, [0,205,0], surface) ] 
    particles += [ Particle(Vec([400,479]), radius, scale, [0,205,205], surface) ] 
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
        key = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or key[K_ESCAPE] or key[K_q]:
                print "quit!"
                return
            elif event.type == MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == MOUSEMOTION:
                if(mouse_down):
                    v = Vec([event.pos[0], event.pos[1]])
                    particles[0].move(v)
            elif event.type == MOUSEBUTTONUP:
                mouse_down = False

        
        screen.blit(background, (0, 0))
        #for p in particles:
        #    p.density(particles)
        density_update(particles)
        force_update(particles)
        #euler_update(particles)
        #particles[0].force(particles);
        for p in particles:
            p.draw()
        pygame.display.flip()

if __name__ == "__main__":
    main()
