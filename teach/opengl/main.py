#Author: Ian Johnson
#referenced: http://www.geometrian.com/Tutorials.php

from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
from pygame.locals import *

#shortcut for initializing OpenGL attributes and scene
import GL
#utility functions for drawing OpenGL stuff
import glutil as gl
from vector import Vec

import os, sys

pygame.init()
pygame.display.set_caption("PyOpenGL Example")

screen = (800, 600)
surface = pygame.display.set_mode(screen, OPENGL|DOUBLEBUF)

#see GL.py
GL.resize(screen)
GL.init()


def get_input():
   key = pygame.key.get_pressed()
   #print key

   for event in pygame.event.get():
       if event.type == QUIT or key[K_ESCAPE] or key[K_q]:
           print "quit!"
           pygame.quit(); sys.exit()


def draw():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()


    glTranslatef(0.0, 0.0, -6.0)
    gl.draw_axes()

    glColor3f(1,1,1)
    glPointSize(5)
    glBegin(GL_POINTS)
    glVertex3f(0,0,0)
    glEnd()


    pygame.display.flip()


def main():
    
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(60)
        get_input()
        draw()


if __name__ == '__main__': main()
