#Author: Ian Johnson
#referenced: http://www.geometrian.com/Tutorials.php

from OpenGL.GL import *
from OpenGL.GLU import *

from OpenGL.arrays import vbo

import pygame
from pygame.locals import *

#shortcut for initializing OpenGL attributes and scene
import GL
#utility functions for drawing OpenGL stuff
import glutil as gl
from vector import Vec, normalize

import os, sys
#import pickle
from math import sqrt, sin, cos
import numpy

pygame.init()
pygame.display.set_caption("PyOpenGL Example")

screen = (800, 600)
surface = pygame.display.set_mode(screen, OPENGL|DOUBLEBUF)


#should just have an interaction class for controlling the window
#global mouse_old, rotate, translate, mouse_down
mouse_down = False
mouse_old = Vec([0.,0.])
rotate = Vec([0., 0., 0.])
#translate = Vec([-10.,-10.,-36.])
translate = Vec([0., 0., 0.])
initrans = Vec([0, 0, -6])


#see GL.py
#GL.resize(screen)
#GL.init()

gl.init(screen)
#gl.lights()



def get_input():
    global mouse_down, mouse_old, translate, rotate
    key = pygame.key.get_pressed()
    #print key

    trans = 2.0


    for event in pygame.event.get():
        if event.type == QUIT or key[K_ESCAPE] or key[K_q]:
            print "quit!"
            pygame.quit(); sys.exit()

        elif event.type == MOUSEBUTTONDOWN:
            print "MOUSE DOWN"
            mouse_down = True
            mouse_old = Vec([event.pos[0]*1., event.pos[1]*1.])

        elif event.type == MOUSEMOTION:
            if(mouse_down):
                print "MOUSE MOTION"
                m = Vec([event.pos[0]*1., event.pos[1]*1.])
                dx = m.x - mouse_old.x
                dy = m.y - mouse_old.y
                button1, button2, button3 = pygame.mouse.get_pressed()
                if button1:
                    rotate.x += dy * .2
                    rotate.y += dx * .2
                elif button3:
                    translate .z -= dy * .01 

                mouse_old = m

                print "rotate", rotate, "translate", translate

        elif event.type == MOUSEBUTTONUP:
            print "MOUSE UP"
            mouse_down = False

        elif key[K_w]:
            translate.z += .1*trans   #y is z and z is y
        elif key[K_s]:
            translate.z -= .1*trans
        elif key[K_a]:
            translate.x += .1*trans
        elif key[K_d]:
            translate.x -= .1*trans



    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(initrans.x, initrans.y, initrans.z)
    #glTranslatef(-10, -10, -30)
    #glRotatef(-90, 1, 0, 0)
    glRotatef(rotate.x, 1, 0, 0)
    glRotatef(rotate.y, 0, 1, 0) #we switched around the axis so make this rotate_z
    glTranslatef(translate.x, translate.y, translate.z)
    #glRotatef(rotate.y, 0, 0, 1) #we switched around the axis so make this rotate_z
    #glTranslatef(translate.x, translate.z, translate.y)





def orthovec(u):
    """return arbitrary orthogonal vectors to u"""
    v = Vec([1., -u.x/u.y, 0.])
    b = 1.
    a = b*u.x/u.y
    c = -b*(u.x**2 + u.y**2)/(u.y*u.z)
    w = Vec([a, b, c])
    print "u dot v", numpy.dot(u, v)
    print "w dot v", numpy.dot(v, w)
    print "w dot u", numpy.dot(u, w)
    return normalize(v), normalize(w)

def rotateaxis(u,v,theta):
    """rotate v about u by angle theta (in degrees)"""
    theta *= numpy.pi/180.
    #matrix gotten from http://www.fastgraph.com/makegames/3drotation/ and originally from Graphics Gems 1990
    s = sin(theta)
    c = cos(theta)
    t = 1. - cos(theta)
    x = u.x
    y = u.y
    z = u.z
    R = numpy.array([[t*x**2 + c, t*x*y - s*z, t*x*z + s*y, 0.],
                     [t*x*y + s*z, t*y**2 + c, t*y*z - s*x, 0.],
                     [t*x*z - s*y, t*y*z + s*x, t*z**2 + c, 0.],
                     [0., 0., 0., 1.]])

    v = Vec([v.x, v.y, v.z, 1.])
    r = Vec(numpy.dot(R, v))
    v = normalize(Vec([r.x, r.y, r.z]))
    return v



def distribute_disc(p, u, v, r, n, spacing):
    """ distribute maximum of n points in a disc on the u, v plane within radius r
        spaced evenly every spacing units
        p is a 3D point
        u and v should be normal vectors
        r is a scalar radius

    """
    umin = -u*r
    #umax = p + u*r
    vmin = -v*r
    #umax = p + v*r
    particles = []
    count = 0
    for du in numpy.arange(0, 2.*r, spacing):
        for dv in numpy.arange(0, 2.*r, spacing):
            if count > n: break
            x = umin + u*du
            y = vmin + v*dv
            part = p + x + y
            if sqrt(numpy.dot(part-p, part-p)) <= r:
                count += 1
                particles += [ part ]
    
    return particles

def explicit_count(p, u, v, r1, r2, n, spacing):
    """ distribute maximum of n points in a disc on the u, v plane within radius r
        spaced evenly every spacing units
        p is a 3D point
        u and v should be normal vectors
        r is a scalar radius

    """
    umin = -u*r
    #umax = p + u*r
    vmin = -v*r
    #umax = p + v*r
    particles = []
    count = 0
    for du in numpy.arange(0, 2.*r, spacing):
        for dv in numpy.arange(0, 2.*r, spacing):
            if count > n: break
            x = umin + u*du
            y = vmin + v*dv
            part = p + x + y
            if sqrt(numpy.dot(part-p, part-p)) <= r:
                count += 1
                particles += [ part ]
    
    return particles



def draw():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


    #glLoadIdentity()
    #glTranslatef(-10.0, -10.0, -36.0)
    #glTranslatef(0.,0.,-6.)

    gl.draw_axes()

    glColor3f(1,1,1)
    glPointSize(5)
    #glBegin(GL_POINTS)
    #glVertex3f(0,0,0)
    #glVertex3f(1,1,1)
    #glVertex3f(0, 1, 0)
    #glEnd()

    #origin
    o = Vec([0.,0.,0.])
    #direction
    u = normalize(Vec([1., 1., 1.]))
    #orthogonal vec
    v,w = orthovec(u)
    #w = rotateaxis(u, v, 90)

    #print u
    #print v
    #print w


    glColor3f(.5, 1, 0)
    gl.draw_line(o, u)
    glColor3f(0, 1, .5)
    gl.draw_line(o, v)
    glColor3f(.5, 0, .5)
    gl.draw_line(o, w)

    particles = distribute_disc(u, v, w, 1, 300, .1)
    print len(particles)
    glBegin(GL_POINTS)
    glColor3f(1, 1, 1)
    glVertex3f(u.x, u.y, u.z)
    glColor3f(.5, .5, .5)
    for p in particles:
        glVertex3f(p.x, p.y, p.z)
    glEnd()

    pygame.display.flip()


def main():

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        get_input()
        draw()


if __name__ == '__main__': main()
