from OpenGL.GL import *
from OpenGL.GLU import *

from vector import Vec


def draw_line(v1, v2):
    glBegin(GL_LINES)
    glVertex3f(v1.x, v1.y, v1.z)
    glVertex3f(v2.x, v2.y, v2.z)
    glEnd()


def draw_axes():
    #X Axis
    glColor3f(1,0,0)    #red
    v1 = Vec([0,0,0])
    v2 = Vec([1,0,0])
    draw_line(v1, v2)

    #Y Axis
    glColor3f(0,1,0)    #green
    v1 = Vec([0,0,0])
    v2 = Vec([0,1,0])
    draw_line(v1, v2)

    #Z Axis
    glColor3f(0,0,1)    #blue
    v1 = Vec([0,0,0])
    v2 = Vec([0,0,1])
    draw_line(v1, v2)


