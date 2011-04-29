#from OpenGL.GL import *
import OpenGL.GL as gl
from OpenGL.GLU import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.arrays import ArrayDatatype as ADT

from vector import Vec


class VBO:
#hacking this together because on Mac I can't use the PyOpenGL vbo class
#mixed with PyOpenCL for some reason
    def __init__(self, data):
        self.data = data
        self.vbo_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        rawGlBufferData(gl.GL_ARRAY_BUFFER, ADT.arrayByteCount(data), ADT.voidDataPointer(data), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def bind(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)



def init((width, height)):

    #glEnable(GL_DEPTH_TEST)
    gl.glEnable(gl.GL_NORMALIZE)
    gl.glShadeModel(gl.GL_SMOOTH)


    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gluPerspective(60.0, width/float(height), 1, 1000.0)
    #glEnable(GL_DEPTH_TEST)
    gl.glMatrixMode(gl.GL_MODELVIEW)




def lights():
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_COLOR_MATERIAL)

    light_position = [10., 10., 200., 0.]
    light_ambient = [.2, .2, .2, 1.]
    light_diffuse = [.6, .6, .6, 1.]
    light_specular = [2., 2., 2., 0.]
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular)
    gl.glEnable(gl.GL_LIGHT0)

"""
    mat_ambient = [.2, .2, 1.0, 1.0]
    mat_diffuse = [.2, .8, 1.0, 1.0]
    mat_specular = [1.0, 1.0, 1.0, 1.0]
    high_shininess = 3.

    mat_ambient_back = [.5, .2, .2, 1.0]
    mat_diffuse_back = [1.0, .2, .2, 1.0]

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    glMaterialfv(GL_BACK, GL_AMBIENT,   mat_ambient_back);
    glMaterialfv(GL_BACK, GL_DIFFUSE,   mat_diffuse_back);
    glMaterialfv(GL_BACK, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_BACK, GL_SHININESS, high_shininess);
"""


def draw_line(v1, v2):
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(v1.x, v1.y, v1.z)
    gl.glVertex3f(v2.x, v2.y, v2.z)
    gl.glEnd()


def draw_axes():
    #X Axis
    gl.glColor3f(1,0,0)    #red
    v1 = Vec([0,0,0])
    v2 = Vec([1,0,0])
    draw_line(v1, v2)

    #Y Axis
    gl.glColor3f(0,1,0)    #green
    v1 = Vec([0,0,0])
    v2 = Vec([0,1,0])
    draw_line(v1, v2)

    #Z Axis
    gl.glColor3f(0,0,1)    #blue
    v1 = Vec([0,0,0])
    v2 = Vec([0,0,1])
    draw_line(v1, v2)



