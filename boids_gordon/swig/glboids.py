from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys

from flock import *

import numpy as np

#number of particles
max_num = 2000
#time step for integration
dt = .01

class window(object):
    def __init__(self, *args, **kwargs):
        #mouse handling for transforming scene
        self.mouse_down = False

        self.width = 512 
        self.height = 512

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow("Part 2: Python")

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        
        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        self.dim = 300
       
        #setup OpenGL scene
        self.glinit()


        #set up initial conditions
        (self.boids, self.pos_vbo, self.col_vbo, self.num) = boids(max_num)
        #create our OpenCL instance
        #self.boids = boids.Boids(num, dt, self.dim)
        #self.boids.loadData(pos_vbo, col_vbo, vel, acc)
        self.boids.setDomainSize(self.dim)

        self.posnp = np.ndarray((self.num, 4), dtype=np.float32)
        glutMainLoop()


    def draw(self):
        """Render the particles"""        
        self.boids.update()
        glFlush()


        newpos = self.boids.getPos()
        for i in xrange(0, self.num):
            self.posnp[i,0] = newpos[i].x
            self.posnp[i,1] = newpos[i].y
            self.posnp[i,2] = 0.
            self.posnp[i,3] = 1.

        self.pos_vbo.set_array(self.posnp)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glBegin(GL_POINTS)
        glColor3f(1.,1.,1.)
        for p in self.posnp:
            glVertex2f(p[0], p[1]);
        glEnd()


        glEnable(GL_POINT_SMOOTH)
        glPointSize(2)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)

        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_POINTS, 0, self.num)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)        
        
        glutSwapBuffers()

    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-self.dim, self.dim, -self.dim, self.dim, -1, 1);
        #gluPerspective(60., self.width / float(self.height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            sys.exit()

def boids(max_num):
    bmin = float4(-100., -100., 0.,0.)
    bmax = float4(100., 100., 0.,0.)
    spacing = 10.
    #num, pos_tmp = addRect(max_num, bmin, bmax, spacing)
    pos = float4vec(max_num)
    GE_addRect(max_num, bmin, bmax, spacing, 1., pos)
    num = pos.size()
    print num


    posnp = np.zeros((num, 4), dtype=np.float32)

    for i in xrange(0, num):
        posnp[i,0] = pos[i].x
        posnp[i,1] = pos[i].y
        posnp[i,2] = 0.
        posnp[i,3] = 1.

    
    col = np.zeros((num, 4), dtype=np.float32)

    col[:,0] = 1.
    col[:,3] = 1.

    boy = Boids(pos)

    vel = float4vec(num)
    acc = float4vec(num)
    z = float4(0.,0.,0.,1.)
    for i in xrange(0, num):
        vel[i] = z
        acc[i] = z

    boy.set_ic(pos, vel, acc)
    
    #create the Vertex Buffer Objects
    from OpenGL.arrays import vbo 
    pos_vbo = vbo.VBO(data=posnp, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    pos_vbo.bind()
    col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    col_vbo.bind()

    return (boy, pos_vbo, col_vbo, num)


if __name__ == "__main__":
    w = window()

"""

print(a[0])
a[0] = intvec(10)
print(a[0][0])

b = float4vec(10)
print(dir(b))
print(dir(b[0]))

b[0].x = 1.;
b[0].y = 2.;
b[0].z = 0.;

b[1].x = -1.;
b[1].y = 0.;
b[1].z = 0.5; 

print(b[0])
print(b[0].x)

print("test")
b[0] = b[1] / 5.
print(b[0].x)

b[0] += b[1]
c = 3. * b[1] * 5.
print(c)
print(b[0].x)
print((b[0] + b[1]).x)


boy = Boids(b)

print(dir(boy))
print(boy)
"""
