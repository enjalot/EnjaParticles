#basic glut setup learned from here:
#http://www.java2s.com/Open-Source/Python/Game-2D-3D/PyOpenGL/PyOpenGL-Demo-3.0.1b1/PyOpenGL-Demo/NeHe/lesson2.py.htm

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys

import Image

#helper modules
import glutil
from vector import Vec

#from forces import *
import hash
import forces
import sph
import clsph
import clghost

dt = .001
subintervals = 1


class window(object):

    def make_ghost_system(self):
        #########################################################################
        ghost_max_num = 262144
        #ghost_max_num = 65536
        #ghost_max_num = 16384
        #ghost_max_num = 8192

        print "Ghost System"
        print "-------------------------------------------------------------"
        gdmin = Vec([0.,0.,0.])
        gdmax = Vec([1.,1.,1.])
        self.ghost_domain = hash.Domain(gdmin, gdmax)
        self.ghost = sph.SPH(ghost_max_num, self.ghost_domain, ghost_factor=.01)
        #self.ghost = sph.SPH(max_num, self.ghost_domain, ghost_factor=.01)

        print "making ghost system"
        self.clghost_system = clsph.CLSPH(dt, self.ghost, is_ghost=True)
        #self.clghost_system = clsph.CLSPH(dt, self.system, is_ghost=True)
        #self.clsystem = clsph.CLSPH(dt, self.system, ghost_system=self.clghost_system)
        #self.clsystem = clsph.CLSPH(dt, self.system, ghost_system=None)
        ##self.clghost_system = clghost.CLGHOST(dt, self.ghost)

        gmin = Vec([0., 0., 0.,0.])
        gmax = Vec([1.,1.,0.,0.])
        color = [1., 0.75, 0.75, .5]
        #self.clghost_system.set_color(color)
        #ghost_pos, ghost_color = sph.addRect(512, Vec([0.1, 0.1, 0.,0.]), Vec([1.,1.,0.,0.]), self.system, color)
        #ghost_pos, ghost_color = sph.addRect(ghost_max_num, gmin, gmax, self.ghost, color)

        img = Image.open('test.jpg')
        print img.size
        #img.show()
        ghost_pos, ghost_color = sph.addPic(img, ghost_max_num, gmin, gmax, self.ghost)
        #print ghost_pos
        self.clghost_system.push_particles(ghost_pos, None, ghost_color)


    def make_sph_system(self):

        #ghost_max_num = 8192
        max_num = 16384
        #max_num = 8192
        #max_num = 2**12 #4096
        #max_num = 2**10 #1024
        #max_num = 2**8 #256
        #max_num = 2**7 #128

        dmin = Vec([0.,0.,0.])
        dmax = Vec([1.,1.,1.])
        print "SPH System"
        print "-------------------------------------------------------------"
        self.domain = hash.Domain(dmin, dmax)
        self.system = sph.SPH(max_num, self.domain)
        
        print "making particle system"
        #self.clsystem = clsph.CLSPH(dt, self.system, ghost_system=None)
        self.clsystem = clsph.CLSPH(dt, self.system, ghost_system=self.clghost_system)

        """
        color = [1., 0., 0., .75]
        self.clsystem.set_color(color)
        ipos, icolor = sph.addRect(512, Vec([.5, .5, 0.,0.]), Vec([1., 1., 0.,0.]), self.system, color)
        print "pushing clsystem particles"
        self.clsystem.push_particles(ipos, None, icolor)
        """


 
    def __init__(self, *args, **kwargs):
        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 0.])
        self.translate = Vec([0., 0., 0.])
        #self.initrans = Vec([0., 0., -2.])
        self.init_persp_trans = Vec([-.5, -0.5, -2.5])
        self.init_ortho_trans = Vec([0., 0., 1.])
        self.init_persp_rotate = Vec([0., 0., 0.])
        #self.init_ortho_rotate = Vec([90., -90., 0.])
        self.init_ortho_rotate = Vec([0., 0., 0.])
 

        self.ortho = True
        self.dt = dt

        self.width = 640
        self.height = 480

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow("SPH: Python")

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)
        
        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        glViewport(0, 0, self.width, self.height)
        #setup OpenGL scene
        self.glprojection()


        self.make_ghost_system()
        self.make_sph_system()


        print "ghost system update"
        self.clghost_system.update()
        #print"clsystem update"
        #self.clsystem.update()

        #########################################################################
        glutMainLoop()
 

    def draw(self):
        """Render the particles"""        
        #TODO: 
        # * set up Ortho2D viewing and mouse controls
        # * find better color mapping for height

        
        #update or particle positions by calling the OpenCL kernel
        for i in range(subintervals):
            self.clsystem.update()
            pass
        #self.cle.execute(subintervals) 
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        #glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0) 
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)
        
        #render the particles
        self.clghost_system.render()
        self.clsystem.render()

        #draw the x, y and z axis as lines
        glutil.draw_axes()

        glutSwapBuffers()


    def glprojection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.ortho:
            glOrtho(-.1, 1.2, -.1, 1.2, -2.5,2.5)
            self.translate = self.init_ortho_trans.copy()
            self.rotate = self.init_ortho_rotate.copy()
        else:
            gluPerspective(60., self.width / float(self.height), .1, 1000.)
            self.translate= self.init_persp_trans.copy()
            self.rotate = self.init_persp_rotate.copy()

        glMatrixMode(GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            sys.exit()
        elif args[0] == 't':
            print initialize.timings
        elif args[0] == 'o':
            self.ortho = not self.ortho
            if self.ortho:
                self.translate = self.init_ortho_trans.copy()
                self.rotate = self.init_ortho_rotate.copy()
            else:
                self.translate = self.init_persp_trans.copy()
                self.rotate = self.init_persp_rotate.copy()
            self.glprojection()
        elif args[0] == 'd':
            self.clsystem.with_ghost_density = not self.clsystem.with_ghost_density
        elif args[0] == 'f':
            self.clsystem.with_ghost_force = not self.clsystem.with_ghost_force
             

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y

        print x, y, self.width, self.height
        cx = x / (1.0 * self.width)
        cy = (self.height - y) / (1.0 * self.height)
        print cx, cy

        color = [1., 0., 0., .75]
        self.clsystem.set_color(color)
        ipos, icolor = sph.addRect(512, Vec([cx - .1, cy -.1, 0.,0.]), Vec([cx + .1, cy + .1, 0.,0.]), self.system, color)
        print "pushing clsystem particles"
        self.clsystem.push_particles(ipos, None, icolor)



    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .01 
        self.mouse_old.x = x
        self.mouse_old.y = y
    ###END GL CALLBACKS




if __name__ == "__main__":
    p2 = window()



