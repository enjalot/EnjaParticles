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
import initialize
#import clsph
import clsystem
#from cldiffuse_system import CLDiffuseSystem, timings
from cldemo_system import CLDemoSystem, timings
from clghost import GhostSystem
#import clghost



dt = .001
subintervals = 5

test = 1
with_ghost = True

class window(object):

    def make_ghost_system(self):
        #########################################################################
        self.gdmin = Vec([0., 0., 0.,0.])
        self.gdmax = Vec([1.,1.,0.,0.])
 
        #ghost_max_num = 16384
        #ghost_max_num = 8192

        print "Ghost System"
        print "-------------------------------------------------------------"
        self.ghost_domain = hash.Domain(self.gdmin, self.gdmax)
        if test == 0:
            ghost_max_num = 65536
            self.ghost = sph.SPH(ghost_max_num, self.ghost_domain, ghost_factor=.01)
            self.clghost_system = GhostSystem(dt, self.ghost)


            gmin = Vec([0., 0., 0.,0.])
            gmax = Vec([.5,1.,0.,0.])

            #high_color = [.0, .1, .0, 1.]
            #low_color = [0., 0.0, .1, .1]
            high_color = [.3, .3, .3, 1.]
            low_color = [0.1, 0.1, .1, .1]
            
            ghost_pos, ghost_color = initialize.addRect(ghost_max_num/2, gmin, gmax, self.ghost, high_color)
            self.clghost_system.push_particles(ghost_pos, None, ghost_color)


            gmin = Vec([.5, 0., 0.,0.])
            gmax = Vec([1.,1.,0.,0.])
            ghost_pos, ghost_color = initialize.addRect(ghost_max_num/2, gmin, gmax, self.ghost, low_color)
            self.clghost_system.push_particles(ghost_pos, None, ghost_color)

        elif test == 1:
            ghost_max_num = 65536
            self.ghost = sph.SPH(ghost_max_num, self.ghost_domain, ghost_factor=.01)
            self.clghost_system = GhostSystem(dt, self.ghost)

            high_color = [.3, .3, .3, 1.]
            low_color = [0.1, 0.1, .1, .1]

            ghost_pos, ghost_color = initialize.test_center(ghost_max_num, self.ghost, high_color, low_color)
            self.clghost_system.push_particles(ghost_pos, None, ghost_color)

        elif test == 2:
            ghost_max_num = 65536
            self.ghost = sph.SPH(ghost_max_num, self.ghost_domain, ghost_factor=.01)
            self.clghost_system = GhostSystem(dt, self.ghost)

            high_color = [.3, .3, .3, 1.]
            low_color = [0.1, 0.1, .1, .1]

            ghost_pos, ghost_color = initialize.test_center(ghost_max_num, self.ghost, high_color, low_color, gradient=True)
            self.clghost_system.push_particles(ghost_pos, None, ghost_color)



        elif test == 3:
            ghost_max_num = 262144
            #ghost_max_num = 65536
            self.ghost = sph.SPH(ghost_max_num, self.ghost_domain, ghost_factor=.01)
            self.clghost_system = GhostSystem(dt, self.ghost)

            img = Image.open('images/test.jpg')
            #img = Image.open('images/shark.gif')
            #img = Image.open('images/vein.png')
            #img = Image.open('images/turtle_nom.png')
            #img = Image.open('images/tallydroiddev_logo_trans_small.png')
            print "image opened"
            #print img.size
            #img.show()
            ghost_pos, ghost_color = initialize.addPic(img, ghost_max_num, self.gdmin, self.gdmax, self.ghost)
            self.clghost_system.push_particles(ghost_pos, None, ghost_color)
        
        self.draw_ghosts = True


    def make_sph_system(self):

        #ghost_max_num = 8192
        #max_num = 32768
        if test == 0 or test == 1 or test == 2:
            max_num = 16384
            #max_num = 4096 
        elif test == 3:
            max_num = 32768*4

        #max_num = 8192
        #max_num = 2**12 #4096
        #max_num = 2**10 #1024
        #max_num = 2**8 #256
        #max_num = 2**7 #128

        #dmin = Vec([0.,0.,0.])
        #dmax = Vec([1.,1.,1.])
        print "SPH System"
        print "-------------------------------------------------------------"
        self.domain = hash.Domain(self.gdmin, self.gdmax)
        self.system = sph.SPH(max_num, self.domain)
        self.system.gravity = -9.8
        self.system.K = 3.
        self.system.xsph_factor = .02
        
        print "making particle system"
        #self.clsystem = clsph.CLSPH(dt, self.system, ghost_system=None)
        #self.clsystem = clsph.CLSPH(dt, self.system, ghost_system=self.clghost_system)
        if with_ghost:
            self.clsystem = CLDemoSystem(self.system, dt, ghost_system=self.clghost_system)
        else:
            self.clsystem = CLDemoSystem(self.system, dt)
        #self.clsystem = clsystem.CLSystem(self.system, dt, ghost_system=self.clghost_system)

        self.draw_particles = True


 
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

        self.draw_brush = True
        self.brush_radius = .1
        self.cx = 0.
        self.cy = 0.
 

        self.pause = False
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
        glutPassiveMotionFunc(self.on_mouse_move)
        glutReshapeFunc(self.on_resize)
        
        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        glViewport(0, 0, self.width, self.height)
        #setup OpenGL scene

        #########################################################################

        self.make_ghost_system()
        self.make_sph_system()


        print "ghost system update"
        self.clghost_system.update()
        #print"clsystem update"
        #self.clsystem.update()

        #########################################################################

        self.glprojection()
        glutMainLoop()
 

    def draw(self):
        """Render the particles"""        
        #TODO: 
        # * set up Ortho2D viewing and mouse controls
        # * find better color mapping for height

        
        if not self.pause:
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
        if self.draw_ghosts:
            self.clghost_system.render()
        if self.draw_particles:
            self.clsystem.render()

        #draw the x, y and z axis as lines
        glutil.draw_axes()

        #draw a box around cursor as "brush"
        if self.draw_brush:
            glColor3f(0., 1., 0.)
            cx = self.cx
            cy = self.cy
            r = self.brush_radius
            #print cx, cy, r
            glBegin(GL_LINES)
            glVertex2f(cx - r, cy - r)
            glVertex2f(cx + r, cy - r)

            glVertex2f(cx + r, cy + r)
            glVertex2f(cx - r, cy + r)

            glVertex2f(cx + r, cy - r)
            glVertex2f(cx + r, cy + r)

            glVertex2f(cx - r, cy - r)
            glVertex2f(cx - r, cy + r)

            glEnd()


        if self.mouse_down and self.button == 0:
            self.add_particles()

        glutSwapBuffers()

    def add_particles(self):
        import random
        rr = random.random()
        rb = random.random()
        #rr *= .1
        #rb *= .1
        color = [rr, 0.3, rb, .5]

        #color = [.2, 0., 0., .5]
        self.clsystem.set_color(color)
        cx = self.cx
        cy = self.cy
        r = self.brush_radius
        ipos, icolor = sph.addRect(512, Vec([cx - r, cy - r, 0.0,0.]), Vec([cx + r, cy + r, 0.0,0.]), self.system, color)
        print "pushing clsystem particles"
        self.clsystem.push_particles(ipos, None, icolor)



    def glprojection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.ortho:
            xmin = self.gdmin.x
            xmax = self.gdmax.x
            ymin = self.gdmin.y
            ymax = self.gdmax.y
            glOrtho(xmin, xmax, ymin, ymax, -2.5,2.5)
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
            print timings
        elif args[0] == 'p':
            self.pause = not self.pause
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
            print "density:", self.clsystem.with_ghost_density
        elif args[0] == 'f':
            self.clsystem.with_ghost_force = not self.clsystem.with_ghost_force
            print "force:", self.clsystem.with_ghost_force
        elif args[0] == 'a':
            #add particles
            pass
        elif args[0] == 'w':
            #whirlpool
            pass
        elif args[0] == 'g':
            self.draw_ghosts = not self.draw_ghosts
        elif args[0] == 'p':
            self.draw_particles = not self.draw_particles

        #begin mouse UI commands
        elif args[0] == '-':
            self.brush_radius -= .02
        elif args[0] == '=':
            self.brush_radius += .02

        #change target intensity
        elif args[0] == '[':
            self.clsystem.target_intensity -= .05
            print "target intensity:", self.clsystem.target_intensity
        elif args[0] == ']':
            self.clsystem.target_intensity += .05
            print "target intensity:", self.clsystem.target_intensity


             

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button

            print x, y, self.width, self.height
            cx = self.gdmax.x * x / (1.*self.width)
            cy = (self.height - y) / (self.gdmax.y * self.height)
            self.cx = cx
            self.cy = cy
            print "CX, CY", cx, cy

            self.add_particles() 
 
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y


    
    def on_mouse_motion(self, x, y):
        print "mouse motion"
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            pass
            #self.rotate.x += dy * .2
            #self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .01 
        self.mouse_old.x = x
        self.mouse_old.y = y

        cx = self.gdmax.x * x / (1.*self.width)
        cy = (self.height - y) / (self.gdmax.y * self.height)
        self.cx = cx
        self.cy = cy



        #glutPostRedisplay()

    def on_mouse_move(self, x, y):
        #print "MOTION"
        cx = self.gdmax.x * x / (1.*self.width)
        cy = (self.height - y) / (self.gdmax.y * self.height)
        self.cx = cx
        self.cy = cy

    ###END GL CALLBACKS

    def on_resize(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        glutPostRedisplay()



if __name__ == "__main__":
    p2 = window()



