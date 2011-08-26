#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush, glGenBuffers, glBindBuffer
from OpenGL.GL import *

import numpy
import pyopencl as cl
import glutil
import util
timings = util.timings

from system import CLSystem 
import permute
import density
import magnet 
import force
import collision_wall
import leapfrog


class CLMagnetSystem(CLSystem):
    def __init__(self, sph, dt=.001, is_ghost=False, ghost_system=None):
        CLSystem.__init__(self, sph, dt, is_ghost, ghost_system)
       
        self.density = density.CLDensity(self)
        self.magnet = magnet.CLMagnet(self)
        self.force = force.CLForce(self)
        self.collision_wall = collision_wall.CLCollisionWall(self)
        self.leapfrog = leapfrog.CLLeapFrog(self)

        self.random_inc = 0
        self.target_intensity = 1.0
 
       
    
    def acquire_gl(self):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
    def release_gl(self):
        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)


    def update(self):
        self.acquire_gl()


        numpy.set_printoptions(precision=6, linewidth=1000)
        """
        pos = numpy.ndarray((self.num,4), dtype=numpy.float32)
        cl.enqueue_read_buffer(self.queue, self.position_u, pos)
        print "pos_u at begining"
        print pos.T[0:100] 
        """

        self.exec_hash()

        if self.num > 0:
            self.exec_sort()
 

        #resetup the cell indices arrays 
        negone = numpy.ones((self.system.domain.nb_cells+1,), dtype=numpy.int32)
        negone *= -1
        cl.enqueue_write_buffer(self.queue, self.ci_start, negone)
        cl.enqueue_write_buffer(self.queue, self.ci_end, numpy.zeros((self.system.domain.nb_cells+1), dtype=numpy.int32))
        self.queue.finish()


        self.exec_cellindices()

        self.exec_permute()
        
        self.exec_density()
        
        #self.exec_magnet()

        self.exec_force()

        self.exec_collision_wall()

        self.exec_leapfrog()

            
        self.release_gl()
        

    @timings("Permute")
    def exec_permute(self):
        self.permute.execute(   self.num, 
                                self.position_u,
                                self.position_s,
                                self.velocity_u,
                                self.velocity_s,
                                self.veleval_u,
                                self.veleval_s,
                                self.color_u,
                                self.color_s,
                                self.sort_indices
                                #self.clf_debug,
                                #self.cli_debug
                            )

    @timings("Density")
    def exec_density(self):
        self.density.execute(   self.num, 
                                self.position_s,
                                self.density_s,
                                self.ci_start,
                                self.ci_end,
                                #self.gp,
                                self.gp_scaled,
                                self.systemp,
                                self.clf_debug,
                                self.cli_debug
                            )

    @timings("Magnet")
    def exec_magnet(self):
        for i in xrange(3):
            dt = self.dt * .1
            self.magnet.execute(   self.num, 
                                    self.position_s,
                                    self.density_s,
                                    self.color_s,
                                    numpy.float32(dt),
                                    self.ci_start,
                                    self.ci_end,
                                    #self.gp,
                                    self.gp_scaled,
                                    self.systemp,
                                    self.clf_debug,
                                    self.cli_debug
                                )





    @timings("Force")
    def exec_force(self):
        self.random_inc += 1
        self.force.execute(   self.num, 
                              self.position_s,
                              self.density_s,
                              self.veleval_s,
                              self.force_s,
                              self.xsph_s,
                              numpy.int32(self.random_inc),
                              self.ci_start,
                              self.ci_end,
                              self.gp_scaled,
                              self.systemp,
                              self.clf_debug,
                              self.cli_debug
                          )


    @timings("Collision Wall")
    def exec_collision_wall(self):
        self.collision_wall.execute(  self.num, 
                                      self.position_s,
                                      self.velocity_s,
                                      self.force_s,
                                      self.gp_scaled,
                                      self.systemp,
                                      #self.clf_debug,
                                      #self.cli_debug
                                   )



    @timings("Leapfrog")
    def exec_leapfrog(self):
        self.leapfrog.execute(    self.num, 
                                  self.position_u,
                                  self.position_s,
                                  self.velocity_u,
                                  self.velocity_s,
                                  self.color_u,
                                  self.color_s,
                                  self.veleval_u,
                                  self.force_s,
                                  self.xsph_s,
                                  self.sort_indices,
                                  self.systemp,
                                  #self.clf_debug,
                                  #self.cli_debug
                                  numpy.float32(self.dt)
                             )



    def render(self):

        gc = self.global_color
        glColor4f(gc[0],gc[1], gc[2],gc[3])
        glEnable(GL_POINT_SMOOTH)
        glPointSize(5)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable(GL_DEPTH_TEST)
        glDisable(GL_DEPTH_TEST)
        #glDepthMask(GL_FALSE)

        """
        glColor3f(1., 0, 0)
        glBegin(GL_POINTS)
        for p in self.pos_vbo.data:
            glVertex3f(p[0], p[1], p[2])

        glEnd()
        """

        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)

        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, None)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glDrawArrays(GL_POINTS, 0, self.num)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        #glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

