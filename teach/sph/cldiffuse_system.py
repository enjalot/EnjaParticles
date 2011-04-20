#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush, glGenBuffers, glBindBuffer
from OpenGL.GL import *

import numpy
import pyopencl as cl
import glutil
import util
timings = util.timings

from clsystem import CLSystem 
import clpermute
import cldensity
import cldiffuse 
import clforce
import clcollision_wall
import clleapfrog
import clleapfrog_diffuse
import clghost_density
import clghost_force



class CLDiffuseSystem(CLSystem):
    def __init__(self, sph, dt=.001, is_ghost=False, ghost_system=None):
        CLSystem.__init__(self, sph, dt, is_ghost, ghost_system)
       
        self.with_ghost_density = True
        self.with_ghost_force = True
 
        self.density = cldensity.CLDensity(self)
        self.diffuse = cldiffuse.CLDiffuse(self)
        self.force = clforce.CLForce(self)
        self.collision_wall = clcollision_wall.CLCollisionWall(self)
        #self.leapfrog = clleapfrog.CLLeapFrog(self)
        self.leapfrog = clleapfrog_diffuse.CLLeapFrogDiffuse(self)
        self.ghost_density = clghost_density.CLGhostDensity(self)
        self.ghost_force = clghost_force.CLGhostForce(self)
 
       
    
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

        """
        hashes = numpy.ndarray((self.num,), dtype=numpy.int32)
        cl.enqueue_read_buffer(self.queue, self.sort_hashes, hashes)
        print "hashes"
        print hashes.T
        indices = numpy.ndarray((self.num,), dtype=numpy.int32)
        cl.enqueue_read_buffer(self.queue, self.sort_indices, indices)
        print "indices"
        print indices.T
        """
 

        self.exec_sort()
 

        negone = numpy.ones((self.system.domain.nb_cells+1,), dtype=numpy.int32)
        negone *= -1
        cl.enqueue_write_buffer(self.queue, self.ci_start, negone)

        cl.enqueue_write_buffer(self.queue, self.ci_end, numpy.zeros((self.system.domain.nb_cells+1), dtype=numpy.int32))
        self.queue.finish()

        self.exec_cellindices()


        """
        tmp_start = numpy.ndarray((self.system.domain.nb_cells,), dtype=numpy.uint32)
        tmp_end = numpy.ndarray((self.system.domain.nb_cells,), dtype=numpy.uint32)
        cl.enqueue_read_buffer(self.queue, self.ci_start, tmp_start)
        cl.enqueue_read_buffer(self.queue, self.ci_end, tmp_end)
        import sys
        for i in xrange(len(tmp_start)):
            if tmp_start[i] <= self.system.domain.nb_cells+1 or tmp_end[i] > 0:
                print i, tmp_start[i], tmp_end[i]
        """

        self.exec_permute()
        
        if not self.is_ghost:
        #if True:
            self.exec_density()
            
            self.exec_diffuse()
            """
            color = numpy.ndarray((self.num,4), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.color_s, color)
            print color.T
            """
             
            if self.ghost_system is not None and self.with_ghost_density:
                self.exec_ghost_density()



            """
            density = numpy.ndarray((self.num,), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.density_s, density)
            print "density"
            print density.T


            gdensity = numpy.ndarray((self.num,), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.ghost_density_s, gdensity)
            print gdensity.T


            clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
            print clf
            cl.enqueue_read_buffer(self.queue, self.xsph_s, pos)
            print "xpsh_s before force"
            print pos.T[0:100] 
            """
 
            self.exec_force()

            """
            clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
            print "clf"
            print clf[0:100].T
            """
 


            if self.ghost_system is not None and self.with_ghost_force:
                self.exec_ghost_force()

                """
                force = numpy.ndarray((self.num,4), dtype=numpy.float32)
                cl.enqueue_read_buffer(self.queue, self.force_s, force)
                print force.T[0:100] 


                clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
                cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
                print clf.T
                """
     
 
            """
            cl.enqueue_read_buffer(self.queue, self.force_s, pos)
            print "force_s after force"
            print pos.T[0:100] 
            cl.enqueue_read_buffer(self.queue, self.position_u, pos)
            print "position_u before leapfrog"
            print pos.T[0:100] 
            print "num", self.num
            cl.enqueue_read_buffer(self.queue, self.velocity_s, pos)
            print "velocity_s before leapfrog"
            print pos.T[0:100] 

            """

            self.exec_collision_wall()

           
            self.exec_leapfrog()

            """
            cl.enqueue_read_buffer(self.queue, self.position_u, pos)
            print "position_u after leapfrog"
            print pos.T[0:100] 
            """
 

            #"""                


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

    @timings("Diffuse")
    def exec_diffuse(self):
        self.diffuse.execute(   self.num, 
                                self.position_s,
                                self.density_s,
                                self.color_s,
                                self.ci_start,
                                self.ci_end,
                                #self.gp,
                                self.gp_scaled,
                                self.systemp,
                                self.clf_debug,
                                self.cli_debug
                            )




    @timings("Ghost Density")
    def exec_ghost_density(self):
        self.ghost_density.execute( self.num, 
                                    self.position_s,
                                    self.ghost_system.position_s,
                                    self.density_s,
                                    self.ghost_density_s,
                                    self.ghost_system.color_s,
                                    self.systemp,
                                    self.ghost_system.ci_start,
                                    self.ghost_system.ci_end,
                                    self.ghost_system.gp_scaled,
                                    self.ghost_system.sphp,
                                    self.clf_debug,
                                    self.cli_debug
                                )


    @timings("Force")
    def exec_force(self):
        self.force.execute(   self.num, 
                              self.position_s,
                              self.density_s,
                              self.veleval_s,
                              self.force_s,
                              self.xsph_s,
                              self.ci_start,
                              self.ci_end,
                              self.gp_scaled,
                              self.systemp,
                              self.clf_debug,
                              self.cli_debug
                          )

    @timings("Ghost Force")
    def exec_ghost_force(self):
        self.ghost_force.execute(   self.num, 
                                    self.position_s,
                                    self.ghost_system.position_s,
                                    self.density_s,
                                    self.ghost_density_s,
                                    self.ghost_system.color_s,
                                    self.veleval_s,
                                    self.force_s,
                                    self.xsph_s,
                                    self.systemp,
                                    self.ghost_system.ci_start,
                                    self.ghost_system.ci_end,
                                    self.ghost_system.gp_scaled,
                                    self.ghost_system.sphp,
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
        if self.is_ghost:
            glPointSize(2)
        else:
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

