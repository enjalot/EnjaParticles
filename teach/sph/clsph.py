#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush, glGenBuffers, glBindBuffer
from OpenGL.GL import *

import numpy
import pyopencl as cl
import glutil

import clhash
import clradix
import clcellindices
import clpermute
import cldensity
import clforce
import clcollision_wall
import clleapfrog
import clghost_density
import clghost_force

class CLSPH:
    def __init__(self, dt, sph, is_ghost=False, ghost_system=None):
        #ghost system is just a regular system that doesn't do all the steps of the sph update
        self.is_ghost = is_ghost
        #our real system has access to the arrays from the ghost system
        self.ghost_system = ghost_system
        if is_ghost or ghost_system is None:
            self.clinit()
        else:
            self.ctx = self.ghost_system.ctx
            self.queue = self.ghost_system.queue

        self.prgs = {}  #store our programs
        #of course hardcoding paths here is terrible
        #self.clsph_dir = "/Users/enjalot/code/sph/teach/sph/cl_src"
        self.clsph_dir = "cl_src"
        #self.clcommon_dir = "/Users/enjalot/code/sph/teach/sph/cl_common"
        self.clcommon_dir = "cl_common"

        self.global_color = [1., 1., 1., 1.]
        
        self.dt = dt
        self.num = 0
        self.sph = sph

        self.with_ghost_density = True
        self.with_ghost_force = True
        
        self.loadData()

        self.hash = clhash.CLHash(self)
        self.radix = clradix.Radix(self, self.sph.max_num, 128, numpy.uint32(0))
        self.cellindices = clcellindices.CLCellIndices(self)
        self.permute = clpermute.CLPermute(self)
        if not is_ghost:
            self.density = cldensity.CLDensity(self)
            self.force = clforce.CLForce(self)
            self.collision_wall = clcollision_wall.CLCollisionWall(self)
            self.leapfrog = clleapfrog.CLLeapFrog(self)
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


        self.hash.execute(      self.num,
                                self.position_u,
                                self.sort_hashes,
                                self.sort_indices,
                                self.gp,
                                self.clf_debug,
                                self.cli_debug
                            )

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
 

        self.radix.sort(    self.sph.max_num,
                            self.sort_hashes,
                            self.sort_indices
                        )


 

        negone = numpy.ones((self.sph.domain.nb_cells+1,), dtype=numpy.int32)
        negone *= -1
        cl.enqueue_write_buffer(self.queue, self.ci_start, negone)

        cl.enqueue_write_buffer(self.queue, self.ci_end, numpy.zeros((self.sph.domain.nb_cells+1), dtype=numpy.int32))
        self.queue.finish()

        self.cellindices.execute(   self.num,
                                    self.sort_hashes,
                                    self.sort_indices,
                                    self.ci_start,
                                    self.ci_end,
                                    self.gp,
                                    #self.clf_debug,
                                    #self.cli_debug
                                )

        """
        tmp_start = numpy.ndarray((self.sph.domain.nb_cells,), dtype=numpy.uint32)
        tmp_end = numpy.ndarray((self.sph.domain.nb_cells,), dtype=numpy.uint32)
        cl.enqueue_read_buffer(self.queue, self.ci_start, tmp_start)
        cl.enqueue_read_buffer(self.queue, self.ci_end, tmp_end)
        import sys
        for i in xrange(len(tmp_start)):
            if tmp_start[i] <= self.sph.domain.nb_cells+1 or tmp_end[i] > 0:
                print i, tmp_start[i], tmp_end[i]
        """

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

        if not self.is_ghost:

        #if True:


            self.density.execute(   self.num, 
                                    self.position_s,
                                    self.density_s,
                                    self.ci_start,
                                    self.ci_end,
                                    #self.gp,
                                    self.gp_scaled,
                                    self.sphp,
                                    self.clf_debug,
                                    self.cli_debug
                                )

 
            if self.ghost_system is not None and self.with_ghost_density:
                self.ghost_density.execute(   self.num, 
                                    self.position_s,
                                    self.ghost_system.position_s,
                                    self.density_s,
                                    self.ghost_density_s,
                                    self.ghost_system.color_s,
                                    self.sphp,
                                    self.ghost_system.ci_start,
                                    self.ghost_system.ci_end,
                                    self.ghost_system.gp_scaled,
                                    self.ghost_system.sphp,
                                    self.clf_debug,
                                    self.cli_debug
                                )



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
 
            self.force.execute(   self.num, 
                                  self.position_s,
                                  self.density_s,
                                  self.veleval_s,
                                  self.force_s,
                                  self.xsph_s,
                                  self.ci_start,
                                  self.ci_end,
                                  self.gp_scaled,
                                  self.sphp,
                                  self.clf_debug,
                                  self.cli_debug
                              )

            """
            clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
            print "clf"
            print clf[0:100].T
            """
 


            if self.ghost_system is not None and self.with_ghost_force:
                self.ghost_force.execute(   self.num, 
                                    self.position_s,
                                    self.ghost_system.position_s,
                                    self.density_s,
                                    self.ghost_density_s,
                                    self.ghost_system.color_s,
                                    self.veleval_s,
                                    self.force_s,
                                    self.xsph_s,
                                    self.sphp,
                                    self.ghost_system.ci_start,
                                    self.ghost_system.ci_end,
                                    self.ghost_system.gp_scaled,
                                    self.ghost_system.sphp,
                                    self.clf_debug,
                                    self.cli_debug
                )

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


            self.collision_wall.execute(  self.num, 
                                          self.position_s,
                                          self.velocity_s,
                                          self.force_s,
                                          self.gp_scaled,
                                          self.sphp,
                                          #self.clf_debug,
                                         # self.cli_debug
                                       )


            self.leapfrog.execute(    self.num, 
                                      self.position_u,
                                      self.position_s,
                                      self.velocity_u,
                                      self.velocity_s,
                                      self.veleval_u,
                                      self.force_s,
                                      self.xsph_s,
                                      self.sort_indices,
                                      self.sphp,
                                      #self.clf_debug,
                                      #self.cli_debug
                                      numpy.float32(self.dt)
                                 )

            """
            cl.enqueue_read_buffer(self.queue, self.position_u, pos)
            print "position_u after leapfrog"
            print pos.T[0:100] 
            """
 

            #"""                


        self.release_gl()

    def loadData(self):#, pos_vbo, col_vbo):
        import pyopencl as cl
        mf = cl.mem_flags
        
        #placeholder array used to fill cl buffers
        #could just specify size but might want some initial values later
        tmp = numpy.zeros((self.sph.max_num, 4), dtype=numpy.float32)
        self.pos_vbo = glutil.VBO(tmp)
        self.col_vbo = glutil.VBO(tmp)

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()
        self.position_u = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.vbo_id))

        self.col_vbo.bind()
        self.color_u = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.vbo_id))

        #pure OpenCL arrays
        self.velocity_u = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.velocity_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.veleval_u = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.veleval_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)

        self.position_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.color_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)

        tmp_dens = numpy.zeros((self.sph.max_num,), dtype=numpy.float32)
        self.density_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_dens)
        self.force_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.xsph_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)

        if not self.is_ghost or self.ghost_system is not None:
            self.ghost_density_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_dens)

        import sys
        tmp_uint = numpy.ones((self.sph.max_num,), dtype=numpy.uint32)
        tmp_uint = tmp_uint * sys.maxint

        self.sort_indices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_uint)
        self.sort_hashes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_uint)

        tmp_grid = numpy.ones((self.sph.domain.nb_cells+1, ), dtype=numpy.int32)
        tmp_grid += -1
        #grid size
        self.ci_start= cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_grid)
        self.ci_end = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_grid)

        #make struct buffers

        self.sphp_struct = self.sph.make_struct(self.num)
        self.sphp = cl.Buffer(self.ctx, mf.READ_ONLY, len(self.sphp_struct))
        cl.enqueue_write_buffer(self.queue, self.sphp, self.sphp_struct).wait()

        self.gp_struct = self.sph.domain.make_struct(1.0)
        self.gp = cl.Buffer(self.ctx, mf.READ_ONLY, len(self.gp_struct))
        cl.enqueue_write_buffer(self.queue, self.gp, self.gp_struct)

        self.gp_struct_scaled = self.sph.domain.make_struct(self.sph.sim_scale)
        self.gp_scaled = cl.Buffer(self.ctx, mf.READ_ONLY, len(self.gp_struct_scaled))
        cl.enqueue_write_buffer(self.queue, self.gp_scaled, self.gp_struct_scaled)

        #debug arrays
        tmp_int = numpy.zeros((self.sph.max_num, 4), dtype=numpy.int32)
        self.clf_debug = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.cli_debug = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_int)

        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.position_u, self.color_u]


    def update_sphp(self):
        self.sphp_struct = self.sph.make_struct(self.num)
        print "update sphp", self.num
        cl.enqueue_write_buffer(self.queue, self.sphp, self.sphp_struct).wait()


    def push_particles(self, pos, vel, color):
        nn = pos.shape[0]
        print "NN", nn
        print self.num
        print self.sph.max_num
        if self.num + nn > self.sph.max_num:
            return

        self.acquire_gl()
        offset = self.num * numpy.float32(0).itemsize*4
        cl.enqueue_write_buffer(self.queue, self.position_u, pos, device_offset=offset)
        cl.enqueue_write_buffer(self.queue, self.color_u, color, device_offset=offset)
        self.release_gl()

        self.num += nn
        self.update_sphp()

        self.queue.finish()



 
    def clinit(self):
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        import sys 
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)
                
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print fstr
        #create the program
        prg_name = filename.split(".")[0]   #e.g. wave from wave.cl
        print prg_name
        prg_name = prg_name.split("/")[-1]
        print prg_name
        optionstr = "-I%s/ -I%s/" % (self.clsph_dir, self.clcommon_dir)
        #print optionstr

        plat = cl.get_platforms()[0]
        device = plat.get_devices()[0]

        print "prg name", prg_name
        print filename
        self.prgs[prg_name] = cl.Program(self.ctx, fstr).build(options=optionstr)
        options = self.prgs[prg_name].get_build_info(device, cl.program_build_info.OPTIONS)
        #print "options: ", options
        #print "kernel", dir(self.prgs[prg_name])

    def set_color(self, color):
        self.global_color = color

    def render(self):

        gc = self.global_color
        glColor4f(gc[0],gc[1], gc[2],gc[3])
        glEnable(GL_POINT_SMOOTH)
        if self.is_ghost:
            glPointSize(2)
        else:
            glPointSize(5)

        glEnable(GL_BLEND)
        #glBlendFunc(GL_ONE, GL_ONE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
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

