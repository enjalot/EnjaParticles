#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush, glGenBuffers, glBindBuffer

import numpy
import pyopencl as cl
import glutil

import clhash
import clradix
import clcellindices
"""
import clpermute
"""
import cldensity
"""
import clforce
import clcollision_wall
import clleapfrog
"""

class CLSPH:
    def __init__(self, dt, sph):
        self.clinit()
        self.prgs = {}  #store our programs
        #of course hardcoding paths here is terrible
        self.clsph_dir = "/Users/enjalot/code/sph/teach/sph/cl_src"
        self.clcommon_dir = "/Users/enjalot/code/sph/teach/sph/cl_common"
        
        self.dt = dt
        self.num = 256
        self.sph = sph

        self.loadData()
       
        self.hash = clhash.CLHash(self)
        self.radix = clradix.Radix(self, self.sph.max_num, 128, numpy.uint32(0))
        self.cellindices = clcellindices.CLCellIndices(self)
        self.density = cldensity.CLDensity(self)
         
       
    
    def acquire_gl(self):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
    def release_gl(self):
        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)


    def update(self):

        self.acquire_gl()

        self.hash.execute(      self.num,
                                self.position_u,
                                self.sort_hashes,
                                self.sort_indices,
                                self.gp,
                                self.clf_debug,
                                self.cli_debug
                            )

        self.radix.sort(    self.sph.max_num,
                            self.sort_hashes,
                            self.sort_indices
                        )

        negone = numpy.ones((self.sph.domain.nb_cells+1,), dtype=numpy.uint32)
        cl.enqueue_write_buffer(self.queue, self.ci_start, negone)

        self.cellindices.execute(   self.num,
                                    self.sort_hashes,
                                    self.sort_indices,
                                    self.ci_start,
                                    self.ci_end,
                                    self.gp,
                                    #self.clf_debug,
                                    #self.cli_debug
                                )

        self.density.execute(   self.num, 
                                self.position_s,
                                self.density_s,
                                self.ci_start,
                                self.ci_end,
                                self.sphp,
                                self.gp,
                                self.clf_debug,
                                self.cli_debug
                            )

        self.release_gl()
 


    def loadData(self):#, pos_vbo, col_vbo):
        import pyopencl as cl
        mf = cl.mem_flags
        
        #placeholder array used to fill cl buffers
        #could just specify size but might want some initial values later
        tmp = numpy.ndarray((self.sph.max_num, 4), dtype=numpy.float32)
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

        tmp_dens = numpy.ndarray((self.sph.max_num,), dtype=numpy.float32)
        self.density_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_dens)
        self.force_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.xsph_s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)

        tmp_uint = numpy.ndarray((self.sph.max_num,), dtype=numpy.uint32)
        self.sort_indices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_uint)
        self.sort_hashes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_uint)

        tmp_grid = numpy.ndarray((self.sph.domain.nb_cells+1, ), dtype=numpy.uint32)
        #grid size
        self.ci_start= cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_grid)
        self.ci_end = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_grid)

        #make struct buffers

        self.sphp_struct = self.sph.make_struct(self.num)
        self.sphp = cl.Buffer(self.ctx, mf.READ_ONLY, len(self.sphp_struct))
        cl.enqueue_write_buffer(self.queue, self.sphp, self.sphp_struct).wait()

        self.gp_struct = self.sph.domain.make_struct()
        self.gp = cl.Buffer(self.ctx, mf.READ_ONLY, len(self.gp_struct))

        #debug arrays
        tmp_int = numpy.ndarray((self.sph.max_num, 4), dtype=numpy.int32)
        self.clf_debug = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp)
        self.cli_debug = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmp_int)

        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.position_u, self.color_u]



 
    def reloadData(self):
        import pyopencl as cl
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
        cl.enqueue_copy_buffer(self.queue, self.pos_cl, self.pos_n1_cl).wait()
        cl.enqueue_copy_buffer(self.queue, self.pos_cl, self.pos_n2_cl).wait()
        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
        


 
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
        prg_name = prg_name.split("/")[-1]
        optionstr = "-I%s/ -I%s/" % (self.clsph_dir, self.clcommon_dir)
        #print optionstr

        plat = cl.get_platforms()[0]
        device = plat.get_devices()[0]

        #print prg_name
        self.prgs[prg_name] = cl.Program(self.ctx, fstr).build(options=optionstr)
        options = self.prgs[prg_name].get_build_info(device, cl.program_build_info.OPTIONS)
        #print "options: ", options
        #print "kernel", dir(self.prgs[prg_name])


    def render(self):


        #glColor3f(1,0,0)
        glEnable(GL_POINT_SMOOTH)
        glPointSize(5)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable(GL_DEPTH_TEST)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        """
        glColor3f(1., 0, 0)
        glBegin(GL_POINTS)
        for p in self.pos_vbo.data:
            glVertex3f(p[0], p[1], p[2])

        glEnd()
        """

        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)

        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glDrawArrays(GL_POINTS, 0, self.num*(self.ntracers+1))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        #glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

