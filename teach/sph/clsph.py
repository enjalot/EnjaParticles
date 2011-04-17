#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush, glGenBuffers, glBindBuffer
from OpenGL.GL import *
from OpenGL.arrays import vbo

import numpy
import pyopencl as cl

import cldensity

class CLSPH:
    def __init__(self, max_num, dt):
        self.clinit()
        self.prgs = {}  #store our programs
        #of course hardcoding paths here is terrible
        self.clsph_dir = "/Users/enjalot/code/sph/teach/sph/cl_src"
        self.clcommon_dir = "/Users/enjalot/code/sph/teach/sph/cl_common"
        self.loadProgram(self.clsph_dir + "/density.cl")
        self.loadProgram(self.clsph_dir + "/leapfrog.cl")
        
        self.max_num = max_num
        self.dt = dt
        
        self.density = cldensity.CLDensity(self)
         
       
    
    def acquire_gl(self):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
    def release_gl(self):
        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)


    def update(self):

        self.density.execute(self.num, 1, 2, 3, 4, 5, 6, 7, 8)


    def loadData(self, pos_vbo, col_vbo):
        import pyopencl as cl
        mf = cl.mem_flags
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo

        self.pos = pos_vbo.data
        self.col = col_vbo.data

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()
        self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))

        self.col_vbo.bind()
        self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))

        #pure OpenCL arrays
        self.pos_n1_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pos)
        self.pos_n2_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pos)
        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.pos_cl, self.col_cl]
 
 
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
        optionstr = "-I%s/ -I%s/" % (self.clsph_dir, self.clcommon_dir)
        print optionstr

        plat = cl.get_platforms()[0]
        device = plat.get_devices()[0]

        self.prgs[prg_name] = cl.Program(self.ctx, fstr).build(options=optionstr)
        options = self.prgs[prg_name].get_build_info(device, cl.program_build_info.OPTIONS)
        print "options: ", options


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

