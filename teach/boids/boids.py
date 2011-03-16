#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush
from OpenGL.GL import *
from OpenGL.GLU import *

import pyopencl as cl

import numpy

class Boids(object):
    def __init__(self, num, dt, dim, *args, **kwargs):
        self.clinit()
        #clutil.CLKernel.__init__(self, *args, **kwargs)

        self.programs = {}

        self.loadProgram("boids.cl")
        self.loadProgram("euler.cl")

        self.num = num
        self.dt = numpy.float32(dt)
        self.dim = numpy.float32(dim)


    #TODO pass kernel args here instead of setting in loaddata
    def rules(self):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
        
        
        kernelargs = (self.pos_cl, 
                      self.col_cl, 
                      self.vel_cl, 
                      self.acc_cl, 
                      self.steer_cl,
                      self.avg_pos_cl, 
                      self.avg_vel_cl, 
                      self.dt,
                      self.dim)

    
        self.programs["boids"].rules1(self.queue, self.global_size, self.local_size, *(kernelargs))
        self.programs["boids"].rules2(self.queue, self.global_size, self.local_size, *(kernelargs))

        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
        self.queue.finish()
     
    #TODO pass kernel args here instead of setting in loaddata
    def euler(self):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
        
        
        kernelargs = (self.pos_cl, 
                      self.col_cl, 
                      self.vel_cl, 
                      self.dt,
                      self.dim)

    
        self.programs["euler"].euler(self.queue, self.global_size, self.local_size, *(kernelargs))

        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
        self.queue.finish()
        


    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print fstr
        #create the program
        kernname = filename.split(".")[0]
        self.programs[kernname] = cl.Program(self.ctx, fstr).build()


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



    def loadData(self, pos_vbo, col_vbo, vel, acc):
        import pyopencl as cl
        mf = cl.mem_flags
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo

        self.pos = pos_vbo.data
        self.col = col_vbo.data
        self.vel = vel

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()
        self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))
        self.col_vbo.bind()
        self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))

        #pure OpenCL arrays
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
        self.acc_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=acc)
        self.steer_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel) #these values not used, just same shape
        self.avg_pos_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel) #these values not used, just same shape
        self.avg_vel_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel) #these values not used, just same shape
        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.pos_cl, self.col_cl]
        
        self.global_size = (self.num,)
        self.local_size = None
        print self.global_size


    def render(self):
        
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
     

