#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush
from OpenGL.GL import *
from OpenGL.GLU import *


import clutil
import numpy

class Euler(clutil.CLKernel):
    def __init__(self, num, gl_objects, kernelargs, *args, **kwargs):
        #setup initial values of arrays
        clutil.CLKernel.__init__(self, *args, **kwargs)
        # set up the list of GL objects to share with opencl
        self.gl_objects = gl_objects
        
        # set up the Kernel argument list
        self.kernelargs = kernelargs

        self.global_size = (num,)
 

