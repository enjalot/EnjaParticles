import pyopencl as cl
import numpy

class CLHash:
    def __init__(self, clsph):
        self.clsph = clsph
        self.queue = self.clsph.queue
        self.dt = self.clsph.dt
        self.clsph.loadProgram(self.clsph.clcommon_dir + "/hash.cl")

    
    #@timings
    def execute(self, num, *args, **argv):
        if num > 0:
            worksize = 128
            factor = 1.*num / worksize
            if int(factor) != factor:
                factor = int(factor)
                global_size = (worksize * factor + worksize,)
            else:
                global_size = (num,)
            local_size = (worksize,)
            num = numpy.int32(num)

            self.clsph.prgs["hash"].hash(self.queue, global_size, local_size, num, *(args))

            self.queue.finish()
 
