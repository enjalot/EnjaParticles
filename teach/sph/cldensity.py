import pyopencl as cl

class CLDensity:
    def __init__(self, clsph):
        self.clsph = clsph
        self.queue = self.clsph.queue
        self.dt = self.clsph.dt
        self.clsph.loadProgram(self.clsph.clsph_dir + "/density.cl")
    
    #@timings
    def execute(self, num, *args, **argv):
        if num > 0:
            global_size = (num,)
            local_size = (64,)

            self.clsph.prgs["density"].density_update(self.queue, global_size, local_size, *(args))

            self.queue.finish()
     
