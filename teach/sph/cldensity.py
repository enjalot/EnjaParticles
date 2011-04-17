import pyopencl as cl

class CLDensity:
    def __init__(self, clsph):
        self.clsph = clsph
        self.queue = self.clsph.queue
        self.dt = self.clsph.dt

    
    #@timings
    """
    def execute(self,
                pos_s,
                density_s,
                ci_start,
                ci_end,
                sphp,
                gp,
                clf_debug,
                cli_debug
                ):
    """
    def execute(self, num, *args, **argv):

        print "args"
        print args

        global_size = (num,)
        local_size = None

        self.prgs["density"].density(self.queue, global_size, local_size, *(args))

        self.queue.finish()
 
