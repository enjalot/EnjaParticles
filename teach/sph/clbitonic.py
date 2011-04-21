#http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy as np
import struct
import util
timings = util.timings

#ctx = cl.create_some_context()
mf = cl.mem_flags

class Bitonic:
    def __init__(self, clsph, max_elements, cta_size, dtype):

        self.clsph = clsph
        self.ctx = self.clsph.ctx
        self.queue = self.clsph.queue
        self.dt = self.clsph.dt
 
        self.local_size_limit = 256
        options = "-D LOCAL_SIZE_LIMIT=%d" % (self.local_size_limit,)
        self.clsph.loadProgram(self.clsph.clcommon_dir + "/bitonic.cl", options)

        self.uintsz = dtype.itemsize
        self.d_tempKeys = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.uintsz * max_elements)
        self.d_tempValues = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.uintsz * max_elements)





    def factorRadix2(self, L):
        if(not L):
            log2L = 0
            return log2L, 0
        else:
            #for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
            log2L = 0
            while ((L & 1) == 0):
                L >>=1
                log2L += 1
                
            return log2L, L;



    def sort(self, num, keys, values, batch=1, dir=1):
        
        print "bitonic sort"
        #num must be a power of 2 and <= max_num
        log2l, remainder = self.factorRadix2(num)
        if remainder != 1:
            return

        self.keys = keys
        self.values = values

        dir = (dir != 0)
        array_length = self.keys.size
        print "array_length", array_length

        if array_length < self.local_size_limit:
            self.local(array_length, dir)
        else:
            print "local1"
            self.local1(batch, array_length, dir)
            size = 2 * self.local_size_limit
            while size < array_length:
                stride = size / 2
                while stride > 0:
                    if stride >= self.local_size_limit:
                        #self.merge_global(batch, array_length, size, stride, dir)
                        pass
                    else:
                        #self.merge_local(batch, array_length, size, stride, dir)
                        pass


                    stride >>= 1
                size <<= 1

        print "do we get here?"
        self.queue.finish()
        #need to copy back
        print "copying buffers"
        cl.enqueue_copy_buffer(self.queue, self.keys, self.d_tempKeys).wait()
        cl.enqueue_copy_buffer(self.queue, self.values, self.d_tempValues).wait()
        self.queue.finish()



    @timings("Bitonic: merge local")
    def merge_local(self, batch, array_length, size, stride, dir):
        local_size = (self.local_size_limit / 2,)
        global_size = (batch * array_length / 2,)
        merge_local_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.keys,
                        self.values,
                        np.int32(array_length),
                        np.int32(stride),
                        np.int32(size),
                        np.int32(dir)
                    )

        self.clsph.prgs["bitonic"].bitonicMergeLocal(self.queue, global_size, local_size, *(merge_local_args)).wait()
        self.queue.finish()



    @timings("Bitonic: merge global")
    def merge_global(self, batch, array_length, stride, size, dir):
        local_size = None
        global_size = (batch * array_length / 2,)
        merge_global_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.keys,
                        self.values,
                        np.int32(array_length),
                        np.int32(size),
                        np.int32(stride),
                        np.int32(dir)
                    )

        self.clsph.prgs["bitonic"].bitonicMergeGlobal(self.queue, global_size, local_size, *(merge_global_args)).wait()
        self.queue.finish()



    @timings("Bitonic: local1 ")
    def local1(self, batch, array_length, dir):
        local_size = (self.local_size_limit / 2,)
        global_size = (batch * array_length / 2,)
        print global_size, local_size
        print "local1 args"
        local1_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.keys,
                        self.values
                    )
        print "local1 sort"
        self.clsph.prgs["bitonic"].bitonicSortLocal1(self.clsph.queue, global_size, local_size, *(local1_args)).wait()
        print "queue finish"
        self.queue.finish()


    @timings("Bitonic: local ")
    def local(self, array_length, dir):
        local_size = (self.local_size_limit / 2,)
        global_size = (batch * array_length / 2,)
        local_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.keys,
                        self.values,
                        np.int32(array_length),
                        np.int32(dir)
                    )

        self.clsph.prgs["bitonic"].bitonicSortLocal(self.queue, global_size, local_size, *(local_args)).wait()
        self.queue.finish()



if __name__ == "__main__":
    #These tests wont work as is since class was restructured to fit in with sph

    #n = 1048576
    #n = 32768*2
    #n = 16384
    n = 8192
    hashes = np.ndarray((n,1), dtype=np.uint32)
    indices = np.ndarray((n,1), dtype=np.uint32)
    
    for i in xrange(0,n): 
        hashes[i] = n - i
        indices[i] = i
    
    npsorted = np.sort(hashes,0)

    print "hashes before:", hashes[0:20].T
    print "indices before: ", indices[0:20].T 

    from hash import Domain
    from vector import Vec
    import cldiffuse_system
    import sph
    dmin = Vec([0.,0.,0.])
    dmax = Vec([1.,1.,1.])
    print "SPH System"
    print "-------------------------------------------------------------"
    domain = Domain(dmin, dmax)
    system = sph.SPH(n, domain)

    dt = .001
    clsystem = cldiffuse_system.CLDiffuseSystem(system, dt, ghost_system=None)
    bitonic = Bitonic(clsystem, n, 128, hashes.dtype)
    #num_to_sort = 32768
    num_to_sort = n
    bitonic.sort(num_to_sort, hashes, indices)
    #read from buffer
    """
    hashes = numpy.ndarray((num_to_sort,), dtype=numpy.int32)
    cl.enqueue_read_buffer(clsystem.queue, .sort_hashes, hashes)
    print "hashes"
    print hashes.T
    indices = numpy.ndarray((self.num,), dtype=numpy.int32)
    cl.enqueue_read_buffer(self.queue, self.sort_indices, indices)
    print "indices"
    print indices.T
    """


    print "hashes after:", hashes[0:20].T
    print "indices after: ", indices[0:20].T 

    print np.linalg.norm(hashes - npsorted)







