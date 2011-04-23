#http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy as np
import struct
import timing
timings = timing.Timing()

#ctx = cl.create_some_context()
mf = cl.mem_flags

class Bitonic:
    def __init__(self, clsph, max_elements, cta_size, dtype):


        self.clsph = clsph
        self.ctx = self.clsph.ctx
        self.queue = self.clsph.queue

        self.loadProgram()
        self.uintsz = dtype.itemsize
        self.d_tempKeys = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.uintsz * max_elements)
        self.d_tempValues = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.uintsz * max_elements)



    def loadProgram(self):
        self.local_size_limit = 512
        options = "-D LOCAL_SIZE_LIMIT=%d" % (self.local_size_limit,)

        print "build bitonic"
        f = open(self.clsph.clcommon_dir + "/bitonic.cl", 'r')
        fstr = "".join(f.readlines())
        self.bitonic_prg = cl.Program(self.ctx, fstr).build(options=options)




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



    @timings("Bitonic Sort")
    def sort(self, num, keys, values, batch=1, direction=1):
        
        #num must be a power of 2 and <= max_num
        log2l, remainder = self.factorRadix2(num)
        if remainder != 1:
            return

        self.keys = keys
        self.values = values

        direction = (direction != 0)
        array_length = self.clsph.system.max_num
        #print "array_length", array_length
        #print "local size limit", self.local_size_limit

        if array_length < self.local_size_limit:
            self.local(array_length, direction)
        else:
            self.local1(batch, array_length, direction)
            size = 2 * self.local_size_limit
            while size <= array_length:
                stride = size / 2
                while stride > 0:
                    #print "size, stride", size, stride
                    if stride >= self.local_size_limit:
                        self.merge_global(batch, array_length, stride, size, direction)
                    else:
                        self.merge_local(batch, array_length, size, stride, direction)
                        break


                    stride >>= 1
                size <<= 1

        self.queue.finish()
        #need to copy back

        cl.enqueue_copy_buffer(self.queue, self.d_tempKeys, self.keys).wait()
        cl.enqueue_copy_buffer(self.queue, self.d_tempValues, self.values).wait()
        self.queue.finish()

        """
        print "!!!!!!!!!!!!!!!!!!"
        keys_np = np.ndarray((self.clsph.system.max_num,), dtype=np.uint32)
        values_np = np.ndarray((self.clsph.system.max_num,), dtype=np.uint32)
        cl.enqueue_read_buffer(self.queue, self.d_tempKeys, keys_np).wait()
        cl.enqueue_read_buffer(self.queue, self.d_tempValues, values_np).wait()
        print "TEMP KEYS IN BITONIC)"
        print keys_np[0:35].T
        print "TEMP VALUES IN BITONIC"
        print values_np[0:35].T
        """
        #return keys, values




    @timings("Bitonic: merge global")
    def merge_global(self, batch, array_length, stride, size, direction):
        local_size = None
        global_size = (batch * array_length / 2,)
        merge_global_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.d_tempKeys,
                        self.d_tempValues,
                        np.int32(array_length),
                        np.int32(size),
                        np.int32(stride),
                        np.int32(direction)
                    )

        self.bitonic_prg.bitonicMergeGlobal(self.queue, global_size, local_size, *(merge_global_args))
        self.queue.finish()


    @timings("Bitonic: merge local")
    def merge_local(self, batch, array_length, size, stride, direction):
        local_size = (self.local_size_limit / 2,)
        global_size = (batch * array_length / 2,)
        merge_local_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.d_tempKeys,
                        self.d_tempValues,
                        np.int32(array_length),
                        np.int32(stride),
                        np.int32(size),
                        np.int32(direction)
                    )

        self.bitonic_prg.bitonicMergeLocal(self.queue, global_size, local_size, *(merge_local_args))
        self.queue.finish()




    @timings("Bitonic: local1 ")
    def local1(self, batch, array_length, direction):
        local_size = (self.local_size_limit / 2,)
        global_size = (batch * array_length / 2,)
        #print global_size, local_size
        local1_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.keys,
                        self.values
                    )
        self.bitonic_prg.bitonicSortLocal1(self.queue, global_size, local_size, *(local1_args))
        self.queue.finish()


    @timings("Bitonic: local ")
    def local(self, array_length, direction):
        local_size = (self.local_size_limit / 2,)
        global_size = (batch * array_length / 2,)
        local_args = (
                        self.d_tempKeys,
                        self.d_tempValues,
                        self.keys,
                        self.values,
                        np.int32(array_length),
                        np.int32(direction)
                    )

        self.bitonic_prg.bitonicSortLocal(self.queue, global_size, local_size, *(local_args))
        self.queue.finish()



if __name__ == "__main__":
    #These tests wont work as is since class was restructured to fit in with sph

    n = 1048576
    #n = 32768*2
    #n = 16384
    #n = 8192
    hashes = np.ndarray((n,1), dtype=np.uint32)
    indices = np.ndarray((n,1), dtype=np.uint32)
    print "hashes size", hashes.size
    
    for i in xrange(0,n): 
        hashes[i] = n - i
        indices[i] = i
    
    npsorted = np.sort(hashes,0)

    print "hashes before:", hashes[0:20].T
    print "indices before: ", indices[0:20].T 

    bitonic = Bitonic(n, 128, hashes.dtype)
    #num_to_sort = 32768
    num_to_sort = n
    shashes, sindices = bitonic.sort(num_to_sort, hashes, indices)


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


    print "hashes after:", shashes[0:20].T
    print "sorted hashes:", npsorted[0:20].T
    print "indices after: ", sindices[0:20].T 

    print np.linalg.norm(shashes - npsorted)

    print timings






