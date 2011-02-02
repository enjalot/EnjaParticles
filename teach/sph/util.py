

#Vector class
#http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
#TODO: generalize to 3 and 4 dimensions

import numpy as np

class Vec(np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if len(obj) != 2:
            #not a 2 element vector!
            return None

        obj.x = obj[0]
        obj.y = obj[1]
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        #this gets called after object creation by numpy.ndarray
        #print "array finalize", obj
        if obj is None: return
        self.x = obj[0]
        self.y = obj[1]

    def __array_wrap__(self, out_arr, context=None):
        #this gets called after numpy functions are called on the array
        #out_arr is the output (resulting) array
        out_arr.x = out_arr[0]
        out_arr.y = out_arr[1]
        #print "array wrap:", out_arr
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __repr__(self):
        desc="""Vec2(data=%(data)s, x=%(x)s, y=%(y)s)"""
        return desc % {'data': str(self), 'x':self.x, 'y':self.y }


    def __setitem__(self, ind, val):
        if ind == 0:
            dict.__setattr__(self, 'x', val)
        elif ind == 1:
            dict.__setattr__(self, 'y', val)
        return np.ndarray.__setitem__(self, ind, val)
    
    def __setattr__(self, item, val):
        if item == "x":
            self[0] = val
        elif item == "y":
            self[1] = val


if __name__ == "__main__":
    v1 = Vec(np.arange(2))
    v2 = Vec(np.arange(2))
    print "v1:", repr(v1)
    print "v2:", repr(v2)
    v3 = v1 + v2
    print "v3:", repr(v3)
    v1[0] = 5
    print "v1.x, v1[0]:", v1.x, v1[0]
    v1.y = 6
    print "v1.y, v1[1]:", v1.y, v1[1]
    print "v1:", repr(v1)
