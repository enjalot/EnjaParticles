
# author: Ian Johnson
# co-author: Gordon Erlebacher

""" improve on Ian's module by using properties """

#Vector class
#http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
#TODO: generalize to 3 and 4 dimensions

import numpy as np

class Vec(np.ndarray):

    def __new__(cls, input_array=[0,1]):
        if len(input_array) != 2:
            return None
        obj = np.ndarray.__new__(cls, 2)
        return obj

    def __init__(self, input_array=[0,1]):
        self[0] = input_array[0]
        self[1] = input_array[1]
        self._x, self._y = self[0], self[1]

    def __repr__(self):
        desc="""Vec2(data=%(data)s, x=%(x)s, y=%(y)s)"""
        return desc % {'data': str(self), 'x':self.x, 'y':self.y }


    # properties only work for classes that derive from object, 
    # such as ndarray
    def getx(self): return self._x 
    def gety(self): return self._y 
    def setx(self, _x): self._x = _x  
    def sety(self, _y): self._y = _y  

    x = property(getx, setx)
    y = property(gety, sety)

	# works with Finalize
	# NOT SURE I UNDERSTAND
    def __array_finalize__(self, obj):
        #this gets called after object creation by numpy.ndarray
        print "array finalize", obj
        if obj is None: return
        self._x = obj[0]
        self._y = obj[1]


if __name__ == "__main__":
    v1 = Vec(np.arange(2))
    v2 = Vec(np.arange(2))
    v1 = Vec([1,2])
    v2 = Vec([3,4])
    #print dir(v1)
    #print v1[0]
    print v1.x, v1.y, v1[0], v1[1]
    print v2.x, v2.y, v2[0], v2[1]
    print v1, v2
    v1.x = 3
    v1.y = 5
    print v1, v2
    v1 = v1 + v2
    print v1.x, v1.y  # only works with array_finalize
    print v1, v2
    v2.x = 3
    print v2.x, v2.y
    print repr(v1)
    print str(v1)
    print v1.x, v1.y
    print "v1:", repr(v1), str(v1), v1, v1.x, v1.y
    print "v2:", repr(v2), str(v2), v2
    v3 = v1 + v2
    print "v3:", v3
    print "v3:", repr(v3)
    v1[0] = 5
    print "v1.x, v1[0]:", v1.x, v1[0]
    v1.y = 6
    print "v1.y, v1[1]:", v1.y, v1[1]
    print "v1:", repr(v1)

