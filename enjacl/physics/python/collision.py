import numpy
from math import *

pi = numpy.pi

def normalize(v):
    #not pythonic at all, but want to be close to opencl implementation
    magnitude = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    vn = [v[0]/magnitude, v[1]/magnitude, v[2]/magnitude]
    return vn

"""
a = [1,1,0]
print a
b = normalize(a)
print b
"""

R = numpy.zeros((3,3))
#rotate by 45 degrees
#R[0][0] = cos(pi/4.)
#R[0][1] = -sin(pi/4.)
#R[1][0]

xn = -sin(pi/4.)
yn = cos(pi/4.)
v = [xn, yn, 0]
print v
y = normalize(v)
print y
#print xn, yn

