from vector import Vec

from math import *
import numpy as np
from numpy import pi

def mag(x):
    return sqrt(np.dot(x, x))
def mag2(x):
    return np.dot(x,x)


class Kernel(object):
    def __init__(self, h):
        self.h = h
        h6 = h**6
        h9 = h6*h*h*h
        self.coeffs = {}
        self.coeffs["poly6"] = 315./(64.*pi*h9)
        self.coeffs["dspiky"] = -45./(pi*h6)
        self.coeffs["ddvisc"] = 45./(pi*h6)

    def poly6(self, r):
        mr2 = mag2(r)
        if mr2 < self.h*self.h:
            return self.coeffs["poly6"] * (self.h**2 - mr2)**3
        else:
            return 0.
    
    def dspiky(self, r):
        magr = mag(r)
        if magr == 0:
            magr = 1E-6

        hr2 = self.h - magr

        if magr < self.h:
            return self.coeffs["dspiky"] * hr2 * hr2 / magr
        return 0

    def ddvisc(self, r):
        magr = mag(r)
        if magr == 0:
            magr = 1E-6

        hr = self.h - magr

        if magr < self.h:
            return self.coeffs["ddvisc"] * hr
        return 0



def Wpoly6(h, r):
    coeff = 315./(64.*pi*h**9)
    #magr = abs(r)
    magr = mag(r)

    if magr < h:
        return coeff*(h**2 - magr**2)**3
    return 0

def Wspiky(h, r):
    coeff = 15./(pi*h**6)
    #magr = abs(r)
    magr = mag(r)

    if magr < h:
        return coeff*(h-magr)**3
    return 0

def dWspiky(h, r):
    """ still need to multiply this quantity by r """
    #magr = abs(r)
    magr = mag(r)
    #print "magr", magr
    if magr == 0:
        magr = 1E-6

    den = magr * pi * h**6
    coeff = -45./den
    hr2 = h - magr

    if magr < h:
        return coeff*hr2*hr2
    return 0

def main():
    import numpy as np
    import pylab

    h = 1
    k = Kernel(h)

    X = np.linspace(-h, h, 100)
    Y = []
    dY = []
    for x in X:
        #Y += [Wpoly6(h, [0,x])]
        Y += [Wspiky(h,[0, x])]
        #dY += [dWspiky(h, [0,x])]
        #dY += [k.dspiky([0,x])]
        dY += [k.ddvisc([0,x])]

    #pylab.plot(X,Y)
    pylab.plot(X,dY)
    pylab.show()


if __name__ == "__main__":
    main()


