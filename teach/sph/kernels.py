from vector import Vec

from math import *
import numpy as np
from numpy import pi

def mag(x):
    return sqrt(np.dot(x, x))

def Wpoly6(h, r):
    coeff = 315./(64*pi*h**9)
    #magr = abs(r)
    magr = mag(r)
    print magr

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
    if magr == 0:
        magr = 1E-6
    coeff = -45./(magr*pi*h**6)
    if magr < h:
        return coeff*(h - magr)**2
    return 0

def main():
    import numpy as np
    import pylab

    h = 1

    X = np.linspace(-h, h, 100)
    Y = []
    dY = []
    for x in X:
        #Y += [Wpoly6(h, [0,x])]
        Y += [Wspiky(h,[0, x])]
        dY += [dWspiky(h, [0,x])]

    #pylab.plot(X,Y)
    pylab.plot(X,dY)
    pylab.show()


if __name__ == "__main__":
    main()


