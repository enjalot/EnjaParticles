import numpy
import matplotlib

x = [1]
y = [0]
z = [0]

def FE(x, y, z, h):
    sigma = 10. 
    beta = 8./3.
    rho = 99.96
    
    #do for each particle
    i = 0
    xn = x[i]
    yn = y[i]
    zn = z[i]
    x[i] = xn + h*(sigma * (yn - xn))
    y[i] = yn + h*(xn*(rho - zn))
    z[i] = zn + h*(xn*yn - beta * zn)

print "solving"

h = .001
n = 10000
xp = numpy.zeros((n,1))
yp = numpy.zeros((n,1))
zp = numpy.zeros((n,1))
for i in range(0, n):
    xp[i] = x[0]
    yp[i] = y[0]
    zp[i] = z[0]
    FE(x, y, z, h)

print "about to plot"
#print xp
#print yp
#print zp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(xp,yp,zp, color="blue")
plt.show()
