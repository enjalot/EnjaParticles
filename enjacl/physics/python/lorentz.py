import numpy
import matplotlib

x = [1]
y = [0]
z = [0]

def FE(x, y, z, h):
    sigma = 10. 
    rho = 8./3.
    beta = 99.96
    
    #do for each particle
    i = 0
    xn = x[i]
    yn = y[i]
    zn = z[i]
    x[i] = xn + h*(sigma * (yn - xn))
    y[i] = yn + h*(xn*(rho - zn))
    z[i] = zn + h*(xn*yn - beta * zn)

h = .001
xp = []
yp = []
zp = []
for t in range(0, 1000):
    xp += [x[0]]
    yp += [y[0]]
    zp += [z[0]]
    FE(x, y, z, h)

#print xp
#print yp
#print zp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x,y,z, color="blue")


