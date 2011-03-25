import flock

print("*********  ALL METHODS FROM flock **********")
print(dir(flock))
print("*********  END METHODS FROM flock **********")

from flock import *

a = vecvec(10)
print(a[0])
a[0] = intvec(10)
print(a[0][0])

b = float4vec(10)
print(dir(b))
print(dir(b[0]))

b[0].x = 1.;
b[0].y = 2.;
b[0].z = 0.;

b[1].x = -1.;
b[1].y = 0.;
b[1].z = 0.5; 

print(b[0])
print(b[0].x)

print("test")
b[0] = b[1] / 5.
print(b[0].x)

b[0] += b[1]
c = 3. * b[1] * 5.
print(c)
print(b[0].x)
print((b[0] + b[1]).x)


boy = Boids(b)

print(dir(boy))
print(boy)
