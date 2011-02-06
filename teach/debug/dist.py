import re

pos = []

f = open('points.txt', 'r')
for line in f.readlines():
    pse = line.split("=")
    m = re.search('[0-9]+', pse[0])
    ind = m.group(0)

    pts = pse[1].split(" ")
    x = float(pts[1])
    y = float(pts[2])
    z = float(pts[3])
    pos.append( [x,y,z] )

import math
def distance(pi, pj):
    return math.sqrt( (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2 + (pi[2] - pj[2])**2)

for i,pi in enumerate(pos):
    for j,pj in enumerate(pos):
        if i == j: continue
        d = distance(pi, pj)
        print d
        if d == 0:
            print "OMG %d %d are TWINS" % ( i, j)
