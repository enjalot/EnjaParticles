
import numpy as np
from vector import Vec

def test_center(num, sphp, color1, color2, gradient=False):
    #setup a square in middle of color1 with everything else color2
    omin = Vec([0., 0., 0., 0.])
    omax = Vec([1., 1., 1., 0.])
    cmin = Vec([0.3, 0.3, 0.,0.])
    cmax = Vec([.7,.7,0.,0.])
    #rv1, col1 = addRect(num, cmin, cmax, sphp, color1)
    if gradient:
        color1g = [c*.5 for c in color1]
        print color1
        print color1g
        rv1, col1 = addRectGradient(num, cmin, cmax, sphp, color1, color1g)
    else:
        rv1, col1 = addRect(num, cmin, cmax, sphp, color1)

    #left
    lmin = Vec([ omin.x, omin.y, 0, 0])
    lmax = Vec([ cmin.x, omax.y, 0, 0])
    rv2, col2 = addRect(num, lmin, lmax, sphp, color2)
    
    #right
    rmin = Vec([ cmax.x, omin.y, 0, 0])
    rmax = Vec([ omax.x, omax.y, 0, 0])
    rv3, col3 = addRect(num, rmin, rmax, sphp, color2)
    
    #top
    tmin = Vec([ cmin.x, cmax.y, 0, 0])
    tmax = Vec([ cmax.x, omax.y, 0, 0])
    rv4, col4 = addRect(num, tmin, tmax, sphp, color2)

    #bottom
    bmin = Vec([ cmin.x, omin.y, 0, 0])
    bmax = Vec([ cmax.x, cmin.y, 0, 0])
    rv5, col5 = addRect(num, bmin, bmax, sphp, color2)

    rv = np.concatenate([rv1, rv2, rv3, rv4, rv5])
    col = np.concatenate([col1, col2, col3, col4, col5])
    return rv, col

#std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
def addRect(num, pmin, pmax, sphp, global_color):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addRect ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    spacing = .99 * sphp.rest_distance / sphp.sim_scale;
    print "spacing", spacing

    xmin = pmin.x# * scale
    xmax = pmax.x# * scale
    ymin = pmin.y# * scale
    ymax = pmax.y# * scale

    print "min, max", xmin, xmax, ymin, ymax
    rvec = []
    color = []
    i=0
    import copy
    for y in np.arange(ymin, ymax, spacing):
        gcolor = copy.copy(global_color)
        for x in np.arange(xmin, xmax, spacing):
            if i >= num: break
            #print "x, y", x, y
            #rvec += [ Vec([x,y]) * sphp.sim_scale];
            rvec += [[x, y, 0., 1.]]
            #gcolor[3] += x / (xmax - xmin) *.0001
            #gcolor[0] = gcolor[3]
            color += [gcolor]
            

            i+=1;
    print "%d particles added" % i
    rvecnp = np.array(rvec, dtype=np.float32)
    colornp = np.array(color, dtype=np.float32)
    return rvecnp, colornp



#std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
def addRectGradient(num, pmin, pmax, sphp, color1, color2):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addRect ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    spacing = .99 * sphp.rest_distance / sphp.sim_scale;
    print "spacing", spacing

    xmin = pmin.x# * scale
    xmax = pmax.x# * scale
    ymin = pmin.y# * scale
    ymax = pmax.y# * scale

    print "min, max", xmin, xmax, ymin, ymax
    rvec = []
    color = []
    i=0
    import copy
    for y in np.arange(ymin, ymax, spacing):
        #gcolor = copy.copy(color1)
        for x in np.arange(xmin, xmax, spacing):
            if i >= num: break
            #print "x, y", x, y
            #rvec += [ Vec([x,y]) * sphp.sim_scale];
            rvec += [[x, y, 0., 1.]]
            #gcolor[3] += x / (xmax - xmin) *.0001
            #gcolor[0] = gcolor[3]
            ratio = (x-xmin) / (xmax - xmin)
            #print ratio, gcolor, color2
            ccolor = [color1[j] * ratio + color2[j] * (1. - ratio) for j in range(4)]
            #print ccolor
            #gcolor = gcolor * x/xmax + color2 * (1. - x/xmax)
            color += [ccolor]
            

            i+=1;
    print "%d particles added" % i
    rvecnp = np.array(rvec, dtype=np.float32)
    colornp = np.array(color, dtype=np.float32)
    return rvecnp, colornp



#std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
def addRect(num, pmin, pmax, sphp, global_color, spacing_factor=.99):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addRect ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    #spacing_factor = 1.5
    spacing = spacing_factor * sphp.rest_distance / sphp.sim_scale;
    print "spacing", spacing

    xmin = pmin.x# * scale
    xmax = pmax.x# * scale
    ymin = pmin.y# * scale
    ymax = pmax.y# * scale

    print "min, max", xmin, xmax, ymin, ymax
    rvec = []
    color = []
    i=0
    import copy
    for y in np.arange(ymin, ymax, spacing):
        gcolor = copy.copy(global_color)
        for x in np.arange(xmin, xmax, spacing):
            if i >= num: break
            #print "x, y", x, y
            #rvec += [ Vec([x,y]) * sphp.sim_scale];
            rvec += [[x, y, 0., 1.]]
            #gcolor[3] += x / (xmax - xmin) *.0001
            #gcolor[0] = gcolor[3]
            color += [gcolor]
            

            i+=1;
    print "%d particles added" % i
    rvecnp = np.array(rvec, dtype=np.float32)
    colornp = np.array(color, dtype=np.float32)
    return rvecnp, colornp

def addPic(image, num, pmin, pmax, sphp):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addPic ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    #spacing = .99 * sphp.rest_distance / sphp.sim_scale;
    spacing = .70 * sphp.rest_distance / sphp.sim_scale;
    print "spacing", spacing

    xmin = pmin.x# * scale
    xmax = pmax.x# * scale
    ymin = pmin.y# * scale
    ymax = pmax.y# * scale

    print "min, max", xmin, xmax, ymin, ymax
    rvec = []
    color = []
    i=0

    ima = np.array(image)
    #ima.shape = (ima.shape[0] * ima.shape[1], )
    print "np array size", ima.size
    print "image size", image.size
    print ima
    print dir(image)
    print "COLORNP"
    #print image.layers


    yi = 0
    #this is all kinds of stupid. should really handle images better
    print image.size
    if len(image.size) == 2:
        ima = ima.T
    print image.size[0], image.size[1]

    for y in np.arange(ymax, ymin, -spacing):
        xi = 0
        if yi >= image.size[1]:
            break
        for x in np.arange(xmin, xmax, spacing):
            if i >= num: break
            xi += 1
            if xi >= image.size[0]:
                break

            #print "x, y", x, y
            #rvec += [ Vec([x,y]) * sphp.sim_scale];
            rvec += [[x, y, 0., 1.]]
            #print len(image.getpixel((xi,yi)))
            if len(image.size) == 2:
                #intensity = image.getpixel((xi, yi))/255.
                #g = ima.getpixl(xi, yi)/255.
                #b = ima[i*3+2]/255.
                mult = .1
                intensity = ima[xi, yi] / 255. * mult
                r = intensity
                g = intensity
                b = intensity
                a = intensity / mult
                #print ima[xi]
                #print xi, yi
                color += [[r, g, b, a]]
            else:
                color += [ ima[yi, xi] /255. * .4 ]
            """
            r = ima[x,y,0]/255. * .1
            g = ima[x,y,1]/255. * .1
            b = ima[x,y,2]/255. * .1
            a = ima[x,y,3]/255. * .1
            color += [[r, g, b, a]]
            """
            i+=1;
        yi += 1
    print "%d particles added" % i
    rvecnp = np.array(rvec, dtype=np.float32)
    #print color
    colornp = np.array(color, dtype=np.float32)

    return rvecnp, colornp;



def addRect3D(num, pmin, pmax, sphp):
    #Create a rectangle with at most num particles in it.  The size of the return
    #vector will be the actual number of particles used to fill the rectangle
    print "**** addRect ****"
    print "rest dist:", sphp.rest_distance
    print "sim_scale:", sphp.sim_scale
    spacing = 1.1 * sphp.rest_distance / sphp.sim_scale;
    print "spacing", spacing

    xmin = pmin.x# * scale
    xmax = pmax.x# * scale
    ymin = pmin.y# * scale
    ymax = pmax.y# * scale
    zmin = pmin.z
    zmax = pmax.z

    print "min, max", xmin, xmax, ymin, ymax, zmin, zmax
    rvec = [];
    i=0;
    for z in np.arange(zmin, zmax, spacing):
        for y in np.arange(ymin, ymax, spacing):
            for x in np.arange(xmin, xmax, spacing):
                if i >= num: break
                print "x, y", x, y, z
                #rvec += [ Vec([x,y]) * sphp.sim_scale];
                rvec += [[x, y, z, 1.]]
                i+=1;
    print "%d particles added" % i
    rvecnp = np.array(rvec, dtype=np.float32)
    return rvecnp;


  
