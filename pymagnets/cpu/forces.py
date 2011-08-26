import pygame
from pygame.locals import *

from kernels import *
from vector import Vec

#from timing import print_timing
from timing import Timing
timings = Timing()


@timings
def contact_update(system, particles):
    for pi in particles:
        for pj in particles:
            if pj == pi:
                continue
            
            #get distance between particles
            r = pi.pos - pj.pos
            magr = mag(r)
            if magr > 2*pi.r: continue
            
            repulsion = system.repulse * r / magr**3
            pi.force = Vec([0., 0.])
            pi.vel = repulsion #Vec([0, 0])
            #pi.force = repulsion
            #pi.angular /= 100.
            #print repulsion



@timings
def magnet_update(system, particles):
    for pi in particles:
        pi.force = Vec([0.,0.])
        pi.angular = Vec([0., 0.])

        for pj in particles:
            if pj == pi:
                continue
            
            #get distance between particles
            r = pi.pos - pj.pos
            magr = mag(r)
            if magr > pi.h: continue

            #calculate north 
            pinp = pi.pos + pi.np
            pjnp = pj.pos + pj.np
            rnn = pinp - pjnp
            fnn = system.tra * rnn / mag(rnn)**3 

            pinp = pi.pos + pi.np
            pjsp = pj.pos + pj.sp
            rns = pinp - pjsp
            fns = system.tra * rns / mag(rns)**3 

            Fn = -fns + fnn

            #calculate south
            pisp = pi.pos + pi.sp
            pjnp = pj.pos + pj.np
            rsn = pisp - pjnp
            fsn = system.tra * rsn / mag(rsn)**3 

            pisp = pi.pos + pi.sp
            pjsp = pj.pos + pj.sp
            rss = pisp - pjsp
            fss = system.tra * rss / mag(rss)**3 

            Fs = fss - fsn

            #translation force
            Ft = np.dot(Fn, pi.np) / mag(pi.np) + np.dot(Fs, pi.sp) / mag(pi.sp)
            #pi.force += Ft
            pi.force += Fn + Fs
            #print Ft

            #angular accel (torque) / (moment of inertia)
            Fr = (Fn + Fs - Ft) / (2 * pi.mass * pi.r**2 / 5.)
            #print Fr
            pi.angular += Fr




@timings
def rk4_update(system, particles, dt):
    for pi in particles:
        if pi.lock:
            continue

        xp = np.cross(pi.angular, pi.np)
        if xp > 0:
            theta = mag(pi.angular) * dt * system.rot
        else:
            theta = -mag(pi.angular) * dt * system.rot
        #theta = 0.
        #rotate north pole
        npx = pi.np[0]*cos(theta) - pi.np[1]*sin(theta)
        npy = pi.np[0]*sin(theta) + pi.np[1]*cos(theta)
        #rotate south pole
        spx = pi.sp[0]*cos(theta) - pi.sp[1]*sin(theta)
        spy = pi.sp[0]*sin(theta) + pi.sp[1]*cos(theta)

        pi.np = Vec([npx, npy])
        pi.sp = Vec([spx, spy])

        f = pi.force / pi.mass

        pi.vel += f * dt
        pi.pos += pi.vel * dt






@timings
def euler_update(system, particles, dt):
    i = 0
    for pi in particles:
        if pi.lock:
            continue

        i += 1

        xp = np.cross(pi.angular, pi.np)
        if xp > 0:
            theta = mag(pi.angular) * dt * system.rot
        else:
            theta = -mag(pi.angular) * dt * system.rot
        #theta = 0.
        #rotate north pole
        npx = pi.np[0]*cos(theta) - pi.np[1]*sin(theta)
        npy = pi.np[0]*sin(theta) + pi.np[1]*cos(theta)
        #rotate south pole
        spx = pi.sp[0]*cos(theta) - pi.sp[1]*sin(theta)
        spy = pi.sp[0]*sin(theta) + pi.sp[1]*cos(theta)

        pi.np = Vec([npx, npy])
        pi.sp = Vec([spx, spy])

        f = pi.force

        #f.y += -9.8

        #speed = mag(f);
        #print "speed", speed
        #if speed > system.velocity_limit:
        #    f *= system.velocity_limit/speed;

        #print "force", f
        pi.vel += f * dt
        pi.pos += pi.vel * dt







#@print_timing
@timings
def density_update(sphp, particles):
    #brute force
    for pi in particles:
        pi.dens = 0.
        for pj in particles:
            #print pi.pos, pj.pos
            r = pi.pos - pj.pos
            #print r
            if mag(r) > pi.h: continue
            #pi.dens += pj.mass*Wpoly6(pi.h, r)
            pi.dens += pj.mass * sphp.kernels.poly6(r)

#@print_timing
@timings
def force_update(sphp, particles):
    #brute force
    rho0 = sphp.rho0
    K = sphp.K

    for pi in particles:
        pi.force = Vec([0.,0.])
        pi.xsph = Vec([0.,0.])
        di = pi.dens
        #print "di", di
        Pi = K*(di - rho0)
        #print "Pi", Pi
        #print "pi.h", pi.h
        for pj in particles:
            if pj == pi:
                continue
            r = pi.pos - pj.pos
            #temporary optimization until we do efficient neighbor search
            if mag(r) > pi.h: continue
            #print "r", r

            dj = pj.dens
            Pj = K*(dj - rho0)

            #print "dWspiky", dWspiky(pi.h, r)
            #kern = .5 * (Pi + Pj) * dWspiky(pi.h, r)
            kern = .5 * (Pi + Pj) * sphp.kernels.dspiky(r)
            f = r*kern
            #does not account for variable mass
            f *= pi.mass / (di * dj)

            #print "force", f
            pi.force += f

            #XSPH
            #float4 xsph = (2.f * sphp->mass * Wijpol6 * (velj-veli)/(di.x+dj.x));
            xsph = pj.veleval - pi.veleval
            xsph *= 2. * pi.mass * sphp.kernels.poly6(r) / (di + dj) 
            pi.xsph += xsph


        #print "pi.force", pi.force
  


#@timings
def calcRepulsionForce(normal, vel, sphp, distance):
    repulsion_force = (sphp.boundary_stiffness * distance - sphp.boundary_dampening * np.dot(normal, vel))*normal;
    return repulsion_force;

#@timings
def calcFrictionForce(v, f, normal, friction_kinetic, friction_static_limit):
    pass

#@print_timing
@timings
def collision_wall(sphp, domain, particles):
    
    dmin = domain.bnd_min * sphp.sim_scale
    dmax = domain.bnd_max * sphp.sim_scale
    bnd_dist = sphp.boundary_distance
    #float diff = params->boundary_distance - (p.z - gp->bnd_min.z);
    #if (diff > params->EPSILON)
    #r_f += calculateRepulsionForce(normal, v, params->boundary_stiffness, params->boundary_dampening, diff);
    #f_f += calculateFrictionForce(v, f, normal, friction_kinetic, friction_static_limit);
   

    for pi in particles:
        f = Vec([0.,0.])
        p = pi.pos
        #X Min
        diff = bnd_dist - (p.x - dmin.x)
        if diff > 0:
            normal = Vec([1.,0.])
            f += calcRepulsionForce(normal, pi.vel, sphp, diff)
            #calcFrictionForce(pi.v, pi.f, normal, friction_kinetic, friction_static_limit)
        #X Max
        diff = bnd_dist - (dmax.x - p.x)
        if diff > 0:
            normal = Vec([-1.,0.])
            f += calcRepulsionForce(normal, pi.vel, sphp, diff)
            #calcFrictionForce(pi.v, pi.f, normal, friction_kinetic, friction_static_limit)
        #Y Min
        diff = bnd_dist - (p.y - dmin.y)
        if diff > 0:
            normal = Vec([0.,1.])
            f += calcRepulsionForce(normal, pi.vel, sphp, diff)
            #calcFrictionForce(pi.v, pi.f, normal, friction_kinetic, friction_static_limit)
        #Y Max
        diff = bnd_dist - (dmax.y - p.y)
        if diff > 0:
            normal = Vec([0.,-1.])
            f += calcRepulsionForce(normal, pi.vel, sphp, diff)
            #calcFrictionForce(pi.v, pi.f, normal, friction_kinetic, friction_static_limit)

        #print "collision force", f
        pi.force += f



#@print_timing
@timings
def leapfrog_update(sphp, particles):
    dt = .001

    #print "LEAPFROG++++++++++++++++++++++"
    for pi in particles:
        f = pi.force
        #print "f", f

        f.y += -9.8

        speed = mag(f);
        #print "speed", speed
        if speed > sphp.velocity_limit:
            f *= sphp.velocity_limit/speed;
    
        #print "force", f
        vnext = pi.vel + f * dt
        vnext += sphp.xsph_factor * pi.xsph
        pi.pos += vnext * dt

        veval = .5 * (pi.vel + vnext)
        pi.vel = vnext
        pi.veleval = veval



