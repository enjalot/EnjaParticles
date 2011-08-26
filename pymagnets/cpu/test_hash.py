from vector import Vec
from hash import Domain
import sph


def print_hash(domain, v):
    print "position", v
    gp = domain.calc_cell(v)
    print "grid cell", gp
    print "grid hash", domain.calc_hash(gp)
 

if __name__ == "__main__":

    #doing some testing on hashing out of bounds
    dmin = Vec([0.,0.,0.])
    dmax = Vec([5.,5.,5.])
    domain = Domain(dmin, dmax, None)

    max_num = 8192 
    system = sph.SPH(max_num, domain)

    dmin_s = dmin * system.sim_scale
    dmax_s = dmax * system.sim_scale
    domain_s = Domain(dmin_s, dmax_s, None)
    domain_s.setup(system.smoothing_radius)

    #print domain_s
    print domain

    print "=================="

    v = Vec([1., 1., 1.])
    print_hash(domain, v)
   
    v = Vec([4.9, 4.9, 4.9])
    print_hash(domain, v)

    v = Vec([4.9999, 4.9999, 4.9999])
    print_hash(domain, v)

    print "out of positive bounds"
    v = Vec([6., 6., 6.])
    print_hash(domain, v)

    v = Vec([6.5, 6.5, 6.5])
    print_hash(domain, v)

    v = Vec([7., 7., 7.])
    print_hash(domain, v)

    print "out of negative bounds"
    #these are not accurate because hash should be unsigned int
    v = Vec([-1., -1., -1.])
    print_hash(domain, v)

    v = Vec([-2., -2., -2.])
    print_hash(domain, v)





