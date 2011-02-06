from vector import Vec
from domain import Domain


class SPH:
    def __init__(self, max_num, domain):
        self.max_num = max_num

        p0 = 1000.                  #rest density [ kg/m^3 ]
        V0 = 0.000005547            #initial volume [ m^3 ]
        n = 27                      #number of particles to occupy V0
        VP = V0 / n                 #particle volume [ m^3 ]
        m = p0 * VP                 #particle mass [ kg ]
        VF = VP * max_num           #fluid volume [ m^3 ]
        re = (m/p0)**(1/3.)         #particle radius [ m ]
        rest_distance = .87 * re    #rest distance between particles [ m ]
        
        #the ratio between the particle radius in simulation space and world space
        sim_scale = (VF / domain.V)**(1/3.)

        self.p0 = p0
        self.V0 = V0
        self.n = n
        self.m = m
        self.VF = VF
        self.re = re
        self.rest_distance = rest_distance
        self.sim_scale = sim_scale

        
        print "particle mass:", self.m
        print "Fluid Volume VF:", self.VF
        print "simulation scale:", self.sim_scale


if __name__ == "__main__":
    dmin = Vec([0,0,0])
    dmax = Vec([5,5,5])
    domain = Domain(dmin, dmax)
    system = SPH(2**14, domain)     #16384
    #system = SPH(2**12, domain)     #4096

