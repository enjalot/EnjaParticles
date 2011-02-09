#ifndef _FORCE_CL_
#define _FORCE_CL_

	// TODO: NEED TO DEFINE MY SMALL FUNCTIONS SOMEWHERE (times, devides, normalize, distanceFrom)


	#define	separationdist	0.2f
	#define	searchradius	0.5f
	#define	maxspeed	0.3f
	#define MinUrgency      0.05f
	#define MaxUrgency      0.1f

	// index of closest flockmate
	int index_c = (int) density(index_i);

	// positions
	float4 pi = pos(index_i);
	float4 pj = pos(index_j);
	float4 pc = pos(index_c);

	// velocities
	float4 vi = force(index_i);
	float4 vj = force(index_j);
	float4 vc = force(index_c);

	// initialize acc to zero
        float4 acc = {0.f, 0.f, 0.f, 0.f};

        // Step 1. Update position
	// REMEMBER THAT MY VELOCITIES ARE GOING TO BE STORED IN THE FORCE VECTOR FROM NOW ON
//        pt->position += vi;

        // Step 2. Search for neighbors
        //numFlockmates = SearchFlockmates(i);

	// flockmates
	int closestFlockmate = index_c;
	int numFlockmates = index_c;	// TODO: is the y component but Im getting error: make sure that density is a float4

        // Step 3. Compute the three rules
	
	// RULE 1. Separation
	float d = distance(pi,pc);
	float r = d / separationdist;  				// TODO: NEED THE CLOSEST FLOCKMATE (ID OR POSITION), AND THE DISTANTCE FROM IT TO THE CURRENT BOID

        float4 separation = pc - pi;
        separation /= normalize(separation);			// TODO: search for normalization in OpenCL Specification

        if(d > separationdist){
               	separation *=  r;
        }
       	else if(closestFlockmate < separationdist){
               	separation *= -r;
        }
        else{
               	separation *= 0;
        }
        
	acc += separation;
                
	// RULE 2. Alignment
	float4 alignment = vc;
        alignment /= normalize(alignment);

        acc += alignment;

      	// RULE 3. Cohesion
      	float4 flockCenter= vel(index_i);
	float4 cohesion;

        flockCenter /= numFlockmates;
        cohesion = flockCenter - pi;
        cohesion /= normalize(cohesion);

	acc += cohesion;

        // Step 4. Constrain acceleration
        if(length(acc) > maxspeed*MaxUrgency){
                // set magnitude to MaxChangeInAcc
                acc *= (maxspeed*MaxUrgency);
        }

        // Step 5. Add acceleration to velocity
        pt->force += acc;

    	// Step 6. Constrain velocity
        if(length(vi) > maxspeed){
                // set magnitude to MaxSpeed
                pt->force *= maxspeed;
        }

#endif
