#include <stdio.h>
#include "../Swarm.h"

namespace rtps {
// float4 times constant
float4 times (float4 v, float a)
{
    v.x *= a;
    v.y *= a;
    v.z *= a;
//    v.w *= a;
    
    return v;
}

// float4 divided constant
float4 divides(float4 v, float a)
{
    v.x /= a;
    v.y /= a;
    v.z /= a;
//    v.w /= a;

    return v;
}

// normalize a vector
float4 normalize(float4 v){
    float n = magnitude(v);
 
    v.x /= n;
    v.y /= n;
    v.z /= n;
//    v.w /= n;

    return v;
}

// distance
float distanceFrom(float4 p1, float4 p2)
{
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

// constants
#define	MaxSpeed	0.5f
#define MinUrgency	0.05f
#define	MaxUrgency	0.1f
#define MaxChangeInAcc	(MaxSpeed * MaxUrgency)

#define	DesireSpeed	(MaxSpeed/2)
#define	SeparationDist	.3f
#define SearchRadius	.5f


//****************************************************************
// Search for flockmates
int Swarm::SearchFlockmates(int i){
	int numFlockmates=0;
	float4 p1, p2;	
	float d, dmin=SearchRadius;
		
	int MaxFlockmates = num/2;	 	

	flockmates = new int[MaxFlockmates];

	p1 = positions[i];
	d_closestFlockmate = SearchRadius;	
//printf("inside SearchFlockmates\n");
	// loop over all boids
	for(int j=0; j < num; j++){
	     if(j != i){
		p2 = positions[j];
		d = distanceFrom(p1, p2);
		
		// is boid j a flockmate?
		if(d < SearchRadius){
//printf("boid %d has %d as a flockmate\n", i, j);  
		  	flockmates[numFlockmates] = j;
			numFlockmates++;
			
			// is boid j the closest flockmate?
			if(d < dmin){
			   dmin = d;
			   d_closestFlockmate = d;
			   ID_closestFlockmate = j;
			}
	
			// did I find the max num of flockmates already?
			if(numFlockmates == MaxFlockmates) break;
		}
	     }		
	}
	return numFlockmates;
}

//****************************************************************
// Separation
float4 Swarm::Separation(int i){
	float r = d_closestFlockmate / SeparationDist;	// TODO: compute the distance of the closest flockmate,  don't need the variable

	float4 separation = positions[ID_closestFlockmate] - positions[i];
	separation = normalize(separation);
	
	if(d_closestFlockmate > SeparationDist){
		separation = times(separation, r);
	}
	else if(d_closestFlockmate < SeparationDist){
		separation = times(separation, -r);
	}
	else{
		separation = times(separation, 0);
	}
	return separation;
}

//****************************************************************
// Alignment
float4 Swarm::Alignment(){
	float4 alignment = velocities[ID_closestFlockmate];
	alignment = normalize(alignment);
	
	return alignment;
}

//****************************************************************
// Cohesion
float4 Swarm::Cohesion(int i){
	float4 flockCenter, cohesion;
	
	for(int k; k < numFlockmates; k++){
		flockCenter = flockCenter + positions[flockmates[k]];
	}
	flockCenter = divides(flockCenter, numFlockmates);

	cohesion = flockCenter - positions[i];
	cohesion = normalize(cohesion);

	return cohesion;
}

//****************************************************************
// FlockIt CPU version
void Swarm::FlockIt_CPU(){


  for(int i=0; i < num; i++){	
	// initialize acc to zero
	acc = float4(0.f, 0.f, 0.f, 0.f);	

	// Step 1. Update position
	positions[i] = positions[i] + velocities[i];

	// Step 2. Search for neighbors
	numFlockmates = SearchFlockmates(i);

	// Step 3. If they are flockmates, compute the three rules
	if(numFlockmates){
		// Separation
		acc = acc + Separation(i);

		// Alignment
		acc = acc + Alignment();

		// Cohesion
		acc = acc + Cohesion(i);	
	}

	// Step 4. Constrain acceleration
	if(magnitude(acc) > MaxChangeInAcc){
		// set magnitude to MaxChangeInAcc
		acc = times(acc, MaxChangeInAcc);
	}

	// Step 5. Add acceleration to velocity
	velocities[i] = velocities[i] + acc;

	// Step 6. Constrain velocity
	if(magnitude(velocities[i]) > MaxSpeed){
		// set magnitude to MaxSpeed
		velocities[i] = times(velocities[i], MaxSpeed);
	}  	
  }
}

}



