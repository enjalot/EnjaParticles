#ifndef _cl_snippet_sphere_forces_h_
#define _cl_snippet_sphere_forces_h_

#if 0
void ForNeighbor(__global float4*  var_sorted,
				__constant uint index_i,
				uint index_j,
				float4 r,
				float rlen,
				float rlen_sq,
	  			__constant struct GridParams* gp,
	  			__constant struct FluidParams* fp)
{
}
#endif

	int numParticles = gp->numParticles;
	float4 ri = FETCH_POS(var_sorted, index_i);
	float4 rj = FETCH_POS(var_sorted, index_j);
	float4 relPos = rj-ri;
	float dist = length(relPos);
	float collideDist = 2.*fp->smoothing_length; // smoothing_length = particle radius

	if (dist < collideDist) {
		float4 vi = FETCH_VEL(var_sorted, index_i);
		float4 vj = FETCH_VEL(var_sorted, index_j);
		//float4 force_diff = 2.f*(vi-vj) / fp->dt; // assume gp->mass = 1.
		float4 norm = relPos / dist;

		// relative velocity
		float4 relVel = vj - vi;

		// relative tangential velocity
		float4 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		float4 force = -fp->spring*(collideDist - dist) * norm;

		// dashpot (damping) force
		force +=fp->damping*relVel;

		// tangential shear force
		force += fp->shear*tanVel;
		force += fp->attraction*relPos;
	}

	return force;



	#if 0
	//dist = length(vi);

	// Update forces
	float4 force_i = FETCH_FOR(var_sorted, index_i);
	float4 force_j = FETCH_FOR(var_sorted, index_j);

	// Signs may have to be changed
	//force_i += force_diff;
	//force_j -= force_diff;

	FETCH_VEL(var_sorted, index_i) = force_i;
	FETCH_VEL(var_sorted, index_j) = force_j;

	// Have to make sure gravity is included as an input
	#endif

#endif
