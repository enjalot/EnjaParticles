#ifndef _cl_snippet_sphere_forces_h_
#define _cl_snippet_sphere_forces_h_

#if 0
void ForNeighbor(__global float4*  vars_sorted,
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

#if 1
	int numParticles = gp->numParticles;
	float4 ri = FETCH_POS(vars_sorted, index_i);
	float4 rj = FETCH_POS(vars_sorted, index_j);
	float4 relPos = rj-ri;
	float dist = length(relPos);
	float collideDist = 2.*fp->smoothing_length; // smoothing_length = particle radius
	float4 force;

	if (dist < collideDist) {
		float4 vi = FETCH_VEL(vars_sorted, index_i);
		float4 vj = FETCH_VEL(vars_sorted, index_j);
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

	//FETCH_FOR(vars_sorted, index) = force;
	}

#endif



#endif
