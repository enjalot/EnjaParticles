#ifndef __K_SimDEM_cu__
#define __K_SimDEM_cu__

#include "K_Common.cuh"

#include "cutil_math.h"
#include "vector_types.h"

using namespace SimLib;
using namespace SimLib::Sim::DEM;


#include "K_UniformGrid_Utils.cu"
#include "K_UniformGrid_Update.cu"

class DEMSystem
{
public:

	static __device__ void UpdateSortedValues(DEMData &dParticlesSorted, DEMData &dParticles, uint &index, uint &sortedIndex)
	{
		dParticlesSorted.position[index]= FETCH_NOTEX(dParticles,position,sortedIndex);
		dParticlesSorted.velocity[index]= FETCH_NOTEX(dParticles,velocity,sortedIndex);
		dParticlesSorted.veleval[index]	= FETCH_NOTEX(dParticles,veleval,sortedIndex);
	} 
};



class DEMCollisionCalculation
{
public:

	struct Data
	{
		float3 force;
		float3 veleval_i;
		float3 veleval_j;

		DEMData dParticlesSorted;
	};

	class Calc
	{
	public:

		static __device__ void PreCalc(Data &data, uint const &index_i)
		{
			data.veleval_i	= make_float3(FETCH(data.dParticlesSorted, veleval, index_i));
		}

		static __device__ void ForPossibleNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &position_i)
		{
			// check not colliding with self
			if (index_j != index_i) {  

				// get the particle position (in the current cell) to test against
				float3 position_j = make_float3(FETCH(data.dParticlesSorted, position, index_j));

				// get the relative distance between the two particles, translate to simulation space
				float3 r = (position_j - position_i) * cDEMParams.scale_to_simulation;

				float rlen_sq = dot(r,r);
				// |r|
				float rlen = sqrtf(rlen_sq);

				// is this particle within cutoff?
				if (rlen < cDEMParams.collide_dist) 
				{				
					ForNeighbor(data, index_i, index_j, r, rlen, rlen_sq);
				}
			}
		}

		static __device__ void ForNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &r, float const& rlen, float const &rlen_sq)
		{
			float3 veleval_j = make_float3(FETCH(data.dParticlesSorted, veleval, index_j));

			float3 norm = r / rlen;

			// relative velocity
			float3 relVel = veleval_j - data.veleval_i;

			// relative tangential velocity
			float3 tanVel = relVel - (dot(relVel, norm) * norm);

			// spring force
			data.force += -cDEMParams.spring*(cDEMParams.collide_dist - rlen) * norm;

			// dashpot (damping) force
			data.force += cDEMParams.damping*relVel;

			// tangential shear force
			data.force += cDEMParams.shear*tanVel;

			// attraction
			data.force += cDEMParams.attraction*r;
		}

		static __device__ void PostCalc(Data &data, uint index_i)
		{
			data.dParticlesSorted.force[index_i] = make_vec(data.force);
		}

	};
};


__global__ void computeCollisions(uint			numParticles,
							   NeighborList		dNeighborList, 
							   DEMData			dParticlesSorted,
							   GridData const	dGridData
							   )								
{
	// particle index	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if (index >= numParticles) return;

	DEMCollisionCalculation::Data data;
	data.dParticlesSorted = dParticlesSorted;

	float3 position_i = make_float3(FETCH(dParticlesSorted, position, index));

	// Do calculations on particles in neighboring cells
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	UniformGridUtils::IterateParticlesInNearbyCells<DEMCollisionCalculation::Calc, DEMCollisionCalculation::Data>(data, index, position_i, dNeighborList);	
#else
	UniformGridUtils::IterateParticlesInNearbyCells<DEMCollisionCalculation::Calc, DEMCollisionCalculation::Data>(data, index, position_i, dGridData);
#endif

}


#include "K_Coloring.cu"
#include "K_Boundaries_Terrain.cu"
#include "K_Boundaries_Walls.cu"

__global__ void integrateDEM(int				numParticles,
								bool			gridWallCollisions,
								bool			terrainCollisions,
								float			delta_time,
								bool			progress,
								GridData		dGridData,
								DEMData			dParticles, 
								DEMData			dParticlesSorted, 
								TerrainData		dTerrainData
								) 
{
	int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	float3 pos			= make_float3(FETCH_NOTEX(dParticlesSorted, position, index));
	float3 vel			= make_float3(FETCH_NOTEX(dParticlesSorted, velocity, index));
	float3 vel_eval		= make_float3(FETCH_NOTEX(dParticlesSorted, veleval, index));
	float3 dem_force	= make_float3(FETCH_NOTEX(dParticlesSorted, force, index));

	float3 f_extforce = make_float3(0,0,0);

	// add gravity	
	f_extforce += cDEMParams.gravity;	

	// add no-penetration dem_force due to terrain
	if(terrainCollisions)
		f_extforce += calculateTerrainNoPenetrationForce(
		pos, vel_eval, dTerrainData,
		cDEMParams.boundary_distance,
		cDEMParams.boundary_stiffness,
		cDEMParams.boundary_dampening,
		cDEMParams.scale_to_simulation);

	// todo: add no-slip dem_force due to terrain..
	if(terrainCollisions)
		f_extforce += calculateTerrainFrictionForce(
		pos, vel_eval, dTerrainData,
		cDEMParams.boundary_distance,
		cDEMParams.boundary_stiffness,
		cDEMParams.boundary_dampening,
		cDEMParams.scale_to_simulation);

	// add no-penetration dem_force due to "walls"
	if(gridWallCollisions)
		f_extforce += calculateWallsNoPenetrationForce(
		pos, vel_eval,
		cGridParams.grid_min, 
		cGridParams.grid_max,
		cDEMParams.boundary_distance,
		cDEMParams.boundary_stiffness,
		cDEMParams.boundary_dampening,
		cDEMParams.scale_to_simulation);


	float3 accel = dem_force + f_extforce;

	// Leapfrog integration		
	// v(t+1/2) = v(t-1/2) + a(t) dt	
	float3 vnext = vel + accel * delta_time;
	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5
	vel_eval = (vel + vnext) * 0.5;
	vel = vnext;

	// update position of particle
	pos += vnext * (delta_time / cDEMParams.scale_to_simulation);

	if(progress)
	{
		uint originalIndex = dGridData.sort_indexes[index];

		// writeback to unsorted buffer	
		dParticles.position[originalIndex]	= make_vec(pos);
		dParticles.velocity[originalIndex]	= make_vec(vel);
		dParticles.veleval[originalIndex]	= make_vec(vel_eval);

		float colorScalar = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 10000.0;
		colorScalar = clamp(colorScalar, 0.0f, 1.0f);
 		float3 color = calculateColor(HSVBlueToRed, colorScalar);
		dParticles.color[originalIndex]	= make_float4(color, 1);
	}
}
