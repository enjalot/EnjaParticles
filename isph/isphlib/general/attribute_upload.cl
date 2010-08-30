// TODO transform this function into generic attribute setter
__kernel void SetObjectVelocity
(
	__global vector * attr	: VELOCITIES,
	vector newVel			: VECTOR_VALUE,
	uint firstParticle		: OBJECT_START,
	uint count				: OBJECT_PARTICLE_COUNT
)
{
	uint i = get_global_id(0);
	if(i < count)
		attr[i+firstParticle] = newVel;
}
