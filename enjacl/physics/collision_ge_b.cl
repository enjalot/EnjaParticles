#define STRINGIFY(A) #A

std::string collision_program_source = STRINGIFY(
// prototype does not work?
//bool intersect_triangle_ge(float4 pos, float4 vel, __local Triangle* tri, float dist, bool collided);

//----------------------------------------------------------------------
float4 cross_product(float4 a, float4 b)
{
    return (float4)(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}
//----------------------------------------------------------------------
float4 v3normalize(float4 a)
{
    float mag = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); //store the magnitude of the velocity
    return (float4)(a.x/mag, a.y/mag, a.z/mag, 0);
}
//----------------------------------------------------------------------
typedef struct Triangle
{
    float4 verts[3];
    float4 normal;
    //float  dummy;  // for better global to local memory transfer
} Triangle;
//----------------------------------------------------------------------
// Aug. 4, 2010: Erlebacher version with shared memory

//----------------------------------------------------------------------
void test_local(__global float* tri_gl, __local float* tri_f, int one_tri, int first_tri, int last_tri)
// one_tri: nb floats in one Triangle structure
{
/*
	thread id: 0 --> get_global_size(0);
	local thread id: 0 -> get_local_size(0);
	thread id: get_global_id(0);  ==> particle number
	local id: get_local_id(0);   
*/

	int block_sz = get_local_size(0);
	int loc_tid = get_local_id(0);

	// first = 3, last = 7, tri = 3,4,5,6 = last - first
	int nb_floats = one_tri * (last_tri-first_tri);

	for (int j = loc_tid; j < nb_floats; j += block_sz) {
		//if ((j+first_tri) > last_tri) break;
		tri_f[j] = tri_gl[j+first_tri*one_tri];
	}
}
//----------------------------------------------------------------------
#if 1
bool intersect_triangle_ge(float4 pos, float4 vel, __local Triangle* tri, float dist, bool collided)
{
	// There is serialization (since not all threads in a warp will have 
	// "collided" set the same way)

	// More efficient if this test is not performed. Not sure why.
	//if (collided) return;

    /*
    * Moller and Trumbore
    * take in the particle position and velocity (treated as a Ray)
    * also the triangle vertices for the ray intersection
    * we take in the precalculated triangle's normal to first test for distance
    * dist is the threshold to determine if we are close enough to the triangle
    * to even check for distance
    */
    //can't use commas with STRINGIFY trick
    float4 edge1;
    float4 edge2;

    edge1 = tri->verts[1] - tri->verts[0];
    edge2 = tri->verts[2] - tri->verts[0];

    float4 pvec;
    pvec = cross_product(vel, edge2);

    float det;
    det = dot(edge1, pvec);
    float eps = .000001;

    //non-culling branch
    if(det > -eps && det < eps) {    // <<<<< if
    //if(det < eps)
        return false;
	}

    float4 tvec;
    tvec = pos - tri->verts[0];

	// reduce register usage
    float inv_det = 1.0/det;

    float u;
    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) {     // <<<<< if
        return false;
	}

    float4 qvec;
    qvec = cross_product(tvec, edge1);

    float v;
    v = dot(vel, qvec) * inv_det;
    if (v < 0.0 || (u + v) > 1.0f) { // <<<< if
        return false;
	}

    float t;
    t = dot(edge2, qvec) * inv_det;

    if(t > eps and t < dist)
        return true;

    return false;
}
#endif
//----------------------------------------------------------------------
#if 1
float4 collisions(float4 pos, float4 vel, int first, int last, __global Triangle* triangles_glob, float h, __local Triangle* triangles)
{
	int one_tri = 16;
	// copy triangles to shared memory 
	// first, last: indices into triangles_glob
	test_local(triangles_glob, triangles, one_tri, first, last);
	barrier(CLK_LOCAL_MEM_FENCE);

	//store the magnitude of the velocity
	#if 1
    float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); 
    float damping = 1.0f;

	// variables: 
	// triangles, pos, vel

	bool collided = false;

    //iterate through the list of triangles
	// This for() and/or if() slows things down considerably. 
	// Proof: putting the last three lines inside the for, before the 
	//    break, slows the code down!

    for (int j=0; j < (last-first); j++)
    {
        collided = intersect_triangle_ge(pos, vel, &triangles[j], h, collided);
        if (collided)
        {
            //lets do some specular reflection

            float s = 2.0f*(dot(triangles[j].normal, vel));
			vel = vel - s*triangles[j].normal;

			/*
 				vp = v - (v.n) n
 				vn = (v.n) n

 				New velocity = (vp, -vn) = v - (v.n)n - v.n n
                   = v - 2n v.n
           */


			vel = vel*damping;

			// faster without the break. Not sure why. 
			//break;
        }
    }
#endif

	return vel;
}
#endif
//----------------------------------------------------------------------
__kernel void collision_ge( __global float4* vertices, __global float4* velocities, __global Triangle* triangles_glob, int n_triangles, float h,__global float4* transform, __local Triangle* triangles )
{
    unsigned int i = get_global_id(0);
    float4 pos = vertices[i];
    float4 vel = velocities[i];

    //transform pos to get global coordinates
    //3x3 matrix multiply followed by vector add
    float4 pos_t = (float4)(dot(transform[0], pos), dot(transform[1], pos), dot(transform[2], pos), 0);
    pos = pos_t + transform[3];

	//int one_tri = 16; // nb floats per triangle
	// Find a way to Iterate over batches of n_triangles so the number
	// of triangles can be increased. 

	//int max_tri = 220;
	int max_tri = 220;

	for (int j=0; j < n_triangles; j += max_tri) {
		int first = j;
		int last = first + max_tri;

		if (last > n_triangles) {
			last = n_triangles;
		}
		vel = collisions(pos, vel, first, last, triangles_glob, h, triangles);
	}


    velocities[i].x = vel.x;
    velocities[i].y = vel.y;
    velocities[i].z = vel.z;
}
);
//----------------------------------------------------------------------
