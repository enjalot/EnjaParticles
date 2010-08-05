#define STRINGIFY(A) #A

std::string collision_program_source = STRINGIFY(

float4 cross_product(float4 a, float4 b)
{
    return (float4)(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}
float4 v3normalize(float4 a)
{
    float mag = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); //store the magnitude of the velocity
    return (float4)(a.x/mag, a.y/mag, a.z/mag, 0);
}
typedef struct Triangle
{
    float4 verts[3];
    float4 normal;
    //float  dummy;  // for better global to local memory transfer
} Triangle;
//----------------------------------------------------------------------
// Aug. 4, 2010: Erlebacher version with shared memory

//----------------------------------------------------------------------
void test_local(__global float* tri_gl, __local float* tri_f, int one_tri, n_triangles)
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

	int nb_floats = one_tri * n_triangles;

	for (int j = loc_tid; j < nb_floats; j += block_sz) {
		tri_f[j] = tri_gl[j];
	}
}
//----------------------------------------------------------------------
#if 0
//bool intersect_triangle_ge(float4 pos, float4 vel, float dt, __local Triangle* tri) 
bool intersect_triangle_ge(float4 pos, float4 vel, float dt, __local float4* normal, __local float4* vert0, float4 vert1, float4 vert2)
// Assume triangle is in the x-y plane with normal (0,0,1) for simplicity
{
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
    float4 tvec;
    float4 pvec;
    float4 qvec;
    float det;
    float inv_det;
    float t;
    float u;
    float v;
    float eps = .000001;

	float4 pos1 = pos + dt * vel;
	if   (pos1.z <= (*vert0)[2] && pos.z >= (*vert0)[2]) {
	//if (pos1.z <= -1. && pos.z >= -1.) {  // works

	//if ((pos1.z <= tri->verts[0].z && pos.z >= tri->verts[0].z)) {
		return true;
	}

	return false;

#if 0
    //edge1 = tri.verts[1] - tri.verts[0];
    //edge2 = tri.verts[2] - tri.verts[0];

    edge1 = tri->verts[1] - tri->verts[0];
    edge2 = tri->verts[2] - tri->verts[0];


    pvec = cross_product(vel, edge2);
    det = dot(edge1, pvec);
    
    //non-culling branch
    if(det > -eps && det < eps) {
    //if(det < eps)
        return false;
	}
    
    //tvec = pos - tri.verts[0];
    tvec = pos - tri->verts[0];
    inv_det = 1.0/det;

    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    qvec = cross_product(tvec, edge1);
    v = dot(vel, qvec) * inv_det;
    if (v < 0.0 || (u + v) > 1.0f) {
        return false;
	}

    t = dot(edge2, qvec) * inv_det;
    if(t > eps and t < dt)
        return true;

    return false;
#endif
}
#endif
//----------------------------------------------------------------------

#if 1
bool intersect_triangle_ge(float4 pos, float4 vel, __local Triangle* tri, float dist)
{
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
    float4 tvec;
    float4 pvec;
    float4 qvec;
    float det;
    float inv_det;
    float t;
    float u;
    float v;
    float eps = .000001;

    edge1 = tri->verts[1] - tri->verts[0];
    edge2 = tri->verts[2] - tri->verts[0];


    pvec = cross_product(vel, edge2);
    det = dot(edge1, pvec);
    
    //non-culling branch
    if(det > -eps && det < eps) {
    //if(det < eps)
        return false;
	}
    
    //tvec = pos - tri.verts[0];
    tvec = pos - tri->verts[0];
    inv_det = 1.0/det;

    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    qvec = cross_product(tvec, edge1);
    v = dot(vel, qvec) * inv_det;
    if (v < 0.0 || (u + v) > 1.0f) {
        return false;
	}

    t = dot(edge2, qvec) * inv_det;
    if(t > eps and t < dist)
        return true;

    return false;
}
#endif
//----------------------------------------------------------------------
__kernel void collision_ge( __global float4* vertices, __global float4* velocities, 
     __global Triangle* triangles_glob, int n_triangles, float h, __local Triangle* triangles)
{
    unsigned int i = get_global_id(0);

	int one_tri = 16; // nb floats per triangle
	test_local(triangles_glob, triangles, one_tri, n_triangles);
	barrier(CLK_LOCAL_MEM_FENCE);

	// copy triangles to shared memory 
	// need to get more threads involved with global -> shared memory transfer

    float4 pos = vertices[i];
    float4 vel = velocities[i];

	//int tst = 0;
	//store the magnitude of the velocity
    float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); 

	// variables: 
	// triangles, nvel, pos, vel

	//process_triangles(n_triangles, triangles, pos, vel, h);

    //iterate through the list of triangles
    for(int j = 0; j < n_triangles; j++)
    {
        if(intersect_triangle_ge(pos, vel, &triangles[j], h))
        {
            //lets do some specular reflection
    		float4 nvel = v3normalize(vel);

            float s = 2.0f*(dot(triangles[j].normal, nvel));
            float4 dir = s * triangles[j].normal - nvel; //new direction

            float damping = .8f;
            mag *= damping;
            vel = -mag * dir;
            //vel = 0.001;
			//break;
			//tst = 1;
        }
		//if (tst == 1) break;
    }

	// velocities change due to collision
    velocities[i].x = vel.x;
    velocities[i].y = vel.y;
    velocities[i].z = vel.z;
}
);
