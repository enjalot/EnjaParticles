#define STRINGIFY(A) #A

// Aug. 4, 2010: Erlebacher version with shared memory
// Aug. 6, 2010: Erlebacher version with bounding boxes instead of triangles
//   Test if new position is inside box. If so, then we'll assume a collision 
//   occured. Otherwise not. This is not perfect since a particle could cut 
//   across a corner, but this is unlikely to be a problem unless the 
//   boxes are small. 



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
typedef struct Box
{
// "," not allowed due to STRINGIFY
	float xmin; 
	float xmax;
	float ymin; 
	float ymax;
	float zmin;
	float zmax;
} Box;
//----------------------------------------------------------------------
//----------------------------------------------------------------------
typedef struct Triangle
{
    float4 verts[3];
    float4 normal;
    //float  dummy;  // for better global to local memory transfer
} Triangle;
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void test_local(__global float* box_gl, __local float* box_f, int one_box, int first_box, int last_box)
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

	// one_box: nb floats in Box
	int nb_floats = one_box * (last_box-first_box);

	for (int j = loc_tid; j < nb_floats; j += block_sz) {
		//if ((j+first_tri) > last_tri) break;
		box_f[j] = box_gl[j+first_box*one_box];
	}
}
//----------------------------------------------------------------------
#if 1
bool intersect_box_ge(float4 pos, float4 vel, __local Box* box, float dt, bool collided)
{
	// There is serialization (since not all threads in a warp will have 
	// "collided" set the same way)

	// More efficient if this test is not performed. Not sure why.
	//if (collided) return;

	float4 pos1 = pos + dt*vel;

	// is pos1 inside the box. If yes, collide is true

	if (pos1.x > box.xmin && pos1.x < box.xmax && 
	    pos1.y > box.ymin && pos1.y < box.ymay && 
	    pos1.z > box.zmin && pos1.z < box.zmaz) {

		collided == true;
		return collided
	}
}
#endif
//----------------------------------------------------------------------
float4 collisions(float4 pos, float4 vel, int first, int last, __global Box* boxes_glob, float h, __local Box* boxes)
{
	int one_box = 6; // nb floats in Box
	// copy triangles to shared memory 
	// first, last: indices into triangles_glob
	test_local(boxes_glob, boxes, one_box, first, last);
	barrier(CLK_LOCAL_MEM_FENCE);

	//store the magnitude of the velocity
#if 0
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
        collided = intersect_box_ge(pos, vel, &triangles[j], h, collided);
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
//----------------------------------------------------------------------
__kernel void collision_ge( __global float4* vertices, __global float4* velocities, __global Box* boxes_glob, int n_boxes, float h, __local Box* boxes)
{
    unsigned int i = get_global_id(0);
    float4 pos = vertices[i];
    float4 vel = velocities[i];

	// Find a way to Iterate over batches of n_triangles so the number
	// of triangles can be increased. 

	//int max_tri = 220;
	int max_box = 600;

	for (int j=0; j < n_boxes; j += max_box) {
		int first = j;
		int last = first + max_box;

		if (last > n_boxes) {
			last = n_boxes;
		}
		vel = collisions(pos, vel, first, last, boxes_glob, h, boxes);
	}

    velocities[i].x = vel.x;
    velocities[i].y = vel.y;
    velocities[i].z = vel.z;
}
);
//----------------------------------------------------------------------
