#define STRINGIFY(A) #A

// Aug. 4, 2010: Erlebacher version with shared memory
// Aug. 6, 2010: Erlebacher version with bounding boxes instead of triangles
//   Test if new position is inside box. If so, then we'll assume a collision 
//   occured. Otherwise not. This is not perfect since a particle could cut 
//   across a corner, but this is unlikely to be a problem unless the 
//   boxes are small. 




std::string collision_program_source = STRINGIFY(
// prototype does not work?

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
void global_to_shared_tri(__global float* tri_gl, __local float* tri_f, int one_tri, int first_tri, int last_tri)
// one_tri: nb floats in one Triangle structure
{
/*
	thread id: 0 --> get_global_size(0);
	local thread id: 0 -> get_local_size(0);
	thread id: get_global_id(0);  ==> particle number
	local id: get_local_id(0);   
*/

	#if 1
	int block_sz = get_local_size(0);
	int loc_tid = get_local_id(0);

	// first = 3, last = 7, tri = 3,4,5,6 = last - first
	int nb_floats = one_tri * (last_tri-first_tri);

	for (int j = loc_tid; j < nb_floats; j += block_sz) {
		//if ((j+first_tri) > last_tri) break;
		tri_f[j] = tri_gl[j+first_tri*one_tri];
	}
	#endif
}
//----------------------------------------------------------------------
#if 1
void global_to_shared_boxes(__global float* box_gl, __local float* box_f, int one_box, int first_box, int last_box)
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
#endif
//----------------------------------------------------------------------
#if 1
bool intersect_triangle_ge(float4 pos, float4 vel, __global Triangle* tri, float dist, bool collided)
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
bool intersect_box_ge(float4 pos, float4 vel, __local Box* box, float dt)
{
	// There is serialization (since not all threads in a warp will have 
	// "collided" set the same way)

	float4 pos1 = pos + dt*vel;

	// is pos1 inside the box. If yes, collide is true

	if (pos1.x > box->xmin && pos1.x < box->xmax && 
	    pos1.y > box->ymin && pos1.y < box->ymax && 
	    pos1.z > box->zmin && pos1.z < box->zmax) {

		return true;
	}
	return false;
}
#endif
//----------------------------------------------------------------------

float4 collisions_box(float4 pos, float4 vel, int first, int last, __global Box* boxes_glob, float dt, __local Box* boxes, __global Triangle* triangles, int f_tri, int l_tri, __global int* tri_offsets)
{
#if 1
	int one_box = 6; // nb floats in Box
	// copy triangles to shared memory 
	// first, last: indices into triangles_glob
	global_to_shared_boxes(boxes_glob, boxes, one_box, first, last);
	barrier(CLK_LOCAL_MEM_FENCE);

	//store the magnitude of the velocity
    //float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); 
    float damping = 1.0f;

	// variables: 
	// triangles, pos, vel

    //iterate through the list of boxes
	// This for() and/or if() slows things down considerably. 
	// Proof: putting the last three lines inside the for, before the 
	//    break, slows the code down!

    for (int j=0; j < (last-first); j++)
    {
		f_tri = tri_offsets[j+first];
		l_tri = tri_offsets[j+1+first];
        int collided = intersect_box_ge(pos, vel, &boxes[j], dt);

        if (collided)
        {
			//vel = 0.0;
			// go down the triangle list
			// 3rd arg: nb floats in Triangle

			// Do not put triangle list in local memory 
			bool col = false;
			for (int k=f_tri; k < l_tri; k++) {
				// I should exit both loops if col == true
				col = intersect_triangle_ge(pos, vel, &triangles[k], dt, col);
				if (col) {
            		float s = 2.0f*(dot(triangles[k].normal, vel));
					vel = vel - s*triangles[k].normal;
					vel = vel*damping;
                    //vel = (float4)(0,0,0,0); //this doesn't seem to affect anything
					return vel; // slow down the code? 
					break;
				}
			}
        }
    }

#endif
	return vel;
}
//----------------------------------------------------------------------
__kernel void collision_ge( __global float4* vertices, __global float4* velocities, __global Box* boxes_glob, int n_boxes, float dt, __global int* tri_offsets, __global int* triangles,  __local Box* boxes)
{
#if 1
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
		int f_tri = tri_offsets[j];
		int l_tri = tri_offsets[j+1];
		// offsets are monotonic
		vel = collisions_box(pos, vel, first, last, boxes_glob, dt, boxes, triangles, f_tri, l_tri, tri_offsets);
	}

    velocities[i].x = vel.x;
    velocities[i].y = vel.y;
    velocities[i].z = vel.z;
    /*
       //this destroys the simulation (everything stops moving)
    velocities[i].x = 0;
    velocities[i].y = 0;
    velocities[i].z = 0;
    */
#endif
}
);
//----------------------------------------------------------------------
