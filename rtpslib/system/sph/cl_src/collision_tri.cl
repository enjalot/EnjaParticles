#include "cl_macros.h"
#include "cl_structs.h"
#include "cl_collision.h"


// Aug. 4, 2010: Erlebacher version with shared memory
// Aug. 6, 2010: Erlebacher version with bounding boxes instead of triangles
//   Test if new position is inside box. If so, then we'll assume a collision 
//   occured. Otherwise not. This is not perfect since a particle could cut 
//   across a corner, but this is unlikely to be a problem unless the 
//   boxes are small. 

// Jan. 15, 2011: enjalot version dealing with SPH forces instead of direct velocity changes
//
// 



//----------------------------------------------------------------------
float4 cross_product(float4 a, float4 b)
{
    return (float4)(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}
//----------------------------------------------------------------------
float magnitude(float4 a)
{
    float mag = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); //store the magnitude of the velocity
    return mag;
}
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
		tri_f[j] = tri_gl[j+first_tri*one_tri];       //multiply by simulation scale
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
//bool intersect_triangle_ge(float4 pos, float4 vel, __global Triangle* tri, float dist, bool collided)
float intersect_triangle_ge(float4 pos, float4 vel, __local Triangle* tri, float dist, float eps, float scale)
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
    float4 v0 = tri->verts[0] * scale;
    float4 v1 = tri->verts[1] * scale;
    float4 v2 = tri->verts[2] * scale;

    edge1 = v1 - v0;
    edge2 = v2 - v0;

    float4 pvec;
    pvec = cross_product(vel, edge2);

    float det;
    det = dot(edge1, pvec);

    //non-culling branch
    if(det > -eps && det < eps) {    // <<<<< if
    //culling
    //if(det < eps){
        return 2*dist;
        //return false;
	}

    float4 tvec;
    tvec = pos - v0;

	// reduce register usage
    float inv_det = 1.0/det;

    float u;
    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) {     // <<<<< if
        //return false;
        return 2*dist;
	}

    float4 qvec;
    qvec = cross_product(tvec, edge1);

    float v;
    v = dot(vel, qvec) * inv_det;
    if (v < 0.0 || (u + v) > 1.0f) { // <<<< if
        //return false;
        return 2*dist;
	}

    float t;
    t = dot(edge2, qvec) * inv_det;

    //if(t > eps and t < dist)
    if(t > -2*dist && t < dist)
        return t;

    return 2*dist;
}
#endif

float intersect_triangle_segment(float4 pos, float4 vel, __local Triangle* tri, float dist, float eps, float scale)
{
    /* 
     * Ericson: Real Time Collision Detection. pp 191-192
     */
    float4 a = tri->verts[0] * scale;
    float4 b = tri->verts[1] * scale;
    float4 c = tri->verts[2] * scale;

    float4 ab = b - a;
    float4 ac = c - a;

    //dist /= scale;
    //pos /= scale;
    float4 vn = v3normalize(vel);
    float4 p = pos - vn*dist;
    float4 q = pos + vn*dist;
    float4 qp = p - q;

    //float4 n = tri->normal;
    float4 n = cross_product(ab, ac);
    //something seems to be wrong with Blender's normals
    //so to be safe we calculate here
    tri->normal = v3normalize(n);

    float d = dot(qp, n);
    if ( d <= 0.0f) return 2*dist;

    float4 ap = p - a;
    float t = dot(ap, n);
    if (t < 0.0f || t > d) return 2*dist;

    float4 e = cross_product(qp, ap);
    float v = dot(ac, e);
    if(v < 0.0f || v > d) return 2*dist;
    float w = -dot(ab, e);
    if ( w < 0.0f || v + w > d) return 2*dist;

    //intersection
    float ood = 1.0f / d;
    t *= ood;
    v *= ood;
    w *= ood;
    float u = 1.0f - v - w;
    float4 tp = u*a + v*b + w*c;
    float distance = magnitude(pos - tp);
    return distance;// * scale;

    //return dist - 2.*t*dist;



}

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
#if 1
float4 collisions_triangle(float4 pos, 
        float4 vel, 
        float4 force, 
        int first, 
        int last, 
        __global Triangle* triangles_glob, 
        float dt, 
        __local Triangle* triangles, 
		__constant struct SPHParams* sphp
)
{
	int one_tri = 16;
	// copy triangles to shared memory 
	// first, last: indices into triangles_glob
	global_to_shared_tri(triangles_glob, triangles, one_tri, first, last);
	barrier(CLK_LOCAL_MEM_FENCE);

	//store the magnitude of the velocity
	#if 1
    //float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); 
    //float damping = 1.0f;

    //these should be moved to the sphp struct
    //but set to 0 in both of Krog's simulations...
    float friction_kinetic = 0.0f;
    float friction_static_limit = 0.0f;


    float eps = .000001;
    float distance = 0.0f;
    float4 f = (float4)(0,0,0,0); //returning the force
    int tc = 0;
    float dtdt = 1.0/dt;
    dtdt *= dtdt;
    for (int j=0; j < (last-first); j++)
    {
        //distance = intersect_triangle_ge(pos, vel, &triangles[j], sphp->boundary_distance, eps, sphp->simulation_scale);
        distance = intersect_triangle_segment(pos, vel, &triangles[j], sphp->boundary_distance, eps, sphp->simulation_scale);
        //distance = intersect_triangle_ge(pos, vel, &triangles[j], dt, eps);
        //if ( distance != -1)
        //distance = sphp->boundary_distance - distance;
        distance = sphp->rest_distance - distance;
        if (distance > eps)// && distance < sphp->boundary_distance)
        {
            //Krog boundary forces
            //f += calculateRepulsionForce(triangles[j].normal, vel, 1*sphp->boundary_stiffness, 1*sphp->boundary_dampening, distance);
            //f += calculateFrictionForce(vel, force, triangles[j].normal, friction_kinetic, friction_static_limit);
            //Harada boundary wall forces
            f += distance * triangles[j].normal * dtdt;
            //f += (float4)(1100,1100,1100,1);
			/*
            //lets do some specular reflection
            float s = 2.0f*(dot(triangles[j].normal, vel));
			vel = vel - s*triangles[j].normal;
 				vp = v - (v.n) n
 				vn = (v.n) n

 				New velocity = (vp, -vn) = v - (v.n)n - v.n n
                   = v - 2n v.n
			vel = vel*damping;
           */
        }
        f.w += 1;
    }
    #endif
	return f;
}
#endif



//----------------------------------------------------------------------

float4 collisions_box(float4 pos, float4 vel, int first, int last, __global Box* boxes_glob, float dt, __local Box* boxes, __global Triangle* triangles, int f_tri, int l_tri, __global int* tri_offsets)
{
#if 0
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
					return vel; // slow down the code? 
					break;
				}
			}
        }
    }

#endif
	//TODO
	//need to change this to return a force
	return vel;
}
//----------------------------------------------------------------------
//__kernel void collision_ge( __global float4* pos, __global float4* vel, __global float4* force, __global Box* boxes_glob, int n_boxes, float dt, __global int* tri_offsets, __global int* triangles,  __local Box* boxes)
__kernel void collision_triangle(   //__global float4* vars_sorted, 
                                    __global float4* pos_s,
                                    __global float4* vel_s,
                                    __global float4* force_s,
                                    __global Triangle* triangles_glob, 
                                    int n_triangles, 
                                    float dt, 
		                            __constant struct SPHParams* sphp,
                                    __local Triangle* triangles
				                    DEBUG_ARGS
                                    )
{
#if 1
    unsigned int i = get_global_id(0);
	int num = sphp->num;

    /*
    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);
    */
    float4 p = pos_s[i] * sphp->simulation_scale;
    float4 v = vel_s[i];
    float4 f = force_s[i];



    int tc = 0; //triangle count
	int max_tri = 220;
	//int max_box = 600;
    
    float4 rf = (float4)(0,0,0,0); //returning the force
	//for (int j=0; j < n_boxes; j += max_box) {
	for (int j=0; j < n_triangles; j += max_tri) {
		int first = j;
		//int last = first + max_box;
		int last = first + max_tri;
/*
		if (last > n_boxes) {
			last = n_boxes;
		}
*/
		if (last > n_triangles) {
			last = n_triangles;
		}

		//int f_tri = tri_offsets[j];
		//int l_tri = tri_offsets[j+1];
		// offsets are monotonic
		//f = collisions_box(p, v, first, last, boxes_glob, dt, boxes, triangles, f_tri, l_tri, tri_offsets);
		rf += collisions_triangle(p, v, f, first, last, triangles_glob, dt, triangles, sphp);
    //rf = (float4)(11.,11.,11.,1);
	}

    if(i > num) return;


    /*
    clf[i] = rf;
    //clf[i].w = pos(i).z / sphp->simulation_scale;
    clf[i].w = pos_s[i].z / sphp->simulation_scale;
    cli[i].x = (int)rf.w;
    cli[i].y = sphp->num;
    cli[i].z = get_local_size(0);
    */
    
    /*
    float mag = sqrt(rf.x*rf.x + rf.y*rf.y + rf.z*rf.z); //store the magnitude of the velocity
    if (mag > 0){
        vel(i) = (float4)(0,0,0,0);
        force(i) = (float4)(0,0,0,0);
    }
    else{
    */
    //force(i) += rf;
    force_s[i] += rf;
    //}
    
/*
    vel[i].x = v.x;
    vel[i].y = v.y;
    vel[i].z = v.z;
*/
#endif
}
//----------------------------------------------------------------------
