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
bool intersect_triangle(float4 pos, float4 vel, float4 tri[3], float4 triN, float dist)
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

    edge1 = tri[1] - tri[0];
    edge2 = tri[2] - tri[0];

    pvec = cross_product(vel, edge2);
    det = dot(edge1, pvec);
    
    //non-culling branch
    if(det > -eps && det < eps)
    //if(det < eps)
        return false;
    
    tvec = pos - tri[0];
    inv_det = 1.0/det;

    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    qvec = cross_product(tvec, edge1);
    v = dot(vel, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0f)
        return false;

    t = dot(edge2, qvec) * inv_det;
    if(t > eps and t < dist)
        return true;

    return false;

}
__kernel void collision( __global float4* vertices, __global float4* velocities, float h)
{
    unsigned int i = get_global_id(0);

    float4 pos = vertices[i];
    float4 vel = velocities[i];

    //set up test plane
    float4 plane[4];
    plane[0] = (float4)(-2,-2,-1,0);
    plane[1] = (float4)(-2,2,-1,0);
    plane[2] = (float4)(2,2,-3,0);
    plane[3] = (float4)(2,-2,-1,0);

    //triangle fan from plane (for handling faces)
    float4 tri[3];
    tri[0] = plane[0];
    tri[1] = plane[1];
    tri[2] = plane[2];

    //calculate the normal of the triangle
    //might not need to do this if we just have plane's normal
    float4 A = tri[0];
    float4 B = tri[1];
    float4 C = tri[2];
    
    float4 triN = v3normalize(cross_product(B - A, C - A));
    //float4 triN = (float4)(0.0, 0.0, 1.0, 0.0); 

    if(intersect_triangle(pos, vel, tri, triN, h))
    {
        //lets do some specular reflection
        float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); //store the magnitude of the velocity
        float4 nvel = v3normalize(vel);
        float s = 2.0f*(dot(triN, nvel));
        float4 dir = s * triN - nvel; //new direction
        float damping = .5f;
        mag *= damping;
        vel = -mag * dir;
    }
   
    velocities[i].x = vel.x;
    velocities[i].y = vel.y;
    velocities[i].z = vel.z;
}
);
