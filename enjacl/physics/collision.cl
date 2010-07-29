
float4 cross_product(float4 a, float4 b)
{
    return (float4)(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

//Moller and Trumbore
bool intersect_triangle(float4 pos, float4 vel, float4 tri[3], float4 triN, float dist)
{
    //take in the particle position and velocity (treated as a Ray)
    //also the triangle vertices for the ray intersection
    //we take in the precalculated triangle's normal to first test for distance
    //dist is the threshold to determine if we are close enough to the triangle
    //to even check for distance
    float4 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det, u, v;
    float eps = .000001;

    //check distance
    tvec = pos - tri[0];
    float distance = -dot(tvec, triN) / dot(vel, triN);
    if (distance > dist)
        return false;


    edge1 = tri[1] - tri[0];
    edge2 = tri[2] - tri[0];

    pvec = cross(vel, edge2);
    det = dot(edge1, pvec);
    //culling branch
    //if(det > -eps && det < eps)
    if(det < eps)
        return false;

    u = dot(tvec, pvec);
    if (u < 0.0 || u > det)//1.0)
        return false;

    qvec = cross(tvec, edge1);
    v = dot(vel, qvec);
    if (v < 0.0 || u + v > det)//1.0f)
        return false;

    return true;
}



//update the particle position and color
__kernel void enja(__global float4* vertices, __global float4* colors, __global int* indices, __global float4* vert_gen, __global float4* velo_gen, __global float4* velocities, float h)
{
    unsigned int i = get_global_id(0);

    float life = velocities[i].w;
    life -= h/10;    //should probably depend on time somehow
    //h = h*10;
    if(life <= 0.)
    {
        //reset this particle
        vertices[i].x = vert_gen[i].x;
        vertices[i].y = vert_gen[i].y;
        vertices[i].z = vert_gen[i].z;

        velocities[i].x = velo_gen[i].x;
        velocities[i].y = velo_gen[i].y;
        velocities[i].z = velo_gen[i].z;
        life = 1.;
    } 
    float4 pos = vertices[i];
    float4 vel = velocities[i];

    float xn = pos.x;
    float yn = pos.y;
    float zn = pos.z;

    float vxn = vel.x;
    float vyn = vel.y;
    float vzn = vel.z;
    vel.x = vxn;
    vel.y = vyn - h*9.8;
    vel.z = vzn;// - h*9.8;

    xn += h*vel.x;
    yn += h*vel.y;
    zn += h*vel.z;
    

    //set up test plane
    float4 plane[4];
    plane[0] = (float4)(-2,-1,-2,0);
    plane[1] = (float4)(-2,-1,2,0);
    plane[2] = (float4)(2,-1,2,0);
    plane[3] = (float4)(2,-1,-2,0);

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
    
    float4 triN = normalize(cross_product(B - A, C - A));
    //float4 tri1N = (float4)(0.0, 1.0, 0.0, 0.0); 

    if(intersect_triangle(pos, vel, tri, triN, h))
    {
        //lets do some specular reflection
        float mag = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z); //store the magnitude of the velocity
        float4 nvel = normalize(vel);
        float s = 2.0f*(dot(triN, nvel));
        float4 dir = s * triN - nvel; //new direction
        float damping = .5f;
        mag *= damping;
        vel = -mag * dir;

        xn = pos.x + h*vel.x;
        yn = pos.y + h*vel.y;
        zn = pos.z + h*vel.z;

    }
    
    vertices[i].x = xn;
    vertices[i].y = yn;
    vertices[i].z = zn;

     
    colors[i].x = 1.f;
    colors[i].y = life;
    colors[i].z = life;
    //colors[i].w = 1-life;
    colors[i].w = 1;
    
    velocities[i] = vel;
    //save the life!
    velocities[i].w = life;
}

