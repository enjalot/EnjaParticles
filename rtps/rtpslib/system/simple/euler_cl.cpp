//normalized vector pointing from p1 to p2
float4 norm_dir(float4 p1, float4 p2)
{
    float4 dir = (float4)(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, 0.0f);
    float norm = length(dir);
    if(norm > 0)
    {
        dir /= norm;
    }
    return dir;
}

float4 predator_prey(float4 p)
{
    float4 v = (float4)(0,0,0,0);
    int a1 = 2;
    int a2 = 2;
    int b1 = 1;
    int b2 = 1;
    v.x = a1*p.x - b1*p.x*p.y;
    v.y = -a2*p.y + b2*p.y*p.x;
    //v.x = a1 - b1*p.y;
    //v.y = -a2 + b2*p.x;
    return v;
}

float4 runge_kutta(float4 yn, float h)
{
    float4 k1 = predator_prey(yn); 
    float4 k2 = predator_prey(yn + .5f*h*k1);
    float4 k3 = predator_prey(yn + .5f*h*k2);
    float4 k4 = predator_prey(yn + h*k3);

    float4 vn = (k1 + 2.f*k2 + 2.f*k3 + k4);
    return vn/6.0f;
}



float4 force_field(float4 p, float4 ff, float dist, float max_force)
{
    float d = distance(p, ff);
    if(d < dist)
    {
        float4 dir = norm_dir(p, ff);
        float mag = max_force * (dist - d)/dist;
        dir *= mag;
        return dir;
    }
    return (float4)(0, 0, 0, 0);
}


__kernel void euler(__global float4* pos, __global float4* vel, __global float4* force, __global float4* colors, float h)
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i];
    float4 f = force[i];


    //external force is gravity
    //f.z += -9.8f;
    float4 ffp = (float4)(1.0f,1.0f,0.0f,1.0f);
    float dist = 1.0f;
    float max_force = 20.0f;
    float4 ff = force_field(p, ffp, dist, max_force);
    f += ff;

    v += h*f;
    p += h*v;
    p.w = 1.0f; //just in case

    vel[i] = v;
    pos[i] = p;
/*
    float colx = v.x;
    float coly = v.y;
    float colz = v.z;
    if(colx < 0) {colx = -1.0f*colx;}
    if(colx > 1) {colx = 1.0f;}
    if(coly < 0) {coly = -1.0f*coly;}
    if(coly > 1) {coly = 1.0f;}
    if(colz < 0) {colz = -1.0f*colz;}
    if(colz > 1) {colz = 1.0f;}

    colors[i].x = colx;
    colors[i].y = coly;
    colors[i].z = colz;
*/
    float mv = length(v);
    colors[i] = (float4)(mv,1.0f/mv,0.0f,colors[i].w);


}



