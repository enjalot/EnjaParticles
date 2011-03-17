
float length3(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}  
float4 normalize3(float4 vec)
{
    float4 retv;
    float magi = length3(vec);
    magi = magi < 1.e-8 ? 1. : 1./magi;
    retv.xyz = vec.xyz * magi;
    retv.w = vec.w;
    return retv;
}

__kernel void rules1(__global float4* pos, 
                     __global float4* color, 
                     __global float4* vel, 
                     __global float4* acc, 
                     __global float4* steer, 
                     __global float4* avg_pos, 
                     __global float4* avg_vel, 
                     float dt, 
                     float dim)
{
    //first pass to collect average values
    //get our index in the array
    unsigned int i = get_global_id(0);
    unsigned int num = get_global_size(0);

    //HARDCODED
    float h = 30.;
    float desired_sep = 20.;


    float4 p = pos[i];
    float4 v = vel[i];

    steer[i] = (float4)(0.,0.,0.,0.);
    avg_pos[i] = (float4)(0.,0.,0.,0.);
    avg_vel[i] = (float4)(0.,0.,0.,0.);
    //brute force
    for(int j = 0; j < num; j++)
    {
        float4 diff = p - pos[j];
        float d = length3(diff);
        if(d < h)
        {
            avg_pos[i].xyz += pos[j].xyz;
            avg_pos[i].w += 1;
            avg_vel[i].xyz += vel[j].xyz;
            avg_vel[i].w += 1;

            if(d < desired_sep && d != 0.)
            {
                //diff = normalize3(diff);
                diff /= d*d;    //normalized then divided by
                steer[i].xyz += diff.xyz;
                steer[i].w += 1;
            }
        }
    }
    

}

__kernel void rules2(__global float4* pos, 
                     __global float4* color, 
                     __global float4* vel, 
                     __global float4* acc, 
                     __global float4* steer, 
                     __global float4* avg_pos, 
                     __global float4* avg_vel, 
                     float dt, 
                     float dim)
{
    //second pass to compute rules
    //get our index in the array
    unsigned int i = get_global_id(0);
    unsigned int num = get_global_size(0);

    //HARDCODED
    float wcoh = 0.03f;
    float wsep = 0.3f;
    float walign = 0.3f;
    float MAX_SPEED = 3.f;


    float4 p = pos[i];
    float4 v = vel[i];

    steer[i].xyz /= steer[i].w > 0.f ? steer[i].w : 1.f;    //take the average
    avg_pos[i].xyz /= avg_pos[i].w > 0.f ? avg_pos[i].w : 1.f;    //take the average
    avg_vel[i].xyz /= avg_vel[i].w > 0.f ? avg_vel[i].w : 1.f;    //take the average

    float4 sep = normalize3(steer[i]);

    float4 coh = avg_pos[i] - p;
    coh = normalize3(coh);

    float4 align = avg_vel[i] - v;
    align = normalize3(align);

    acc[i] += wcoh*coh +wsep*sep + walign*align;
    float acc_mag = length3(acc[i]);
    //acc[i].print("acc");
    float4 acc_norm = normalize3(acc[i]);
    // MAX_SPEED is crucial
    if (acc_mag > MAX_SPEED) { 
        acc[i] = acc_norm*MAX_SPEED;
    }
    acc[i].w = 1.f;

    //acc_mag = length3(acc[i]);
    //vel[i] += acc[i]*dt;

    //the right way?
    //vel[i] = acc[i];

    float4 vv = (float4)(-3.*pos[i].y, pos[i].x, 0, 0.);
	vv = vv*.01f;
	vel[i] = vv + acc[i];

    

}
