#include "../OUTER.h"

namespace rtps
{

    void OUTER::cpuXOUTER()
    {
		#if 0

        float scale = sphp.simulation_scale;
        float h = sphp.smoothing_distance;

        float h9 = h*h*h * h*h*h * h*h*h;
        float alpha = 315.f / 64.0f / sphp.PI / h9;

        float4* vel;
        if (integrator == EULER)
        {
            vel = &velocities[0];
        }
        else if (integrator == LEAPFROG)
        {
            vel = &veleval[0];
        }

        for (int i = 0; i < num; i++)
        {

            float4 p = positions[i];
            float4 v = vel[i];
            p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);
            //v = float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);

            float4 f = float4(0.0f, 0.0f, 0.0f, 0.0f);

            //stuff from Tim's code (need to match #s to papers)
            //float alpha = 315.f/208.f/sphp->PI/h/h/h;

            for (int j = 0; j < num; j++)
            {
                if (j == i) continue;
                float4 pj = positions[j];
                float4 vj = vel[j];
                pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
                //vj = float4(vj.x * scale, vj.y * scale, vj.z * scale, vj.w * scale);
                float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);

                float rlen = magnitude(r);
                if (rlen < h)
                {
                    float r2 = rlen*rlen;
                    float re2 = h*h;
                    if (r2/re2 <= 4.f)
                    {
                        //float R = sqrt(r2/re2);
                        //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                        //float Wij = Wpoly6(r, h);
                        float hr2 = (h*h - dist_squared(r));
                        float Wij = alpha * hr2*hr2*hr2;
                        float fcoeff = 2.0 * sphp.mass * Wij  / (densities[j] + densities[i]);
                        f.x += fcoeff * (vj.x - v.x); 
                        f.y += fcoeff * (vj.y - v.y); 
                        f.z += fcoeff * (vj.z - v.z); 
                    }

                }
            }
            xsphs[i].x = f.x;
            xsphs[i].y = f.y;
            xsphs[i].z = f.z;
        }
		#endif

    }

}
