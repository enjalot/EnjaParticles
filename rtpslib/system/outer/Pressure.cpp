#include "OUTER.h"
#include <math.h>



namespace rtps
{
    float OUTER::Wspiky(float4 r, float h)
    {
        float h6 = h*h*h * h*h*h;
        float alpha = -45.f/sphp.PI/h6;
        float rlen = magnitude(r);
        float hr2 = (h - rlen);
        float Wij = alpha * hr2*hr2/rlen;
        return Wij;
    }
    /*
        //stuff from Tim's code (need to match #s to papers)
        //float alpha = 315.f/208.f/sphp->PI/h/h/h;
        //float R = sqrt(r2/re2);
        //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
    
     */

    void OUTER::cpuPressure()
    {
		#if 0
        float scale = sphp.simulation_scale;
        float h = sphp.smoothing_distance;

        for (int i = 0; i < num; i++)
        {

            float4 p = positions[i];
            p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);

            float4 f = float4(0.0f, 0.0f, 0.0f, 0.0f);

            //super slow way, we need to use grid + sort method to get nearest neighbors
            //this code should never see the light of day on a GPU... just sayin
            for (int j = 0; j < num; j++)
            {
                if (j == i) continue;
                float4 pj = positions[j];

                pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
                float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);

                float rlen = magnitude(r);
                if (rlen < h)
                {
                    float r2 = rlen*rlen;
                    float re2 = h*h;
                    if (r2/re2 <= 4.f)
                    {
                        //from tim's code
                        /*
                        float Pi = 1.013E5*(pow(density[i]/1000.0f, 7.0f) - 1.0f);
                        float Pj = 1.013E5*(pow(density[j]/1000.0f, 7.0f) - 1.0f);
                        float kern = sphp->mass * Wij * (Pi + Pj) / (density[i] * density[j]);
                        */
                        //from simple OUTER in Krog's thesis
                        float Pi = sphp.K*(densities[i] - 1000.0f); //rest density
                        float Pj = sphp.K*(densities[j] - 1000.0f); //rest density
                        //float kern = sphp->mass * -1.0f * Wij * (Pi + Pj) / (2.0f * density[j]);
                        float Wij = Wspiky(r, h);
                        float kern = sphp.mass * -.5f * Wij * (Pi + Pj) / (densities[i] * densities[j]);
                        //float kern = sphp.mass * -.5f * Wij * (Pi/(densities[i]*densities[i]) + Pj/(densities[j]*densities[j]));
                        f.x += kern * r.x;
                        f.y += kern * r.y;
                        f.z += kern * r.z;
                    }

                }
            }
            //printf("forces[%d] = %f %f %f\n", i, f.x, f.y, f.z);
            forces[i] = f;

        }
		#endif
    }


}
