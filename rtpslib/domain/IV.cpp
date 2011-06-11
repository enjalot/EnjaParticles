#include "IV.h"
#include <vector>

//for random
#include<stdlib.h>
#include<time.h>

namespace rtps
{

    std::vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale)
    {
        /*!
         * Create a rectangle with at most num particles in it.
         *  The size of the return vector will be the actual number of particles used to fill the rectangle
         */
        spacing *= 1.1f;

        float xmin = min.x / scale;
        float xmax = max.x / scale;
        float ymin = min.y / scale;
        float ymax = max.y / scale;
        float zmin = min.z / scale;
        float zmax = max.z / scale;

        std::vector<float4> rvec(num);
        int i=0;
        for (float z = zmin; z <= zmax; z+=spacing)
        {
            for (float y = ymin; y <= ymax; y+=spacing)
            {
                for (float x = xmin; x <= xmax; x+=spacing)
                {
                    if (i >= num) break;
                    rvec[i] = float4(x,y,z,1.0f);
                    i++;
                }
            }
        }
        rvec.resize(i);
        return rvec;

    }

    void addCube(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& rvec)
    {
    /*!
     * Create a rectangle with at most num particles in it.
     *  The size of the return vector will be the actual number of particles used to fill the rectangle
     */
        float xmin = min.x / scale;
        float xmax = max.x / scale;
        float ymin = min.y / scale;
        float ymax = max.y / scale;
        float zmin = min.z / scale;
        float zmax = max.z / scale;

        rvec.resize(num);

        int i=0;
        for (float z = zmin; z <= zmax+.0*(zmax-zmin); z+=spacing) {
        for (float y = ymin; y <= ymax+.0*(ymax-ymin); y+=spacing) {
        for (float x = xmin; x <= xmax+.0*(xmax-xmin); x+=spacing) {
            if (i >= num) break;				
            rvec[i] = float4(x,y,z,1.0f);
    //        rvec[i].print("pos[i]");
            i++;
        }}
        }
        rvec.resize(i);

    }

	//----------------------------------------------------------------------
    std::vector<float4> addSphere(int num, float4 center, float radius, float spacing, float scale)
    {
        /*!
         * Create a sphere with at most num particles in it.
         *  The size of the return vector will be the actual number of particles used to fill the rectangle
         */
        spacing *= 1.9f;
        float xmin = (center.x - radius) / scale;
        float xmax = (center.x + radius) / scale;
        float ymin = (center.y - radius) / scale;
        float ymax = (center.y + radius) / scale;
        float zmin = (center.z - radius) / scale;
        float zmax = (center.z + radius) / scale;
        float r2 = radius*radius;
        float d2 = 0.0f;

        std::vector<float4> rvec(num);
        int i=0;
        for (float z = zmin; z <= zmax; z+=spacing)
        {
            for (float y = ymin; y <= ymax; y+=spacing)
            {
                for (float x = xmin; x <= xmax; x+=spacing)
                {
                    if (i >= num) break;
                    d2 = (x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) + (z - center.z)*(z - center.z);
                    if (d2 > r2) continue;
                    rvec[i] = float4(x,y,z,1.0f);
                    i++;
                }
            }
        }
        rvec.resize(i);
        return rvec;


    }
	//----------------------------------------------------------------------


    std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing)
    {

        printf("num: %d\n", num);
        spacing *= 1.999f; //should probably just figure out whats up with my spacing
        printf("spacing: %f\n", spacing);

        float4 umin = -radius*u;
        float4 vmin = -radius*v;
        printf("u %f %f %f %f\n", u.x, u.y, u.z, u.w);
        printf("v %f %f %f %f\n", v.x, v.y, v.z, v.w);
        printf("umin %f %f %f %f\n", umin.x, umin.y, umin.z, umin.w);
        printf("vmin %f %f %f %f\n", vmin.x, vmin.y, vmin.z, vmin.w);

        std::vector<float4> rvec;
        int i = 0;
        float d2 = 0.;
        float r2 = radius*radius;
        for (float du = 0.; du < 2.*radius; du += spacing)
        {
            for (float dv = 0.; dv < 2.*radius; dv += spacing)
            {
                if (i >= num) break;
                float4 part = center + umin + u*du + vmin + v*dv;
                part.w = 1.0f;
                printf("part %f %f %f %f\n", part.x, part.y, part.z, part.w);
                d2 = dist_squared(part-center);
                printf("d2: %f, r2: %f\n", d2, r2);
                if (d2 < r2)
                {
                    rvec.push_back(part);
                    i++;
                }
            }
        }
        return rvec;

    }

    std::vector<float4> addDiscRandom(int num, float4 center, float4 v, float4 u, float4 w, float radius, float spacing)
    {
        /*
         * randomly space the particles on a grid, rather than evenly
         * randomize the velocity within some bounds 
         * so each particle needs its own velocity
         */

        //seed random
        //srand ( time(NULL) );

        printf("num: %d\n", num);
        spacing *= 1.1f; //should probably just figure out whats up with my spacing
        printf("spacing: %f\n", spacing);
        float pert = .1f*spacing;   //amount of perterbation
        float vpert = 100.f;

        float4 umin = -radius*u;
        float4 wmin = -radius*w;
        /*
        printf("u %f %f %f %f\n", u.x, u.y, u.z, u.w);
        printf("v %f %f %f %f\n", v.x, v.y, v.z, v.w);
        printf("umin %f %f %f %f\n", umin.x, umin.y, umin.z, umin.w);
        printf("vmin %f %f %f %f\n", vmin.x, vmin.y, vmin.z, vmin.w);
        */

        std::vector<float4> rvec;
        int i = 0;
        float d2 = 0.;
        float r2 = radius*radius;
        for (float du = 0.; du < 2.*radius; du += spacing)
        {
            for (float dw = 0.; dw < 2.*radius; dw += spacing)
            {
                if (i >= num) break;
                float rrv = rand()*vpert*2./RAND_MAX - vpert;   //random number between -pert and pert
                float rru = rand()*pert*2./RAND_MAX - pert;   //random number between -pert and pert
                float rrw = rand()*pert*2./RAND_MAX - pert;   //random number between -pert and pert
                du += rru;
                dw += rrw;
                float4 part = center + umin + u*du + wmin + w*dw + rrv*v;
                part.w = 1.0f;
                //printf("part %f %f %f %f\n", part.x, part.y, part.z, part.w);
                d2 = dist_squared(part-center);
                //printf("d2: %f, r2: %f\n", d2, r2);
                if (d2 < r2)
                {
                    rvec.push_back(part);
                    i++;
                }
            }
        }
        return rvec;

    }



    std::vector<float4> addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax)
    {
    /*!
     * Create a rectangle with at most num particles in it.
     *  The size of the return vector will be the actual number of particles used to fill the rectangle
     */

        srand(time(NULL));	

        printf("random num: %f\n", rand()/(float) RAND_MAX);
        spacing *= 1.1f;
    min.print("Box min: ");
    max.print("Box max: ");
        float xmin = min.x  / scale;
        float xmax = max.x  / scale;
        float ymin = min.y  / scale;
        float ymax = max.y  / scale;
        float zmin = min.z  / scale;
        float zmax = max.z  / scale;

        std::vector<float4> rvec(num);
        int i=0;
        for (float z = zmin; z <= zmax; z+=spacing) {
        for (float y = ymin; y <= ymax; y+=spacing) {
        for (float x = xmin; x <= xmax; x+=spacing) {
            if (i >= num) break;	
    //printf("adding particles: %f, %f, %f\n", x, y, z);			
            rvec[i] = float4(x-rand()/RAND_MAX,y-rand()/RAND_MAX,z-rand()/RAND_MAX,1.0f);


            i++;
        }}}
        rvec.resize(i);
        return rvec;

    }


    std::vector<float4> addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax)
    {
    /*!
     * Create a rectangle with at most num particles in it.
     *  The size of the return vector will be the actual number of particles used to fill the rectangle
     */
        srand(time(NULL));	
        
        spacing *= 1.1;

        float xmin = (center.x - radius)  / scale;
        float xmax = (center.x + radius)  / scale;
        float ymin = (center.y - radius)  / scale;
        float ymax = (center.y + radius)  / scale;
        float zmin = (center.z - radius)  / scale;
        float zmax = (center.z + radius)  / scale;
        
        float r2 = radius*radius;
        float d2 = 0.0f;

        std::vector<float4> rvec(num);
        int i=0;
        for (float z = zmin; z <= zmax; z+=spacing) {
        for (float y = ymin; y <= ymax; y+=spacing) {
        for (float x = xmin; x <= xmax; x+=spacing) {
            if (i >= num) break;				
            d2 = (x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) + (z - center.z)*(z - center.z);
            if(d2 > r2) continue;
            rvec[i] = float4(x-rand()/RAND_MAX,y-rand()/RAND_MAX,z-rand()/RAND_MAX,1.0f);
            i++;
        }}}
        rvec.resize(i);
        return rvec;
    }

    std::vector<float4> addRandArrangement(int num, float scale, float4 dmin, float4 dmax)
    {

            srand(time(NULL));	
        
            std::vector<float4> rvec(num);
        int i=0;

        for(int z=dmin.z/scale; z <= dmax.z/scale; z+=0.3){
        for(int y=dmin.y/scale; y <= dmax.y/scale; y+=0.3){
        for(int x=dmin.x/scale; x <= dmax.x/scale; x+=0.3){
            if(i >= num) break;
            rvec[i] = float4(rand()/RAND_MAX,rand()/RAND_MAX,rand()/RAND_MAX,1.0f);
            i++;
        }}}
        rvec.resize(i);
        return rvec;
    }

//----------------------------------------------------------------------
	std::vector<float4> addHollowSphere(int num, float4 center, float radius_in, float radius_out, float spacing, float scale, std::vector<float4>& normals)
	{
        spacing *= 1.9f;
        float xmin = (center.x - radius_out) / scale;
        float xmax = (center.x + radius_out) / scale;
        float ymin = (center.y - radius_out) / scale;
        float ymax = (center.y + radius_out) / scale;
        float zmin = (center.z - radius_out) / scale;
        float zmax = (center.z + radius_out) / scale;
        float r2in  = radius_in  * radius_in;
        float r2out = radius_out * radius_out;
        float d2 = 0.0f;

        std::vector<float4> rvec; // num
        //std::vector<float4> nvec; //(num);
		int i=0;

        for (float z = zmin; z <= zmax; z+=spacing)
        {
            for (float y = ymin; y <= ymax; y+=spacing)
            {
                for (float x = xmin; x <= xmax; x+=spacing)
                {
                    if (i >= num) break;
					float4 n(x-center.x, y-center.y, z-center.z, 0.);
                    d2 = (x - center.x)*(x - center.x) + (y - center.y)*(y - center.y) + (z - center.z)*(z - center.z);

					// ideally, replace r2out by r2out-spacing/2 to take radius of particle into 
					// account. Can fix later when we are refining. 
                    if (d2 > r2out || d2 < r2in) continue;
					float4 r(x,y,z,1.0f);
					float sqi = sqrt(1. / (n.x*n.x + n.y*n.y + n.z*n.z));
					n.x *= sqi;
					n.y *= sqi;
					n.z *= sqi;
					rvec.push_back(r);
					normals.push_back(n);
					i++;
                }
            }
        }
        return rvec;


		;
	}
//----------------------------------------------------------------------



}

