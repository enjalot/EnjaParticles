#ifndef RTPS_COLLISION_TRI_H_INCLUDED
#define RTPS_COLLISION_TRI_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class CollisionTriangle
    {
        public:
            CollisionTriangle() { cli = NULL; timer = NULL; };
            CollisionTriangle(std::string path, CL* cli, EB::Timer* timer, int max_triangles);
            void execute(int num,
                        float dt,
                        //input
                        //Buffer<float4>& svars, 
                        Buffer<float4>& pos_s, 
                        Buffer<float4>& vel_s, 
                        Buffer<float4>& force_s, 
                        //output
                        //params
                        Buffer<SPHParams>& sphp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            
           
        void loadTriangles(std::vector<Triangle> &triangles);

        private:
            CL* cli;
            Kernel k_collision_tri;
            EB::Timer* timer;
            Buffer<Triangle> cl_triangles;
            bool triangles_loaded; //keep track if we've loaded triangles yet
    };
}

#endif
