#ifndef RTPS_OUTER_COLLISION_CLOUD_H_INCLUDED
#define RTPS_OUTER_COLLISION_CLOUD_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class CollisionCloud
    {
        public:
            CollisionCloud() { cli = NULL; timer = NULL; };
            CollisionCloud(std::string path, CL* cli, EB::Timer* timer, int max_points);
            void execute(int num, int num_pts_cloud, 
                        Buffer<float4>& pos_s, 
                        Buffer<float4>& cloud_pos_s, 
                        Buffer<float4>& cloud_normals_s, 
                        Buffer<float4>& force_s, 
            			//output
            			Buffer<unsigned int>& ci_start,
            			Buffer<unsigned int>& ci_end,
                        //params
                        Buffer<SPHParams>& sphp,
                        Buffer<GridParams>& gp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            

        private:
            CL* cli;
            Kernel k_collision_cloud;
            EB::Timer* timer;
    };
}

#endif
