#ifndef RTPS_CLOUDDATASTRUCTURES_H_INCLUDED
#define RTPS_CLOUDDATASTRUCTURES_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class CloudDataStructures 
    {
        public:
            DataStructures() { cli = NULL; timer = NULL; };
            DataStructures(CL* cli, EB::Timer* timer);
            int execute(int num,
                    //input
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& normal_u,
                    Buffer<float4>& normal_s,
                    //Buffer<float4>& vel_u,
                    //Buffer<float4>& vel_s,
                    //Buffer<float4>& veleval_u,
                    //Buffer<float4>& veleval_s,

                    //Buffer<float4>& uvars, 
                    //Buffer<float4>& color_u,
                    //Buffer<float4>& svars, 
                    //Buffer<float4>& color_s,
                    //output
                    Buffer<unsigned int>& hashes,
                    Buffer<unsigned int>& indices,
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_stop,
                    //params
                    Buffer<SPHParams>& sphp,
                    Buffer<CLOUDParams>& cloudp,
                    Buffer<GridParams>& gp,
                    int nb_cells,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_cloud_datastructures;
            EB::Timer* timer;
    };
}

#endif
