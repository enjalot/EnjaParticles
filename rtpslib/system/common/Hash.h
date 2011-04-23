#ifndef RTPS_HASH_H_INCLUDED
#define RTPS_HASH_H_INCLUDED


#include <RTPS.h>
//#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class Hash
    {
        public:
            Hash() { cli = NULL; timer = NULL; };
            Hash(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num,
                        //input
                        //Buffer<float4>& uvars, 
                        Buffer<float4>& pos_u, 
                        //output
                        Buffer<unsigned int>& hashes,
                        Buffer<unsigned int>& indices,
                        //params
                        //Buffer<SPHParams>& sphp,
                        Buffer<GridParams>& gp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            
           

        private:
            CL* cli;
            Kernel k_hash;
            EB::Timer* timer;
    };
}

#endif
