#ifndef RTPS_LIFETIME_H_INCLUDED
#define RTPS_LIFETIME_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class Lifetime
    {
        public:
            Lifetime() { cli = NULL; timer = NULL; };
            Lifetime(CL* cli, EB::Timer* timer, std::string filename);
            void execute(int num,
                    float dt,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& color_u, 
                    Buffer<float4>& color_s, 
                    Buffer<unsigned int>& indices,
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug
                );
            

        private:
            CL* cli;
            Kernel k_lifetime;
            EB::Timer* timer;
    };
}

#endif
