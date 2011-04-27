#ifndef RTPS_CELLINDICES_H_INCLUDED
#define RTPS_CELLINDICES_H_INCLUDED


#include <RTPS.h>
//#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class CellIndices 
    {
        public:
            CellIndices() { cli = NULL; timer = NULL; };
            CellIndices(std::string path, CL* cli, EB::Timer* timer);
            int execute(int num,
                    Buffer<unsigned int>& hashes,
                    Buffer<unsigned int>& indices,
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_stop,
                    //params
                    //Buffer<SPHParams>& sphp,
                    Buffer<GridParams>& gp,
                    int nb_cells,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_cellindices;
            EB::Timer* timer;
    };
}

#endif
