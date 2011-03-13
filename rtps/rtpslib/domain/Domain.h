#ifndef DOMAIN_H_INCLUDED
#define DOMAIN_H_INCLUDED

#include "../structs.h"
#include <vector>

namespace rtps {



class Domain
{
public:
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     */
    Domain(){};
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @param min 
     * @param max 
     */
    Domain(float4 min, float4 max);
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     */
    ~Domain();

    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @param cell_size 
     */
    void calculateCells(float cell_size);

    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getMin(){ return min; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getMax(){ return max; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getBndMin(){ return bnd_min; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getBndMax(){ return bnd_max; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getDelta() { return delta; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
    float4 getRes() { return res; };
    /**
     * 
     * 
     * @author andrew (3/13/2011)
     * 
     * @return float4 
     */
	float4 getSize() { return size; };


private:
    float4 min;
    float4 max; 
	float4 bnd_min;
	float4 bnd_max;

    float4 size;
    float4 res;
    float4 delta;

};

//-------------------------
// GORDON Datastructure for Grids. To be reconciled with Ian's
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;
    // particles stay within bnd
    float4          bnd_min; 
    float4          bnd_max;
    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;
    //float4          grid_inv_delta;
    // nb grid cells
	int 			nb_cells; 

	void print()
	{
		printf("\n----- GridParams ----\n");
		grid_size.print("grid_size"); 
		grid_min.print("grid_min"); 
		grid_max.print("grid_max"); 
		bnd_min.print("bnd_min"); 
		bnd_max.print("bnd_max"); 
		grid_res.print("grid_res"); 
		grid_delta.print("grid_delta"); 
		//grid_inv_delta.print("grid_inv_delta"); 
		printf("nb grid cells: %d\n", nb_cells);
	}
};





}
#endif
