
#ifndef _POLAR_3D_KERNEL_H
#define _POLAR_3D_KERNEL_H


// directives in Makefile are not available if OpenCL file is compiled within 
// the cpp code. 
#define OPENCL

#ifdef OPENCL
 #define ARGS_OCL   , nx, ny, nz, nb_zblocks, bx, cx, zblocksz
 #define ARGS_OTE   , KM_t, KP_t, LM_t, LP_t, MM_t, MP_t 
 #define ARGS_CL    , int nx, int ny, int nz, int nb_zblocks, int bx, int cx, int zblocksz
 #define ARGS_TE    , __global float* KM_t, __global float* KP_t, __global float* LM_t, __global float* LP_t, __global float* MM_t, __global float* MP_t 
#else
 #ifdef CUDA
  #define ARGS_OCL  
  #define ARGS_OTE 
  #define ARGS_CL 
  #define ARGS_TE 
 #endif
#endif



#ifdef DEBUG
// should not need ARGS_TE on next line
#define ARGS_GLOB   ARGS_CL  ARGS_TE,  float4* deb
#define ARGS_DUMMY  ARGS_CL  ARGS_TE,  float4* deb
#define ARGS_REAL   ARGS_OCL ARGS_OTE, deb
#else
// should not need ARGS_TE on next line
#define ARGS_GLOB   ARGS_CL  ARGS_TE
#define ARGS_DUMMY  ARGS_CL   ARGS_TE
#define ARGS_REAL   ARGS_OCL  ARGS_OTE
#endif

#if 0
#define ARGS_GLOB
#define ARGS_DUMMY
#define ARGS_REAL
#define ARGS_OCL
#define ARGS_CL
#endif


//#define blkDim blockDim


//----------------------------------------------------------------------

// setup a texture for epsilon

#if 1
void read_row(int row, __global float* arr, __local float* sh ARGS_CL );
void read_row_polar_3d(int row, __global float* arr, __local float* sh ARGS_CL);
void read_plane_polar_3d(int row, __global float* arr, __local float* sha, int zblock ARGS_CL);
void copy_two_z_planes(__local float* sh ARGS_CL);


void matrix_vec_one_z_block(__global float* x, __global float* y, int zblock, 
							__local float* sh ARGS_DUMMY);
#endif

void matrix_vec_all_z_blocks(__global float* x, __global float* y, 
							 __local float* sh ARGS_DUMMY);

//----------------------------------------------------------------------
#if 0
__kernel void tst_mul(__global float* x, __global float* y)
//__kernel void tst_mul(__global float* x, __global float* y, __local float* sh)
{
	//x[0] = 3.;

	return;
}
#endif
//----------------------------------------------------------------------

#if 1
__kernel void matrix_vec_polar_3d(__global float* x, __global float* y, __local float* sh ARGS_GLOB)
//__kernel void matrix_vec_polar_3d(__global float* x, __global float* y  ARGS_GLOB)
{
	//sh[3] = 4.;

	//return;

	matrix_vec_all_z_blocks(x, y, sh ARGS_REAL);
}
#endif



// DEBUG

//----------------------------------------------------------------------
// I must handle zblocks myself
void matrix_vec_all_z_blocks(__global float* x, __global float* y, 
									  __local float* sh ARGS_DUMMY)
{
	#if 1
	for (int i_block=0; i_block < nb_zblocks; i_block++) {
		matrix_vec_one_z_block(x, y, i_block, sh ARGS_REAL);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	#endif
}
//----------------------------------------------------------------------
#if 1
void matrix_vec_one_z_block(__global float* x, __global float* y, 
								int zblock, __local float* sh ARGS_DUMMY)
{
#if 1
	#define VOLATILE volatile
#else
	#define VOLATILE
#endif

	// reduce register count to 30 by using volatile and moving these
	// higher up in the code
	VOLATILE int xi = (get_group_id(0) * get_local_size(0)) + get_local_id(0);
	VOLATILE int yi = (get_group_id(1) * get_local_size(1)) + get_local_id(1);
	VOLATILE int zi = (zblock     * get_local_size(2)) + get_local_id(2);

	barrier(CLK_LOCAL_MEM_FENCE);

// GE: March 26, 2009
// initialize shared memory to zero (one block of floats)
// size: is (block.x+2)*(block.y+2)*(block.z+2)

        int nbt = get_local_id(0)*get_local_id(1)*get_local_id(2);
        barrier(CLK_LOCAL_MEM_FENCE);

	// solution x is array of size (nx+2)x(ny+2)x(nz+2) (includes ghost points)
	// Not really necessary to include ghost points

	// read (blkDim.z+2) planes (includes two boundary planes to store in shared memory)
	// currently, blkDim.z = 2

	// only works with two threads in z

	if (zblock == 0) {
		// boundary planes not stored in x
		// read planes 0, 1 of x, store in planes 0,1 of sh
		// 1st arg: 0, 1
		read_plane_polar_3d(get_local_id(2), x, sh, zblock ARGS_OCL);
	} else {
		copy_two_z_planes(sh ARGS_OCL);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(2) < 2) {
		// read planes blkDim.z+0, blkDim.z+1
		// 1st arg: 2, 3
		read_plane_polar_3d(get_local_id(2)+get_local_size(2), x, sh, zblock ARGS_OCL);
	}

	barrier(CLK_LOCAL_MEM_FENCE);


// avoid use of registers

#define i get_local_id(0)
#define j get_local_id(1)
#define k get_local_id(2)

	// recall: size of shx is (blkDim.x+2) x (blkDim.x + 2)

	// orig
	int ii = (get_local_id(0)+1)+bx*(get_local_id(1)+1)+cx*(get_local_id(2)+1); // move away from the border

	// +1,-1: radial direction
	// i+bx,i-bx: theta direction (E is already periodic in theta)
	// i+cx,i-cx: z direction 
	// Must make shared memory periodic in theta when global row = -1 and ny

	barrier(CLK_LOCAL_MEM_FENCE);

// 3D case
	volatile float a;

	float kp  = KP_t[xi];
	float km  = KM_t[xi];
	float lp  = LP_t[yi];
	float lm  = LM_t[yi];
	float mp  = MP_t[zi];
	float mm  = MM_t[zi];

	//mm = mp = lp = lm = kp = km = 1.0;

	// error is related to shared memory

	a =     kp*(sh[ii+1]-sh[ii])  - km*(sh[ii]-sh[ii-1]);
	a = a + lp*(sh[ii+bx]-sh[ii]) - lm*(sh[ii]-sh[ii-bx]);
	a = a + mp*(sh[ii+cx]-sh[ii]) - mm*(sh[ii]-sh[ii-cx]);

	// ERROR RELATED to z direction (?? not sure)
	//a =  mp*(sh[ii+cx]-sh[ii]);  // Creates bad norm!!!
	//a =  mp*sh[ii+cx]);  // Creates bad norm!!!
	//a =  sh[ii+cx];  // (norm = 1.4)
	//a =  mp;   // (norm = 270000)

	//a = kp + km + mp + mm + lp + mm;
//	a = sh[ii+1] + sh[ii-1] + sh[ii+bx] + sh[ii-bx] + sh[ii-cx] + sh[ii+cx];


// DIRICHLET CONDITIONS
//   xi=nx-1 is the last point. The value of xi(nx+nx*j) is assumed to be given

	barrier(CLK_LOCAL_MEM_FENCE);

// correct statement

	int glob = xi+nx*(yi+ny*zi);

	y[glob] = a; // correct version

#ifdef DEBUG
	#if 1
		//float a2 = k0*(e011+e010+e001+e000)*(sh[ii-1]-sh[ii]);
		#if 1
		deb[glob].x = sh[ii]; //zi; //y[glob]; //sh[ii]; //zblock; //sh[ii];
		deb[glob].y = sh[ii+1]; //sh[ii]; //zblock; //sh[ii];
		deb[glob].z = sh[ii+bx]; //sh[ii+cx]; //zblock; //sh[ii];
		deb[glob].w = cx; //sh[ii+cx]; //sh[ii-cx]; //zblock; //sh[ii];
		#endif
	#endif 
#endif  


#undef i
#undef j
#undef k
#undef VOLATILE
}
#endif
//----------------------------------------------------------------------
#if 1
void read_plane_polar_3d(int plane, __global float* sol, 
						__local float* sha, int zblock ARGS_CL)
{
#if 0
// plane = (0,1) (2,3)
// each row is a row in y (a radial slice)
// plane in [0,blkDim.z + 1] = [0,blkDim.z-1] + [0,1]  (0,1,2,3)

// CONSTANTS: bx, cx, nx,ny, nz
#endif

/* glob_plane_offset = [-1,0,...,nz] (includes additional boundary planes) */
/*    for access into solution array (sol)  */
	volatile int glob_plane_offset = plane-1 + zblock*get_local_size(2);


	/* volatile float* not allowed */

	/* offset into shared memory array */
	int loc_plane_offset = cx * plane;

	__local float* sha_loc = sha+loc_plane_offset;

	/* NEW 5/3/2010 */
	/* if (glob_plane_offset < 0 || glob_plane_offset == nz) { */
	/* I should set the shared memory to zero in these two planes */

	if (glob_plane_offset < 0) {
		sha_loc[1+get_local_id(0)+bx*(1+get_local_id(1))] = 0.; /* Dirichlet BC */
		return;
	}
	if (glob_plane_offset == nz) {
		sha_loc[1+get_local_id(0)+bx*(1+get_local_id(1))] = 0.; /* Dirichlet BC */
		return;
	}

	/* assumes solution array of size nx*ny*nz (does not include boundary points with Dirichlet BC) */
	glob_plane_offset = glob_plane_offset*nx*ny;
	__global float* sol_glob = sol+glob_plane_offset;

	barrier(CLK_LOCAL_MEM_FENCE);
	/* get_local_id(1) == 0 maps to row -1 (ghost row) */
	/* get_local_id(1) == 1 maps to row 0 */
	read_row_polar_3d(get_local_id(1), sol_glob, sha_loc ARGS_OCL);
	barrier(CLK_LOCAL_MEM_FENCE);

	/* What is this for? y direction is theta (capacitance code) */
	/* This reads for get_local_id(1) == (0,1) */
	if (get_local_id(1) < 2) {
		/* get_local_id(1) == 0 maps to row ny-1 */
		/* get_local_id(1) == 1 maps to row ny (ghost row) */
		read_row_polar_3d(get_local_id(1)+get_local_size(1), sol_glob, sha_loc ARGS_OCL);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
#endif
//----------------------------------------------------------------------
#if 1
void read_row_polar_3d(int row, __global float* arr, __local float* sha ARGS_CL)
{
// each row is a row in y (a radial slice)
// row in [0,blkDim. + 1] = [0,...,blkDim.-1] + [0,1]
// (x) width of shared memory block: cx  (xy plane skip factor)
// (y) height of shared memory block: bx (x row skip factor)
// arr: solution array (nx+2) x (ny+2) x (nz+2)

	volatile int loc_row;

	// glob_row = -1,0,...,ny-1,ny (includes to rows beyond block)
	volatile int glob_row = row-1 + get_local_size(1) * get_group_id(1);  // blocks form partition of arr

	loc_row = bx * row;  // start of row in shared memory (row is along x)
	__local float* sha_loc = sha + loc_row; 

	volatile int x = get_local_id(0) + get_group_id(0) * get_local_size(0); // index within block

#define SHA(tid)  sha[1+(tid)+(loc_row)]
#undef SHA

//   first column of shared block is r=-0.5*delta_r
//    last column of shared block is r=rmax+0.5*delta_r (rmax=7)
//   shared memory cells correspond to cell centers

	#if 1
	if ((glob_row < 0)) {
		sha_loc[1+get_local_id(0)] = 0.; // Dirichlet BC
		return;
	}

	if ((glob_row == ny)) {
		sha_loc[1+get_local_id(0)] = 0.; // Dirichlet BC
		return;
	}
	#endif

	// interior radial line
	//volatile int x = get_local_id(0) + get_group_id(0) * get_local_size(0); // index within block

	//volatile int nx_glob = nx * glob_row; // offset into 3D array
	loc_row = bx * row;
	// I'd like sol_row to be a register variable, even though arr is a pointer
	// to a global array. Otherwise there is a needless global access
	// The __global on the next line is INEFFICIENT!!
	__global float* sol_row = arr + nx*glob_row;

	// interior points on the row
	// Something wrong? (2,2,...) has wrong value
	// if 1+thread -> 0+thread, (0,2) has wrong value
	sha_loc[1+get_local_id(0)] = sol_row[x];   //<<< incoherent loads!! WHY?

	// Now handle first and last points on the row (Dirichlet BC)

	// make sure indices are positive
	// Choose single thread for this operation)

	#if 1
	// r=0 (center)
	if (get_local_id(0) == 15) {
		if (get_group_id(0) != 0) {
			// first point, not but first block
			sha_loc[0] = sol_row[get_group_id(0)*get_local_size(0) - 1]; // index must be > 0
		} else {
			sha_loc[0] = 0.0; // Dirichlet BC (first point on row and first block)
		}
	}
	#endif

	//barrier(CLK_LOCAL_MEM_FENCE);

	// Choose single thread for this operation)

	// This if statement only works if more than 16 threads in x block direction
	// incoherent
	#if 1
	// r=rmax (if polar coord)  or Dirichlet BC (Cartesian)
	if (get_local_id(0) == 0) {  // any other threadIdx gives incoherent loads
		//loc_row = bx * row;
		// this part is ok it seems
		// Last point, but not last block
		if (get_group_id(0) != (get_num_groups(0)-1) ) {
			sha_loc[1+get_local_size(0)] = sol_row[get_group_id(0)*get_local_size(0) + get_local_size(0)]; // index must be < nx
		} else {
			// last point in last block
			sha_loc[1+get_local_size(0)] = 0.0; 
		}
	}
	#endif
}
#endif
//----------------------------------------------------------------------
// should be much faster than getting data from device memory
// copy planes k+2,k+3 in sh, to planes k+0,k+1
#if 1
void copy_two_z_planes(__local float* sh ARGS_CL)
{
/* CONSTANTS: bx, cx */

	volatile int base = get_local_id(0) + bx * get_local_id(1) + cx * get_local_id(2);
	sh[base] = sh[base + cx*2]; // planes 0 <--> 2

	// But the base has size (get_local_size(0) + 2) in the x direction

	// last two radial points
	if (get_local_id(0) < 2) {
		base = get_local_size(0) + get_local_id(0) + bx * get_local_id(1) + cx * get_local_id(2);
		sh[base] = sh[base + cx*2]; // planes 0 <--> 2
	}

	//last two points in y

	barrier(CLK_LOCAL_MEM_FENCE);

	base = get_local_id(0) + bx * (2+get_local_id(1)) + cx * get_local_id(2);
	sh[base] = sh[base + cx*2]; // planes 0 <--> 2

	// But the base has size (get_local_size(0) + 2) in the x direction

	// last two radial points
	if (get_local_id(0) < 2) {
		base = get_local_size(0) + get_local_id(0) + bx * (2+get_local_id(1)) + cx * get_local_id(2);
		sh[base] = sh[base + cx*2]; // planes 0 <--> 2
	}
}
//----------------------------------------------------------------------

// DEBUG
#endif




#undef VOLATILE
// #ifndef _POLAR_3D_KERNEL_H
#endif



