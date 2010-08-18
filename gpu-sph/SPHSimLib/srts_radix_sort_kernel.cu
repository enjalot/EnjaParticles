/**
 * Copyright 2010 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 */


#ifndef _SRTS_RADIX_SORT_KERNEL_H_
#define _SRTS_RADIX_SORT_KERNEL_H_


#define SRTS_LOG_CYCLE_ELEMENTS		9
#define SRTS_CYCLE_ELEMENTS			(1 << SRTS_LOG_CYCLE_ELEMENTS)
#define SRTS_LOG_THREADS			7		// 128 threads
#define SRTS_THREADS				(1 << SRTS_LOG_THREADS)	
#define LOG_MEM_BANKS				4		// 5 for fermi
#define LOG_WARP_THREADS			5



//------------------------------------------------------------------------------
// Vector types
//------------------------------------------------------------------------------

template <typename K, int vec_elements> struct VecType;

template<>
struct VecType<short, 1>
{
	typedef short Type;
};

template<>
struct VecType<short, 2>
{
	typedef short2 Type;
};

template<>
struct VecType<short, 4>
{
	typedef short4 Type;
};


template<>
struct VecType<unsigned short, 1>
{
	typedef unsigned short Type;
};

template<>
struct VecType<unsigned short, 2>
{
	typedef ushort2 Type;
};

template<>
struct VecType<unsigned short, 4>
{
	typedef ushort4 Type;
};


template<>
struct VecType<int, 1>
{
	typedef int Type;
};

template<>
struct VecType<int, 2>
{
	typedef int2 Type;
};

template<>
struct VecType<int, 4>
{
	typedef int4 Type;
};


template<>
struct VecType<unsigned int, 1>
{
	typedef unsigned int Type;
};

template<>
struct VecType<unsigned int, 2>
{
	typedef uint2 Type;
};

template<>
struct VecType<unsigned int, 4>
{
	typedef uint4 Type;
};

template<>
struct VecType<float, 1>
{
	typedef float Type;
};

template<>
struct VecType<float, 2>
{
	typedef float2 Type;
};

template<>
struct VecType<float, 4>
{
	typedef float4 Type;
};


template<>
struct VecType<long, 1>
{
	typedef long Type;
};

template<>
struct VecType<long, 2>
{
	typedef long2 Type;
};

template<>
struct VecType<long, 4>
{
	typedef long4 Type;
};


template<>
struct VecType<unsigned long, 1>
{
	typedef unsigned long Type;
};

template<>
struct VecType<unsigned long, 2>
{
	typedef ulong2 Type;
};

template<>
struct VecType<unsigned long, 4>
{
	typedef ulong4 Type;
};



//------------------------------------------------------------------------------
// Inline Routines
//------------------------------------------------------------------------------

#define BYTE_ENCODE_SHIFT 	3u


template <typename K, unsigned long RADIX_DIGITS, unsigned int BIT>
__device__ inline unsigned int DecodeDigit(K key) 
{
	const K DIGIT_MASK1 = RADIX_DIGITS - 1;

	return (key & (DIGIT_MASK1 << BIT)) >> BIT;
}


template <typename K, unsigned long RADIX_DIGITS, unsigned int BIT>
__device__ inline void DecodeDigit(
	K key, 
	unsigned int &digit_group, 
	unsigned int &digit_shift) 
{
	const K DIGIT_MASK1 = RADIX_DIGITS - 1;
	const K DIGIT_MASK2 = (RADIX_DIGITS < 4) ? 0x1 : 0x3;

	digit_group = (key & (DIGIT_MASK1 << BIT)) >> (BIT + 2);

	// The stupid template generator causes warnings because it has 
	// not done dead-code-elimination by the time it checks for shift errors

	if (BIT > BYTE_ENCODE_SHIFT) {
		digit_shift = (key & (DIGIT_MASK2 << BIT)) >> (BIT - BYTE_ENCODE_SHIFT);
	} else {
		digit_shift = (key & (DIGIT_MASK2 << BIT)) << (BYTE_ENCODE_SHIFT - BIT);
	}
}


template <typename K, unsigned long RADIX_DIGITS, unsigned int BIT>
__device__ inline void DecodeDigit(
	K key, 
	unsigned int &digit, 
	unsigned int &digit_group, 
	unsigned int &digit_shift) 
{
	const K DIGIT_MASK1 = RADIX_DIGITS - 1;
	const K DIGIT_MASK2 = (RADIX_DIGITS < 4) ? 0x1 : 0x3;
	
	K masked_digit = key & (DIGIT_MASK1 << BIT);

	digit = masked_digit >> BIT;
	digit_group = masked_digit >> (BIT + 2);

	// The stupid template generator causes warnings because it has 
	// not done dead-code-elimination by the time it checks for shift errors

	if (BIT > BYTE_ENCODE_SHIFT) {
		digit_shift = (key & (DIGIT_MASK2 << BIT)) >> (BIT - BYTE_ENCODE_SHIFT);
	} else {
		digit_shift = (key & (DIGIT_MASK2 << BIT)) << (BYTE_ENCODE_SHIFT - BIT);
	}
}


__device__ inline unsigned int DecodeInt(unsigned int encoded, unsigned int digit_shift){
	return (encoded >> digit_shift) & 0xff;			// shift right 8 bits per digit and return rightmost 8 bits
}


__device__ inline unsigned int EncodeInt(unsigned int count, unsigned int digit_shift) {
	return count << digit_shift;					// shift left 8 bits per digit
}


__device__ inline unsigned int EncodeInt(unsigned int counts[4]) {
	
	unsigned int retval = counts[0];
	retval += counts[1] << 8;
	retval += counts[2] << 16;
	return retval += counts[3] << 24;
}


template <unsigned int NUM_ELEMENTS, unsigned int ACTIVE_THREADS> 
__device__ inline unsigned int WarpScan(
	volatile unsigned int warpscan[NUM_ELEMENTS * 2],
	unsigned int partial_reduction) {
	
	unsigned int warpscan_idx = NUM_ELEMENTS;
	if (ACTIVE_THREADS > NUM_ELEMENTS) {
		warpscan_idx += threadIdx.x & (NUM_ELEMENTS - 1);
	} else {
		warpscan_idx += threadIdx.x;
	}

	warpscan[warpscan_idx] = partial_reduction;

	if (NUM_ELEMENTS >= 2) warpscan[warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[warpscan_idx - 1];
	if (NUM_ELEMENTS >= 4) warpscan[warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[warpscan_idx - 2];
	if (NUM_ELEMENTS >= 8) warpscan[warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[warpscan_idx - 4];
	if (NUM_ELEMENTS >= 16) warpscan[warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[warpscan_idx - 8];
	if (NUM_ELEMENTS >= 32) warpscan[warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[warpscan_idx - 16];
	
	return warpscan[warpscan_idx - 1];
}




//------------------------------------------------------------------------------
// Dummy Kernels
//------------------------------------------------------------------------------

__global__ void DummyKernel()
{
}

__global__ void FlushKernel()
{
}



//------------------------------------------------------------------------------
// TreeReduce
//------------------------------------------------------------------------------

__device__ inline void Put128(
	unsigned int input,
	unsigned int smem_tree[129]) 
{
	smem_tree[threadIdx.x] = input;
}


__device__ inline void Reduce128(
	unsigned int smem_tree[129]) 
{
	smem_tree[threadIdx.x] = smem_tree[threadIdx.x] + smem_tree[threadIdx.x + 64]; 
}


__device__ inline void Reduce64(
	volatile unsigned int smem_tree[129]) 
{
	unsigned int partial_reduction = smem_tree[threadIdx.x];
	
	smem_tree[threadIdx.x] = partial_reduction = partial_reduction + smem_tree[threadIdx.x + 32];
	smem_tree[threadIdx.x] = partial_reduction = partial_reduction + smem_tree[threadIdx.x + 16];
	smem_tree[threadIdx.x] = partial_reduction = partial_reduction + smem_tree[threadIdx.x + 8];
	smem_tree[threadIdx.x] = partial_reduction = partial_reduction + smem_tree[threadIdx.x + 4];
	smem_tree[threadIdx.x] = partial_reduction = partial_reduction + smem_tree[threadIdx.x + 2];
	smem_tree[threadIdx.x] = partial_reduction = partial_reduction + smem_tree[threadIdx.x + 1];

}


__device__ inline void Reduce(
	unsigned int encoded_input,
	unsigned int smem_tree[4][129]) 
{
	Put128(DecodeInt(encoded_input, 0u << BYTE_ENCODE_SHIFT), smem_tree[0]);
	Put128(DecodeInt(encoded_input, 1u << BYTE_ENCODE_SHIFT), smem_tree[1]);
	Put128(DecodeInt(encoded_input, 2u << BYTE_ENCODE_SHIFT), smem_tree[2]);
	Put128(DecodeInt(encoded_input, 3u << BYTE_ENCODE_SHIFT), smem_tree[3]);

	__syncthreads();

	if (threadIdx.x < 64) { 
		Reduce128(smem_tree[0]);
		Reduce128(smem_tree[1]);
		Reduce128(smem_tree[2]);
		Reduce128(smem_tree[3]);
	}
		
	__syncthreads(); 

	if (threadIdx.x < 32) {
		Reduce64(smem_tree[0]);
		Reduce64(smem_tree[1]);
		Reduce64(smem_tree[2]);
		Reduce64(smem_tree[3]);
	}
}


template <unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void ReduceEncodedCounts(
	unsigned int smem_tree[4][129],
	unsigned int shared_counts[RADIX_DIGITS],
	unsigned int encoded_carry[FOURS_GROUPS]) 
{

	Reduce(encoded_carry[0], smem_tree);
	__syncthreads();
	if (threadIdx.x < 4) {
		shared_counts[threadIdx.x] += smem_tree[threadIdx.x][0];
	}
	__syncthreads();

	if (RADIX_DIGITS >= 8) {

		Reduce(encoded_carry[1], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 4] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();
	}
	
	if (RADIX_DIGITS >= 16) {

		Reduce(encoded_carry[2], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 8] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();

		Reduce(encoded_carry[3], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 12] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();
	}


	if (RADIX_DIGITS >= 32) {

		Reduce(encoded_carry[4], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 16] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();

		Reduce(encoded_carry[5], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 20] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();

		Reduce(encoded_carry[6], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 24] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();

		Reduce(encoded_carry[7], smem_tree);
		__syncthreads();
		if (threadIdx.x < 4) {
			shared_counts[threadIdx.x + 28] += smem_tree[threadIdx.x][0];
		}
		__syncthreads();
	}
}

	

template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Bucket(
	K input, 
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	unsigned int digit_group, digit_shift;
	DecodeDigit<K, RADIX_DIGITS, BIT>(input, digit_group, digit_shift);

	unsigned int encode = EncodeInt(1, digit_shift);

	#pragma unroll
	for (unsigned int fours_group = 0; fours_group < FOURS_GROUPS; fours_group++) {
		encoded_carry[fours_group] += (digit_group == fours_group) ? encode : 0;
	}
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle(
	K *d_in_keys,
	unsigned int offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	typename VecType<K, 2>::Type keys0, keys1;				

	if (RADIX_DIGITS >= 16) {

		typename VecType<K, 2>::Type *in2 = (typename VecType<K, 2>::Type *) &d_in_keys[offset];
		keys0 = in2[threadIdx.x];
		keys1 = in2[threadIdx.x + 128];

	} else {

		keys0.x = d_in_keys[offset + threadIdx.x];
		keys0.y = d_in_keys[offset + threadIdx.x + 128];
		keys1.x = d_in_keys[offset + threadIdx.x + 256];
		keys1.y = d_in_keys[offset + threadIdx.x + 384];

		__syncthreads();
	}

	Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(keys0.x, encoded_carry);
	Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(keys0.y, encoded_carry);
	Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(keys1.x, encoded_carry);
	Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(keys1.y, encoded_carry);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle2(
	K *d_in_keys,
	unsigned int offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset + SRTS_CYCLE_ELEMENTS, encoded_carry);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle4(
	K *d_in_keys,
	unsigned int offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle2<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle2<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset + (SRTS_CYCLE_ELEMENTS * 2), encoded_carry);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle8(
	K *d_in_keys,
	unsigned int offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle4<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle4<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset + (SRTS_CYCLE_ELEMENTS * 4), encoded_carry);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle12(
	K *d_in_keys,
	unsigned int &offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle8<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle4<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset + (SRTS_CYCLE_ELEMENTS * 8), encoded_carry);

	offset += (SRTS_CYCLE_ELEMENTS * 12);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle16(
	K *d_in_keys,
	unsigned int &offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle8<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle8<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset + (SRTS_CYCLE_ELEMENTS * 8), encoded_carry);

	offset += (SRTS_CYCLE_ELEMENTS * 16);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle32(
	K *d_in_keys,
	unsigned int &offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle16<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle16<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void Cycle60(
	K *d_in_keys,
	unsigned int &offset,
	unsigned int encoded_carry[FOURS_GROUPS]) 
{
	Cycle32<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle16<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
	Cycle12<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
}


template <typename K, unsigned int RADIX_DIGITS, unsigned int FOURS_GROUPS, unsigned int BIT>
__device__ inline void ProcessCycles(
	K *d_in_keys,
	unsigned int cycles,
	unsigned int &offset,
	unsigned int encoded_carry[FOURS_GROUPS],
	unsigned int smem_tree[4][129],
	unsigned int shared_counts[RADIX_DIGITS])
{
	// Process batches of 60 cycles (more than 63 would risk overflow of the 
	// encoded carry when reading 4 keys per cycle)
	while (cycles >= 60) {
		
		Cycle60<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		cycles -= 60;

		// reduce and reset encoded counters to prevent overflow
		ReduceEncodedCounts<RADIX_DIGITS, FOURS_GROUPS, BIT>(smem_tree, shared_counts, encoded_carry);
		
		#pragma unroll
		for (unsigned int fours_group = 0; fours_group < FOURS_GROUPS; fours_group++) {
			encoded_carry[fours_group] = 0;
		}
	} 

	// Wind down cycles in decreasing batch sizes
	if (cycles >= 32) {
		Cycle32<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		cycles -= 32;
	}
	if (cycles >= 16) {
		Cycle16<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		cycles -= 16;
	}
	if (cycles >= 8) {
		Cycle8<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		cycles -= 8;
		offset += (SRTS_CYCLE_ELEMENTS * 8);
	}
	if (cycles >= 4) {
		Cycle4<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		cycles -= 4;
		offset += (SRTS_CYCLE_ELEMENTS * 4);
	}
	if (cycles >= 2) {
		Cycle2<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		cycles -= 2;
		offset += (SRTS_CYCLE_ELEMENTS * 2);
	}
	if (cycles) {
		Cycle<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(d_in_keys, offset, encoded_carry);
		offset += SRTS_CYCLE_ELEMENTS;
	}
}


template <typename K, unsigned int RADIX_BITS, unsigned int BIT>
__global__ 
void TreeReduce(
	K *d_in_keys,
	unsigned int *d_spine,
	unsigned int num_big_blocks,
	unsigned int big_block_elements,
	unsigned int normal_block_elements, 
	unsigned int extra_elements_last_block)
{
	const unsigned int RADIX_DIGITS = 1 << RADIX_BITS;
	const unsigned int LOG_FOURS_GROUPS = (RADIX_BITS >= 2) ? RADIX_BITS - 2 : 0;	// always at least one fours group
	const unsigned int FOURS_GROUPS = 1 << LOG_FOURS_GROUPS;

	__shared__ unsigned int smem_tree[4][129];				// four counts encoded per group, 128 threads
	__shared__ unsigned int shared_counts[RADIX_DIGITS];

	unsigned int encoded_carry[FOURS_GROUPS];

	bool extra = (blockIdx.x == gridDim.x - 1) && (extra_elements_last_block);

	// calculate our threadblock's range
	unsigned int offset, block_elements;
	if (blockIdx.x < num_big_blocks) {
		offset = big_block_elements * blockIdx.x;
		block_elements = big_block_elements;
	} else {
		offset = (normal_block_elements * blockIdx.x) + (num_big_blocks * SRTS_CYCLE_ELEMENTS);
		block_elements = normal_block_elements;
	}
	if (extra) {
		block_elements -= SRTS_CYCLE_ELEMENTS;
	}			
	unsigned int cycles = block_elements >> SRTS_LOG_CYCLE_ELEMENTS;


	// Initialize digit counters
	if (threadIdx.x < RADIX_DIGITS) {
		shared_counts[threadIdx.x] = 0;
	}

	// Initialize encoded carry
	#pragma unroll
	for (unsigned int fours_group = 0; fours_group < FOURS_GROUPS; fours_group++) {
		encoded_carry[fours_group] = 0;
	}
	
	// Process cycles
	ProcessCycles<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(
		d_in_keys,
		cycles,
		offset,
		encoded_carry,
		smem_tree,
		shared_counts);

	// Cleanup if we're the last block
	if (extra) {
		
		if (threadIdx.x < extra_elements_last_block) {
			K key = d_in_keys[threadIdx.x + offset];
			Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(key, encoded_carry);
		}
		if (threadIdx.x + 128 < extra_elements_last_block) {
			K key = d_in_keys[threadIdx.x + offset + 128];
			Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(key, encoded_carry);
		}
		if (threadIdx.x + 256 < extra_elements_last_block) {
			K key = d_in_keys[threadIdx.x + offset + 256];
			Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(key, encoded_carry);
		}
		if (threadIdx.x + 384 < extra_elements_last_block) {
			K key = d_in_keys[threadIdx.x + offset + 384];
			Bucket<K, RADIX_DIGITS, FOURS_GROUPS, BIT>(key, encoded_carry);
		}
	}

	// Aggregate 
	ReduceEncodedCounts<RADIX_DIGITS, FOURS_GROUPS, BIT>(smem_tree, shared_counts, encoded_carry);

	// write carry in parallel 
	if (threadIdx.x < RADIX_DIGITS) {
		d_spine[(gridDim.x * threadIdx.x) + blockIdx.x] = shared_counts[threadIdx.x];
	}
} 






//------------------------------------------------------------------------------
// Serial Reduce & Scan Routines
//------------------------------------------------------------------------------


template <unsigned int LENGTH>
__device__ inline unsigned int 
SerialReduceSmem(unsigned int *smem_segment) {
	
	unsigned int reduce = smem_segment[0];

	#pragma unroll
	for (int i = 1; i < LENGTH; i++) {
		reduce += smem_segment[i];
	}
	
	return reduce;
}


template <unsigned int LENGTH>
__device__ inline
void SerialScanSmem(unsigned int *smem_segment, unsigned int seed0) {
	
	unsigned int seed1;

	#pragma unroll	
	for (int i = 0; i < LENGTH; i += 2) {
		seed1 = smem_segment[i] + seed0;
		smem_segment[i] = seed0;
		seed0 = seed1 + smem_segment[i + 1];
		smem_segment[i + 1] = seed1;
	}
}




//------------------------------------------------------------------------------
// SrtsScan
//------------------------------------------------------------------------------

template<
	unsigned int SMEM_ROWS,
	unsigned int ELTS_PER_ROW>
__device__ inline void SrtsScan512(
	unsigned int smem[SMEM_ROWS][ELTS_PER_ROW + 1],
	unsigned int *smem_offset,
	unsigned int *smem_segment,
	unsigned int warpscan[64],
	uint2 *in, 
	uint2 *out,
	unsigned int &carry)
{
	const unsigned int HALF_STRIDE = SMEM_ROWS >> 1;
	
	uint2 datum0, datum1; 

	// read input data
	datum0 = in[threadIdx.x];
	__syncthreads();
	datum1 = in[threadIdx.x + 128];

	smem_offset[0] = datum0.x + datum0.y;
	smem_offset[HALF_STRIDE * (ELTS_PER_ROW + 1)] = datum1.x + datum1.y;

	__syncthreads();

	if (threadIdx.x < 32) {

		unsigned int partial_reduction = SerialReduceSmem<8>(smem_segment);
		unsigned int seed = WarpScan<32, 32>(warpscan, partial_reduction);
		seed += carry;		
		carry += warpscan[63];	
		SerialScanSmem<8>(smem_segment, seed);
	}

	__syncthreads();

	unsigned int part;

	part = smem_offset[0];
	datum0.y = datum0.x + part;
	datum0.x = part;

	part = smem_offset[HALF_STRIDE * (ELTS_PER_ROW + 1)];
	datum1.y = datum1.x + part;
	datum1.x = part;

	// write output data
	out[threadIdx.x] = datum0;
	__syncthreads();
	out[threadIdx.x + 128] = datum1;
}


__global__ void SrtsScanSpine(
	unsigned int *d_ispine,
	unsigned int *d_ospine,
	unsigned int normal_block_elements)
{
	const unsigned int LOG_ELTS_PER_SEG 	= (SRTS_LOG_THREADS - LOG_WARP_THREADS) + 1;	// plus 1 for double inputs
	const unsigned int ELTS_PER_SEG 		= 1 << LOG_ELTS_PER_SEG;
	const unsigned int LOG_ELTS_PER_ROW		= (LOG_ELTS_PER_SEG < LOG_MEM_BANKS) ? LOG_MEM_BANKS : LOG_ELTS_PER_SEG;		// floor of 32 elts per row
	const unsigned int ELTS_PER_ROW			= 1 << LOG_ELTS_PER_ROW;
	const unsigned int SMEM_ROWS 			= ((SRTS_THREADS * 2) + ELTS_PER_ROW - 1) / ELTS_PER_ROW;
	const unsigned int LOG_SEGS_PER_ROW 	= LOG_ELTS_PER_ROW - LOG_ELTS_PER_SEG;	
	const unsigned int SEGS_PER_ROW			= 1 << LOG_SEGS_PER_ROW;
	
	__shared__ unsigned int smem[SMEM_ROWS][ELTS_PER_ROW + 1];
	__shared__ unsigned int warpscan[64];

	unsigned int *smem_segment;
	unsigned int carry;

	unsigned int row = threadIdx.x >> LOG_ELTS_PER_ROW;		
	unsigned int col = threadIdx.x - (row << LOG_ELTS_PER_ROW);			// remainder
	unsigned int *smem_offset = &smem[row][col];

	if (threadIdx.x < 32) {
		carry = 0;
		warpscan[threadIdx.x] = 0;

		// two segs per row, odd segs are offset by 8
		smem_segment = &smem[threadIdx.x >> LOG_SEGS_PER_ROW][(threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_ELTS_PER_SEG];
	}

	// scan the spine in blocks of cycle_elements
	unsigned int block_offset = 0;
	while (block_offset < normal_block_elements) {
		
		SrtsScan512<SMEM_ROWS, ELTS_PER_ROW>(	
			smem, smem_offset, smem_segment, warpscan,
			(uint2 *) &d_ispine[block_offset], 
			(uint2 *) &d_ospine[block_offset], 
			carry);

		block_offset += SRTS_CYCLE_ELEMENTS;
	}
} 




//------------------------------------------------------------------------------
// SrtsScanDigit
//------------------------------------------------------------------------------


template <
	typename K,
	typename V,	
	bool KEYS_ONLY, 
	unsigned int RADIX_DIGITS, 
	unsigned int SMEM_ROWS,
	unsigned int ELTS_PER_ROW,
	unsigned int FOURS_GROUPS, 
	unsigned int LOG_FOURS_GROUPS, 
	unsigned int THREADS_PER_GROUP,
	unsigned int LOG_THREADS_PER_GROUP,
	unsigned int ELTS_PER_SEG,
	unsigned int BIT, 
	bool UNGUARDED_IO>
__device__ inline void SrtsScanDigit512(
	unsigned int smem[SMEM_ROWS][ELTS_PER_ROW + 1],
	unsigned int *smem_offset,
	unsigned int *smem_segment,
	unsigned int warpscan[FOURS_GROUPS][THREADS_PER_GROUP * 3],
	typename VecType<K, 2>::Type *d_in_keys, 
	typename VecType<V, 2>::Type *d_in_data, 
	K *d_out_keys, 
	V *d_out_data, 
	unsigned int carry[RADIX_DIGITS], 
	unsigned int digit_scan[RADIX_DIGITS << 1], 
	unsigned int top_count[RADIX_DIGITS],
	volatile unsigned int oob)				// volatile to get the magical 24 registers
{
	const unsigned int DIGIT_MASK = 	(RADIX_DIGITS < 4) ? 0x1u : 0x3u;
	const unsigned int FOURS_STRIDE = 	(SRTS_THREADS / ELTS_PER_ROW) * (ELTS_PER_ROW + 1);

	typename VecType<K, 2>::Type keypair0, keypair1;

	unsigned int digit[4];
	unsigned int digit_group[4];
	unsigned int digit_shift[4];
	unsigned int rank[4];
	unsigned int encode0, encode1;

	if (UNGUARDED_IO) {
	
		// read input keys
		keypair0 = d_in_keys[threadIdx.x];
		keypair1 = d_in_keys[threadIdx.x + 128];

	} else {

		K* in = (K*) d_in_keys;

		keypair0.x = ((threadIdx.x << 1) < oob) ? in[(threadIdx.x << 1)] : 0xffffffff;
		keypair0.y = ((threadIdx.x << 1) + 1 < oob) ? in[(threadIdx.x << 1) + 1] : 0xffffffff;
		keypair1.x = ((threadIdx.x << 1) + 256 < oob) ? in[(threadIdx.x << 1) + 256] : 0xffffffff;
		keypair1.y = ((threadIdx.x << 1) + 257 < oob) ? in[(threadIdx.x << 1) + 257] : 0xffffffff;
	}


	//=========================================================================
	// TOP HALF
	//=========================================================================

	DecodeDigit<K, RADIX_DIGITS, BIT>(keypair0.x, digit[0], digit_group[0], digit_shift[0]);
	DecodeDigit<K, RADIX_DIGITS, BIT>(keypair0.y, digit[1], digit_group[1], digit_shift[1]);

	encode0 = EncodeInt(1u, digit_shift[0]);
	encode1 = EncodeInt(1u, digit_shift[1]);

	
	#pragma unroll
	for (unsigned int fours_group = 0; fours_group < FOURS_GROUPS; fours_group++) {
		smem_offset[fours_group * FOURS_STRIDE] = 
			((digit_group[0] == fours_group) ? encode0 : 0) + 
			((digit_group[1] == fours_group) ? encode1 : 0);
	}
	
	__syncthreads();

	if (threadIdx.x < 32) {

		unsigned int partial_reduction = SerialReduceSmem<ELTS_PER_SEG>(smem_segment);

		// warpscan reduction in digit warpscan_group
		unsigned int warpscan_group = threadIdx.x >> LOG_THREADS_PER_GROUP;

		unsigned int group_prefix = WarpScan<THREADS_PER_GROUP, 32>(
			warpscan[warpscan_group], 
			partial_reduction);

		if (threadIdx.x < RADIX_DIGITS) {

			// second half of carry update
			carry[threadIdx.x] += digit_scan[threadIdx.x + RADIX_DIGITS];

			unsigned int my_digit_shift = (threadIdx.x & DIGIT_MASK) << BYTE_ENCODE_SHIFT;
			unsigned int my_digit_group = threadIdx.x >> 2;
			unsigned int count = DecodeInt(
					warpscan[my_digit_group][(THREADS_PER_GROUP << 1) - 1], 
					my_digit_shift);	

			// Check overflow
			if (UNGUARDED_IO || oob >= 256) {
				if (__all(count <= 1)) {
					// Uncommon overflow: all first-round keys have same digit. 
					count = (threadIdx.x == digit[0]) ? 256 : 0;
				}
			} else if (oob < 256 && __all(count == 0)) {
				// last digit overflowed b/c of first-round f's ; set it to the first-round oob
				count = (threadIdx.x == RADIX_DIGITS - 1) ? oob : 0;
			}
		
			// store count
			top_count[threadIdx.x] = count;
		}

		// Downsweep scan
		SerialScanSmem<ELTS_PER_SEG>(smem_segment, group_prefix);
	}

	__syncthreads();

	// Calculate rank for each item

	rank[0] = DecodeInt(
		smem_offset[digit_group[0] * FOURS_STRIDE], 
		digit_shift[0]);

	rank[1] = (digit[0] == digit[1]) + DecodeInt(
		smem_offset[digit_group[1] * FOURS_STRIDE],
		digit_shift[1]);

	// No sync needed because insert/extract smem locations are not shared between threads

	//=========================================================================
	// BOTTOM HALF
	//=========================================================================

	DecodeDigit<K, RADIX_DIGITS, BIT>(keypair1.x, digit[2], digit_group[2], digit_shift[2]);
	DecodeDigit<K, RADIX_DIGITS, BIT>(keypair1.y, digit[3], digit_group[3], digit_shift[3]);

	encode0 = EncodeInt(1u, digit_shift[2]);
	encode1 = EncodeInt(1u, digit_shift[3]);

	
	#pragma unroll
	for (unsigned int fours_group = 0; fours_group < FOURS_GROUPS; fours_group++) {

		smem_offset[fours_group * FOURS_STRIDE] = 
			((digit_group[2] == fours_group) ? encode0 : 0) + 
			((digit_group[3] == fours_group) ? encode1 : 0);
	}

	__syncthreads();

	if (threadIdx.x < 32) {

		// eight threads (and partials) per digit warpscan_group
		unsigned int partial_reduction = SerialReduceSmem<ELTS_PER_SEG>(smem_segment);

		// warpscan reduction in digit warpscan_group
		unsigned int warpscan_group = threadIdx.x >> LOG_THREADS_PER_GROUP;
		unsigned int group_prefix = WarpScan<THREADS_PER_GROUP, 32>(
			warpscan[warpscan_group], 
			partial_reduction);

		if (threadIdx.x < RADIX_DIGITS) {
	
			unsigned int my_digit_shift = (threadIdx.x & DIGIT_MASK) << BYTE_ENCODE_SHIFT;
			unsigned int my_digit_group = threadIdx.x >> 2;
			unsigned int count = DecodeInt(
					warpscan[my_digit_group][(THREADS_PER_GROUP << 1) - 1], 
					my_digit_shift);	
	
			// Check overflow
			if (UNGUARDED_IO) {
				if (__all(count <= 1)) {
					// Uncommon overflow: all second-round keys have same digit. 
					count = (threadIdx.x == digit[2]) ? 256 : 0;
				}
			} else if (oob > 256 && __all(count == 0)) {
				// last digit overflowed b/c of second-round f's ; set it to the second-round oob
				count = (threadIdx.x == RADIX_DIGITS - 1) ? oob - 256 : 0;
			}
	
			// Perform overflow-free SIMD Kogge-Stone across digits
			unsigned int digit_prefix = WarpScan<RADIX_DIGITS, RADIX_DIGITS>(
					digit_scan, 
					top_count[threadIdx.x] + count);

			// first-half of carry update 
			carry[threadIdx.x] -= digit_prefix;
	
			// saves instructions and conflicts for bottom half
			top_count[threadIdx.x] += digit_prefix;
		}
	
		// Downsweep scan
		SerialScanSmem<ELTS_PER_SEG>(smem_segment, group_prefix);
	}

	__syncthreads();

	// Calculate rank for each item

	rank[0] += digit_scan[digit[0] + RADIX_DIGITS - 1];
	rank[1] += digit_scan[digit[1] + RADIX_DIGITS - 1];
	rank[2] = top_count[digit[2]] + DecodeInt(
				smem_offset[digit_group[2] * FOURS_STRIDE], 
				digit_shift[2]);
	rank[3] = top_count[digit[3]] + (digit[2] == digit[3]) + DecodeInt(
				smem_offset[digit_group[3] * FOURS_STRIDE],
				digit_shift[3]);

	__syncthreads();

		
	//=========================================================================
	// SWAP AND SCATTER
	//=========================================================================

	K *swapkeys = (K*) smem;

	// Push in keys
	swapkeys[rank[0]] = keypair0.x;
	swapkeys[rank[1]] = keypair0.y;
	swapkeys[rank[2]] = keypair1.x;
	swapkeys[rank[3]] = keypair1.y;

	__syncthreads();

	// Extract keys
	keypair0.x = swapkeys[threadIdx.x];
	keypair0.y = swapkeys[threadIdx.x + 128];
	keypair1.x = swapkeys[threadIdx.x + 256];
	keypair1.y = swapkeys[threadIdx.x + 384];

	// Get scatter offsets
	unsigned int offset0 = threadIdx.x + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypair0.x)];
	unsigned int offset1 = (threadIdx.x + 128) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypair0.y)];
	unsigned int offset2 = (threadIdx.x + 256) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypair1.x)];
	unsigned int offset3 = (threadIdx.x + 384) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypair1.y)];

	// Scatter keys
	if (UNGUARDED_IO) {

		d_out_keys[offset0] = keypair0.x;
		d_out_keys[offset1] = keypair0.y;
		d_out_keys[offset2] = keypair1.x;
		d_out_keys[offset3] = keypair1.y;

	} else {

		unsigned int total = digit_scan[(RADIX_DIGITS << 1) - 1];
		if (threadIdx.x < total ) d_out_keys[offset0] = keypair0.x;
		if (threadIdx.x + 128 < total) d_out_keys[offset1] = keypair0.y;
		if (threadIdx.x + 256 < total) d_out_keys[offset2] = keypair1.x;
		if (threadIdx.x + 384 < total) d_out_keys[offset3] = keypair1.y;
	}

	__syncthreads();
	
	if (!KEYS_ONLY) {

		V *swapdata = (V*) smem;

		// Read and push in data
		typename VecType<V, 2>::Type datapair0, datapair1;

		if (UNGUARDED_IO) {

			datapair0 = d_in_data[threadIdx.x];
			datapair1 = d_in_data[threadIdx.x + 128];

			swapdata[rank[0]] = datapair0.x;
			swapdata[rank[1]] = datapair0.y;
			swapdata[rank[2]] = datapair1.x;
			swapdata[rank[3]] = datapair1.y;

		} else {

			V* in = (V*) d_in_data;

			swapdata[rank[0]] = ((threadIdx.x << 1) < oob) ? in[(threadIdx.x << 1)] : 0xffffffff;
			swapdata[rank[1]] = ((threadIdx.x << 1) + 1 < oob) ? in[(threadIdx.x << 1) + 1] : 0xffffffff;
			swapdata[rank[2]] = ((threadIdx.x << 1) + 256 < oob) ? in[(threadIdx.x << 1) + 256] : 0xffffffff;
			swapdata[rank[3]] = ((threadIdx.x << 1) + 257 < oob) ? in[(threadIdx.x << 1) + 257] : 0xffffffff;
		}

		__syncthreads();

		// Extract data
		datapair0.x = swapdata[threadIdx.x];
		datapair0.y = swapdata[threadIdx.x + 128];
		datapair1.x = swapdata[threadIdx.x + 256];
		datapair1.y = swapdata[threadIdx.x + 384];

		// Scatter data
		if (UNGUARDED_IO) {

			d_out_data[offset0] = datapair0.x;
			d_out_data[offset1] = datapair0.y;
			d_out_data[offset2] = datapair1.x;
			d_out_data[offset3] = datapair1.y;

		} else {

			unsigned int total = digit_scan[(RADIX_DIGITS << 1) - 1];
			if (threadIdx.x < total ) d_out_data[offset0] = datapair0.x;
			if (threadIdx.x + 128 < total) d_out_data[offset1] = datapair0.y;
			if (threadIdx.x + 256 < total) d_out_data[offset2] = datapair1.x;
			if (threadIdx.x + 384 < total) d_out_data[offset3] = datapair1.y;
		}

		__syncthreads();
	}
}



template <typename K, typename V, bool KEYS_ONLY, unsigned int RADIX_BITS, unsigned int BIT>
__global__ 
void SrtsScanDigitBulk(
	unsigned int* d_spine,
	K* d_in_keys,
	K* d_out_keys,
	V* d_in_data,
	V* d_out_data,
	unsigned int num_big_blocks,
	unsigned int big_block_elements,
	unsigned int normal_block_elements, 
	unsigned int extra_elements_last_block)
{
	const unsigned int RADIX_DIGITS 		= 1 << RADIX_BITS;
	const unsigned int LOG_FOURS_GROUPS 	= (RADIX_BITS > 2) ? RADIX_BITS - 2 : 0;	// always at least one fours group
	const unsigned int FOURS_GROUPS 		= 1 << LOG_FOURS_GROUPS;

	// There will be (32 / FOURS_GROUPS) warpscanning threads per fours_group

	const unsigned int LOG_THREADS_PER_GROUP 	= LOG_WARP_THREADS - LOG_FOURS_GROUPS;
	const unsigned int THREADS_PER_GROUP 		= 1 << LOG_THREADS_PER_GROUP;

	// Each fours-group needs 128 items (each thread encodes two keys into an item), 
	// and we use 32 items per row to avoid bank conflicts, OR SRTS_CYCLE_ELEMENTS items (whichever is larger)

	const unsigned int MAX_EXCHG_ELT_SIZE 	= (sizeof(K) > sizeof(V)) ? sizeof(K) : sizeof(V);
	const unsigned int EXCH_CONV_SIZE			= (SRTS_CYCLE_ELEMENTS * MAX_EXCHG_ELT_SIZE) / sizeof(unsigned int);
	
	const unsigned int LOG_ELTS_PER_SEG 	= LOG_FOURS_GROUPS + (SRTS_LOG_THREADS - LOG_WARP_THREADS);
	const unsigned int ELTS_PER_SEG 		= 1 << LOG_ELTS_PER_SEG;
	const unsigned int LOG_ELTS_PER_ROW		= (LOG_ELTS_PER_SEG < LOG_MEM_BANKS) ? LOG_MEM_BANKS : LOG_ELTS_PER_SEG;		// floor of 32 elts per row
	const unsigned int ELTS_PER_ROW			= 1 << LOG_ELTS_PER_ROW;

	const unsigned int SMEM_ROWS 			= (SRTS_THREADS * FOURS_GROUPS > EXCH_CONV_SIZE) ? 
												((SRTS_THREADS * FOURS_GROUPS) + ELTS_PER_ROW - 1) / ELTS_PER_ROW : 
												((EXCH_CONV_SIZE)              + ELTS_PER_ROW - 1) / ELTS_PER_ROW;		// when our keys/values are bigger than the space we need for scan
	
	const unsigned int LOG_SEGS_PER_ROW 	= LOG_ELTS_PER_ROW - LOG_ELTS_PER_SEG;	
	const unsigned int SEGS_PER_ROW			= 1 << LOG_SEGS_PER_ROW;


	__shared__ unsigned int smem[SMEM_ROWS][ELTS_PER_ROW + 1];
	__shared__ unsigned int warpscan[FOURS_GROUPS][THREADS_PER_GROUP * 3];		// one warpscan per fours-group
	__shared__ unsigned int carry[RADIX_DIGITS];
	__shared__ unsigned int digit_scan[RADIX_DIGITS << 1];				// twice as many cells as digits 
	__shared__ unsigned int top_count[RADIX_DIGITS];

	unsigned int extra = (blockIdx.x == gridDim.x - 1) ? extra_elements_last_block : 0;

	// calculate our threadblock's range
	unsigned int block_offset, block_elements, oob;
	if (blockIdx.x < num_big_blocks) {
		block_offset = big_block_elements * blockIdx.x;
		block_elements = big_block_elements;
	} else {
		block_offset = (normal_block_elements * blockIdx.x) + (num_big_blocks * SRTS_CYCLE_ELEMENTS);
		block_elements = normal_block_elements;
	}
	if (extra) {
		block_elements -= SRTS_CYCLE_ELEMENTS;
	}			
	oob = block_offset + block_elements;	// out-of-bounds

	unsigned int row = threadIdx.x >> LOG_ELTS_PER_ROW;		
	unsigned int col = threadIdx.x - (row << LOG_ELTS_PER_ROW);		// remainder
	unsigned int *smem_offset = &smem[row][col];					// offset to write into
	unsigned int *smem_segment;										// segment for raking (warp-0 threads only) 

	if (threadIdx.x < 32) {

		if (threadIdx.x < THREADS_PER_GROUP) {
			
			#pragma unroll
			for (int fours_group = 0; fours_group < FOURS_GROUPS; fours_group++) {
				warpscan[fours_group][threadIdx.x] = 0;
			}
		}

		if (threadIdx.x < RADIX_DIGITS) {

			// read carry in parallel 
			carry[threadIdx.x] = d_spine[(gridDim.x * threadIdx.x) + blockIdx.x];

			// initialize digit_scan
			digit_scan[threadIdx.x] = 0;
			digit_scan[threadIdx.x + RADIX_DIGITS] = 0;
		}

		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_ELTS_PER_SEG;	
		smem_segment = &smem[row][col];
	}

	// scan in blocks of cycle_elements
	while (block_offset < oob) {

		SrtsScanDigit512<K, V, KEYS_ONLY, RADIX_DIGITS, SMEM_ROWS, ELTS_PER_ROW, FOURS_GROUPS, LOG_FOURS_GROUPS, THREADS_PER_GROUP, LOG_THREADS_PER_GROUP, ELTS_PER_SEG, BIT, true>(	
			smem,
			smem_offset,
			smem_segment,
			warpscan,
			(typename VecType<K, 2>::Type *) &d_in_keys[block_offset], 
			(typename VecType<V, 2>::Type *) &d_in_data[block_offset], 
			d_out_keys, 
			d_out_data, 
			carry, 
			digit_scan,
			top_count, 
			extra);

		block_offset += SRTS_CYCLE_ELEMENTS;
	}

		
	// cleanup
	if (extra) {

		SrtsScanDigit512<K, V, KEYS_ONLY, RADIX_DIGITS, SMEM_ROWS, ELTS_PER_ROW, FOURS_GROUPS, LOG_FOURS_GROUPS, THREADS_PER_GROUP, LOG_THREADS_PER_GROUP, ELTS_PER_SEG, BIT, false>(	
			smem,
			smem_offset,
			smem_segment,
			warpscan,
			(typename VecType<K, 2>::Type *) &d_in_keys[block_offset], 
			(typename VecType<V, 2>::Type *) &d_in_data[block_offset], 
			d_out_keys, 
			d_out_data, 
			carry, 
			digit_scan, 
			top_count,
			extra);
	}

} 




#endif



