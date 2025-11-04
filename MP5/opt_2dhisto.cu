#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"

#define TILE_WIDTH 32


void opt_2dhisto( /* define your own function parameters */ )
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */


}

/* Include below the implementation of any other functions you need */

__global__ void build_hist_tiles(uint32_t *input, size_t n, uint8_t *tile_bins)
{
	__shared__ uint8_t hist_tile[HISTO_WIDTH];

	//initialize sm hist to 0
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		hist_tile[i] = 0;

	//int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride)
		atomicAdd( &(hist_tile[input[i]]), 1);

	int hist_start = HISTO_WIDTH * blockIdx.x;
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		tile_bins[i + hist_start] = hist_tile[i];
}
