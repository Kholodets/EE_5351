#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"

//maximum threads per block
#define TPB 1024
//this is how many SMs my GPU has
#define BPG 80

__global__ void hist_basic(uint32_t *input, size_t n, uint32_t *bins);
__global__ void hist_private(uint32_t *input, size_t n, uint32_t *bins);
__global__ void hist_private_tileout(uint32_t *input, size_t n, uint32_t *bins);

void opt_2dhisto(uint32_t *in_d, int n, uint32_t *bins_d)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */

	hist_private<<<BPG, TPB>>>(in_d, n, bins_d);
	cudaDeviceSynchronize();

}

/* Include below the implementation of any other functions you need */

void prep_hist(uint32_t *input[], int iw, int ih, uint32_t **in_d, uint32_t **bins_d)
{
	uint32_t *in_dp;
	cudaMalloc((void **)&in_dp, sizeof(uint32_t) * iw * ih);

	for (int i = 0; i < ih; i++) {
		cudaMemcpy(in_dp + (iw * i), input[i], sizeof(uint32_t) * iw, cudaMemcpyHostToDevice);
	}

	uint32_t *bins_dp;
	cudaMalloc((void **)&bins_dp, sizeof(uint32_t) * HISTO_WIDTH * HISTO_HEIGHT /* * BPG */);

	//cudaMemset(bins_dp, 0, sizeof(uint32_t) * HISTO_WIDTH * HISTO_HEIGHT);

	*in_d = in_dp;
	*bins_d = bins_dp;
}

void fin_hist(uint32_t *in_d, uint32_t *bins_d, uint8_t *bins_h)
{
	uint32_t bins_temp[HISTO_HEIGHT * HISTO_WIDTH * BPG];
	cudaMemcpy(bins_temp, bins_d, sizeof(uint32_t) * HISTO_WIDTH * HISTO_HEIGHT  /* * BPG */, cudaMemcpyDeviceToHost);

	/*
	for (int i = 1; i < BPG; i++) {
		for (int j = 0; j < HISTO_HEIGHT * HISTO_WIDTH; j++) {
			bins_temp[j] += bins_temp[j + HISTO_HEIGHT * HISTO_WIDTH * i];
		}
	}
	*/

	for (int i = 0; i < HISTO_HEIGHT * HISTO_WIDTH; i++) {
		bins_h[i] = bins_temp[i] > 255 ? 255: bins_temp[i];
	}
	
	
	cudaFree(in_d);
	cudaFree(bins_d);
}


__global__ void hist_basic(uint32_t *input, size_t n, uint32_t *bins)
{	//int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		bins[i] = 0;
		
	int stride = blockDim.x * gridDim.x;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride)
		atomicAdd( &(bins[input[i]]), 1);
}

__global__ void hist_private(uint32_t *input, size_t n, uint32_t *bins)
{
	__shared__ uint32_t hist_tile[HISTO_WIDTH];

	//initialize sm hist to 0
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x) {
		hist_tile[i] = 0;
		bins[i] = 0;
	}

	int stride = blockDim.x * gridDim.x;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride)
		atomicAdd( &(hist_tile[input[i]]), 1);

	__syncthreads();
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		atomicAdd( &(bins[i]), hist_tile[i]);
}


//this one assumes bins has one bin for each block
__global__ void hist_private_tileout(uint32_t *input, size_t n, uint32_t *bins)
{
	__shared__ uint32_t hist_tile[HISTO_WIDTH];

	//initialize sm hist to 0
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x) {
		hist_tile[i] = 0;
		bins[i] = 0;
	}

	//int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride)
		atomicAdd( &(hist_tile[input[i]]), 1);

	__syncthreads();
	int hist_start = HISTO_WIDTH * blockIdx.x;
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		bins[i + hist_start] = hist_tile[i];
}
