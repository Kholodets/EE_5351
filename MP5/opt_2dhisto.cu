#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"

//maximum threads per block
#define TPB 1
//this is how many SMs my GPU has
#define BPG 1

__global__ void hist_basic(uint32_t *input, size_t n, uint32_t *bins);
__global__ void hist_private(uint32_t *input, size_t n, uint32_t *bins);

void opt_2dhisto(uint32_t *in_d, int n, uint32_t *bins_d)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */

	hist_basic<<<BPG, TPB>>>(in_d, n, bins_d);

}

/* Include below the implementation of any other functions you need */

void prep_hist(uint32_t *input[], int iw, int ih, uint32_t **in_d, uint32_t **bins_d)
{
	uint32_t *in_dp;
	cudaMalloc((void **)&in_dp, sizeof(uint32_t) * iw * ih);

	printf("host:\n");
	for (int i = 0; i < ih; i++) {
		for (int j = 0; j < iw; ++j){
			printf("%d, ", input[i][j]);
		}
		printf("\n");
		cudaMemcpy(in_dp + (iw * i), input[i], sizeof(uint32_t) * iw, cudaMemcpyHostToDevice);
	}

	uint32_t *test = (uint32_t *) malloc(sizeof(uint32_t) * iw * ih);
	cudaMemcpy(test, in_dp, sizeof(uint32_t) * iw * ih, cudaMemcpyDeviceToHost);
	printf("device:\n");
	for (int i =0; i < ih; i++) {
		for (int j = 0; j < iw; j++) {
			printf("%d, ", test[j + iw*i]);
		}
		printf("\n");
	}

	uint32_t *bins_dp;
	cudaMalloc((void **)&bins_dp, sizeof(uint32_t) * HISTO_WIDTH * HISTO_HEIGHT);
	cudaMemset(bins_dp, 0, sizeof(uint32_t) * HISTO_WIDTH * HISTO_HEIGHT);

	*in_d = in_dp;
	*bins_d = bins_dp;
}

void fin_hist(uint32_t *in_d, uint32_t *bins_d, uint8_t *bins_h)
{
	uint32_t bins_temp[HISTO_HEIGHT * HISTO_WIDTH];
	cudaMemcpy(bins_temp, bins_d, sizeof(uint32_t) * HISTO_WIDTH * HISTO_HEIGHT, cudaMemcpyDeviceToHost);

	for (int i = 0; i < HISTO_HEIGHT * HISTO_WIDTH; i++) {
		printf("%d, ", bins_temp[i]);
		bins_h[i] = bins_temp[i] > 255 ? 255: bins_temp[i];
	}
	printf("\n");
	
	cudaFree(in_d);
	cudaFree(bins_d);
}


__global__ void hist_basic(uint32_t *input, size_t n, uint32_t *bins)
{	//int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		bins[i] = 0;
		
	int stride = blockDim.x * gridDim.x;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
		//printf("counting %d, threadIdx.x %d, threadIdx.y %d, i %d, n %d, stride %d\n", input[i], threadIdx.x, threadIdx.y, i, (int) n, stride);
		int what = atomicAdd( &(bins[input[i]]), 1);
		//printf("now = %d\n", what);
	}
	//printf("left loop[????\n");
}

__global__ void hist_private(uint32_t *input, size_t n, uint32_t *bins)
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

	//int hist_start = HISTO_WIDTH * blockIdx.x;
	for (int i = threadIdx.x; i < HISTO_WIDTH; i += blockDim.x)
		atomicAdd( &(bins[i]), hist_tile[i]);
		//tile_bins[i + hist_start] = hist_tile[i];
}
