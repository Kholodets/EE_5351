#include <stdlib.h>
#include <stdio.h>
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE * 2)
#define THRESHOLD 1

// **===--------------------- Modify this function -----------------------===**
//! @param in_data  input data in global memory
//! @param out_data output data in global memory (block sums)
//! @param n        number of input elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *in_data, unsigned int *out_data, int n)
{
	__shared__ float partialSum[TILE_SIZE];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockDim.x * blockIdx.x;
	partialSum[t] = start+t >= n ? 0 : in_data[start + t];
	partialSum[blockDim.x + t] = start + blockDim.x + t >= n ? 0: in_data[start + blockDim.x + t];

	for (unsigned int stride = blockDim.x; stride >= 1; stride >>=1) {
		__syncthreads();
		if (t < stride) {
			partialSum[t] += partialSum[t+stride];
		}
	}

	out_data[blockIdx.x] = partialSum[0];			
}

// **===----------------- Modify this function ---------------------===**
// Setup device data for input and block sums
// Recursively call kernel to perform reduction, swapping input and output 
//   between kernel calls
// Note: unsigned int* h_data is both the input and the output of this function.
unsigned int computeOnDevice(unsigned int* h_data, int num_elements)
{
	unsigned int *a_d;
	cudaMalloc((void **)&a_d, num_elements * sizeof(unsigned int));
	unsigned int *b_d;
	cudaMalloc((void **)&b_d, num_elements * sizeof(unsigned int));

	cudaMemcpy(a_d, h_data, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);

	int elm = num_elements;

	//This could be improved by benchmarking to find a good threshold
	//where the CPU is able to process a shorter array faster than the GPU
	//but theres no benchmarking necessary here for that so this processes the whole  thing on GPU
	while (elm > THRESHOLD) {
		int tiles = (elm + TILE_SIZE - 1) / TILE_SIZE;
		reduction<<<tiles, BLOCK_SIZE>>>(a_d, b_d, elm);
		elm = tiles;
		unsigned int *swap = a_d;
		a_d = b_d;
		b_d = swap;
	}

	unsigned int *a_h = (unsigned int*) malloc(elm * sizeof(unsigned int));
	cudaMemcpy(a_h, a_d, elm * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	unsigned int res = a_h[0];

	cudaFree(&a_d);
	cudaFree(&b_d);
	free(a_h);

	return res;
}
