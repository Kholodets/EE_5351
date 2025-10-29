#include <stdio.h>

extern const int NUM_BANKS = 16;
extern const int LOG_NUM_BANKS = 4;
extern const int TILE_SIZE = 1024;
// You can use any other block size you wish.
extern const int BLOCK_SIZE = (TILE_SIZE / 2);



__global__ void add_all(unsigned int *data, int n, unsigned int *aux);
__global__ void scan_kernel(unsigned int *in_data, unsigned int *out_data, int n, unsigned int *aux);
void prescanArray(unsigned int *outArray, unsigned int *inArray, unsigned int* blockSums, int numElements);
unsigned int* preallocBlockSums(int num_elements);
void deallocBlockSums(unsigned int* bs_d);
void print_d_ar(unsigned int *ar, int n);

// Device Functions






// Kernel Functions

__global__ void add_all(unsigned int *data, int n, unsigned int *aux)
{
	unsigned int t = threadIdx.x + TILE_SIZE * blockIdx.x;
	
	//these two methods seemed to result in the same speedup
/*	__shared__ unsigned int c;
	if (threadIdx.x == 0) {
		c = aux[blockIdx.x];
	}
	__syncthreads();
*/	
	
	unsigned int c = aux[blockIdx.x];

	//if (t < n) {
		int a = data[t];
		int b = data[t + blockDim.x];
		a += c;
		b += c;
		data[t] = a;
		data[t + blockDim.x] = b;
	//}

}


__global__ void scan_kernel(unsigned int *in_data, unsigned int *out_data, int n, unsigned int *aux)
{
	//setup + readin
	__shared__ unsigned int scan_array[BLOCK_SIZE * 2];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockDim.x * blockIdx.x;
	scan_array[t] = start + 2 >= n ? 0 : in_data[start + t];
	scan_array[blockDim.x + t] = start + blockDim.x + t >= n ? 0 : in_data[start + blockDim.x + t];

	__syncthreads();

	//reduction step
	int stride = 1;
	while (stride <= BLOCK_SIZE) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE) {
			scan_array[index] += scan_array[index - stride];
		}
		stride <<= 1;

		__syncthreads();
	}



	//distribution step
	stride = BLOCK_SIZE / 2;
	while (stride) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * BLOCK_SIZE) {
			scan_array[index + stride] += scan_array[index];
		}

		stride >>= 1;
		__syncthreads();
	}

	//writeout
	if (start + t < n) 
		out_data[start + t] = scan_array[t];

	if (start + blockDim.x + t < n)
		out_data[start + blockDim.x + t] = scan_array[t + blockDim.x];

	if (t == 0) {
		aux[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];		
	}

}


// Host Helper Functions (allocate your own data structure...)

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE

void scan_cpu(unsigned int *in_ar_d, unsigned int *out_ar_d, int n)
{
	unsigned int *in_ar_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	unsigned int *out_ar_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	
	cudaMemcpy(in_ar_h, in_ar_d, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	out_ar_h[0] = in_ar_h[0];
	for (int i = 1; i < n; ++i)
		out_ar_h[i] = out_ar_h[i-1] + in_ar_h[i]; 

	cudaMemcpy(out_ar_d, out_ar_h, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

	free(in_ar_h);
	free(out_ar_h);
}


void prescanArray(unsigned int *outArray, unsigned int *inArray, unsigned int* blockSums, int numElements)
{   //all pointers are to device memory regions
	
	//could not find any value for this detour which provided a speedup
/*
	if (numElements <= 4096) {
		scan_cpu(inArray, outArray, numElements);
		return;
	}
*/

	int tiles = (numElements + TILE_SIZE - 1) / TILE_SIZE;

	scan_kernel<<<tiles, BLOCK_SIZE>>>(inArray, outArray, numElements, blockSums);
	if (tiles > 1) {
		unsigned int *outBlocks = preallocBlockSums(numElements);
		unsigned int *nextBlocks = preallocBlockSums(tiles);
		int mtiles = (tiles + TILE_SIZE - 1) / TILE_SIZE;
		prescanArray(outBlocks, blockSums, nextBlocks, TILE_SIZE * mtiles);
		add_all<<<tiles - 1, BLOCK_SIZE>>>(outArray + TILE_SIZE, numElements - TILE_SIZE, outBlocks);
		deallocBlockSums(nextBlocks);
		deallocBlockSums(outBlocks);
	}
}
// **===-----------------------------------------------------------===**

void print_d_ar(unsigned int *ar_d, int n)
{
	unsigned int *ar_h = (unsigned int *) malloc(sizeof(unsigned int) * n);
	cudaMemcpy(ar_h, ar_d, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; ++i) {
		printf("%d ", ar_h[i]);
	} printf("\n");
	free(ar_h);
}

// Use the function to allocate your block sums here
unsigned int *preallocBlockSums(int num_elements)
{
	unsigned int* bs_d = 0; //assign your device memory pointer to this variable
				// =========================================

	int blocks = (num_elements + TILE_SIZE - 1) / TILE_SIZE;
	int mblocks = (blocks + TILE_SIZE - 1) / TILE_SIZE;
	cudaMalloc((void **) &bs_d, mblocks * TILE_SIZE * sizeof(unsigned int));

	// =========================================
	return bs_d;
}

// Use the function to deallocate (free) your block sums
void deallocBlockSums(unsigned int *bs_d)
{
	cudaFree(bs_d);
}
