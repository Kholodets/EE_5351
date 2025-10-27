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
// Device Functions






// Kernel Functions

__global__ void add_all(unsigned int *data, int n, unsigned int *aux)
{
	__shared__ unsigned int b;
	unsigned int t = threadIdx.x + blockDim.x * blockIdx.x;
	if (t == 0) {
		b = aux[blockIdx.x];
	}

	if (t < n) {
		int a = data[t];
		a += b;
		data[t] = a;
	}
}
		

__global__ void scan_kernel(unsigned int *in_data, unsigned int *out_data, int n, unsigned int *aux)
{
	//setup + readin
	__shared__ unsigned int scan_array[BLOCK_SIZE * 2];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockDim.x * blockIdx.x;
	scan_array[t] = start + 2 >= n ? 0 : in_data[start + t];
	scan_array[blockDim.x + t] = start + blockDim.x + t >= n ? 0 : in_data[start + blockDim.x + t];

	//reduction step
	int stride = 1;
	while (stride <= BLOCK_SIZE) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE) {
			scan_array[index] += scan_array[index - stride];
		}
		stride <<= 1;
		
		__syncthreads();

/*		if (t == 0) {
			for (int k = 0; k < BLOCK_SIZE * 2; ++k) {
				printf("%d ", scan_array[k]);
			}
			printf("\n");
		}
*/
	}

	__syncthreads();
	
	if (t == 0) {
			for (int k = 0; k < BLOCK_SIZE * 2; ++k) {
				printf("%d ", scan_array[k]);
			}
			printf("\n");
	}


	//distribution step
	stride = BLOCK_SIZE / 2;
	while (stride) {

		/*if (t == 0) {
			for (int k = 0; k < BLOCK_SIZE * 2; ++k) {
				printf("%d ", scan_array[k]);
			}
			printf("\n");
		}*/		
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if ((index + stride) < 2 * BLOCK_SIZE) {
			scan_array[index + stride] += scan_array[index];
		}
		
		stride >>= 1;
		__syncthreads();
	}

	//writeout
	if (start + t < n) {
		out_data[start + t] = scan_array[t];
	}

	if (start + blockDim.x + t < n) {
		out_data[start + blockDim.x + t] = scan_array[t + blockDim.x];
	}

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

void prescanArray(unsigned int *outArray, unsigned int *inArray, unsigned int* blockSums, int numElements)
{   //all pointers are to device memory regions
    int tiles = (numElements + TILE_SIZE - 1) / TILE_SIZE;
    printf("reducing from %d to %d\n", numElements, tiles);
    scan_kernel<<<tiles, BLOCK_SIZE>>>(inArray, outArray, numElements, blockSums);

    if (tiles > 1) {
	    unsigned int *nextBlocks = preallocBlockSums(tiles);
	    prescanArray(blockSums, blockSums, nextBlocks, tiles);
	    deallocBlockSums(nextBlocks);
	    add_all<<<tiles - 1, BLOCK_SIZE>>>(outArray + TILE_SIZE, numElements - TILE_SIZE, blockSums);
    }


}
// **===-----------------------------------------------------------===**

// Use the function to allocate your block sums here
unsigned int* preallocBlockSums(int num_elements)
{
	unsigned int* bs_d = 0; //assign your device memory pointer to this variable
	// =========================================

	int blocks = (num_elements + TILE_SIZE - 1) / TILE_SIZE;
	cudaMalloc((void **) &bs_d, blocks * sizeof(unsigned int));

	// =========================================
	return bs_d;
}

// Use the function to deallocate (free) your block sums
void deallocBlockSums(unsigned int* bs_d)
{
	cudaFree(bs_d);
}
