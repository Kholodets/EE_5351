#include <stdio.h>
#include "2Dconvolution.h"

Matrix AllocateDeviceMatrix(const Matrix M);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);

//#define TILE_SIZE 28
//#define BLOCK_SIZE TILE_SIZE (TILE_SIZE + KERNEL_SIZE -1)

// includes, kernels
__constant__ float Mc[KERNEL_SIZE * KERNEL_SIZE];

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{
	__shared__ float Ns[BLOCK_SIZE * BLOCK_SIZE];

	//x and y within the block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//output to P location
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;

	//input from N location
	int row_i = row_o - KS_DIV_2;
	int col_i = col_o - KS_DIV_2;


	//copy needed values of N into shared memory Ns
	if ((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width)) {
		Ns[ty * BLOCK_SIZE + tx] = N.elements[row_i * N.pitch + col_i];
	} else {
		Ns[ty * BLOCK_SIZE + tx] = 0.0f;
	}

	//make sure its all loaded before continuing
	__syncthreads();

	float output = 0.0f;

	//calculate convolution for this index
	//this method will cause divergence within the warp, but hopefully to an acceptable degree
	//could reorder the threads within the tile to put all the waiting threads at the end
	//and thus in warp with each other
	if (ty < TILE_SIZE && tx < TILE_SIZE) {
		for(int i = 0; i < KERNEL_SIZE; ++i) {
			for(int j = 0; j < KERNEL_SIZE; ++j) {
				output += Mc[i * KERNEL_SIZE + j] * Ns[(ty + i) * BLOCK_SIZE + tx + j];
			}
		}


		//write output to P
		//again, some divergence here, oh well
		if (row_o < P.height && col_o < P.width) {
			P.elements[row_o * P.pitch + col_o] = output;
	
		}
	}

}


// Kernel calling function specification
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	// Load M and N to the device
	//Matrix Md = AllocateDeviceMatrix(M);
	//CopyToDeviceMatrix(Md, M);
	
	cudaMemcpyToSymbol(Mc, M.elements, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));


	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P); // Clear memory

	// Setup the execution configuration

	int xtiles = (N.width + TILE_SIZE - 1) / TILE_SIZE;
	int ytiles = (N.height + TILE_SIZE - 1) / TILE_SIZE;

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(xtiles, ytiles);


	// Launch the device computation threads!
	/*cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	*/

	ConvolutionKernel<<<dimGrid, dimBlock>>>(Nd, Pd);
	
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("%f, ", time);
	*/

	//cudaDeviceSynchronize();

	//fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
	

	// Read P from the device
	CopyFromDeviceMatrix(P, Pd);

	// Free device matrices
	//FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}
