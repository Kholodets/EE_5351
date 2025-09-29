/* Matrix multiplication: C = A * B.
 * Device code.
 */

#include <stdio.h>
#include "matrixmul.h"

//forward declarations
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);


#define TILE_WIDTH 16


// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float M_s[TILE_WIDTH * TILE_WIDTH];
	__shared__ float N_s[TILE_WIDTH * TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	int tiles = (M.width + TILE_WIDTH - 1) / TILE_WIDTH;

	int active = col < P.width && row < P.height ? 1 : 0;

	float val = 0;
	for(int m = 0; m < tiles; ++m) {
		//adjacent threads should access adjacent memory when loading M_s and N_s
	
		M_s[ty * TILE_WIDTH + tx] = TILE_WIDTH * m + tx < M.width && row < M.height ? M.elements[M.pitch * row + TILE_WIDTH * m + tx] : 0;
		N_s[ty * TILE_WIDTH + tx] = m * TILE_WIDTH + ty < N.height && col < N.width ? N.elements[(m * TILE_WIDTH + ty)*N.pitch + col] : 0;
		

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) {
			val += M_s[ty * TILE_WIDTH + k] * N_s[k * TILE_WIDTH + tx];
		}
		__syncthreads();
	}

	if (active)
		P.elements[row * P.pitch + col] = val;

}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	// Load M and N to the device
	Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	CopyToDeviceMatrix(Pd, P); // Clear memory

	// Setup the execution configuration
	int xtiles = (P.width + TILE_WIDTH - 1) / TILE_WIDTH;
	int ytiles = (P.height + TILE_WIDTH - 1) / TILE_WIDTH;
	dim3 dimGrid(xtiles, ytiles);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	// Launch the device computation threads!
	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);


	// Read P from the device
	CopyFromDeviceMatrix(P, Pd);

	// Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}
