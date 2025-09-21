/* Matrix multiplication: P = M * N.
 * Device code.
 */

//student: lexi maclean macle119

#include <stdio.h>
#include "matrixmul.h"

//forward declarations
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	//Multiply the two matrices
	
	//naive approach with one block
	//appropriate element in P is just the position
	//of the thread in the block
	int x = threadIdx.x;
	int y = threadIdx.y;

	float c = 0;

	for (int i = 0; i < M.width; ++i) {
		c += M.elements[M.width * y + i] * N.elements[N.width * i + x];
	}

	P.elements[y * P.width + x] = c;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	//Interface host call to the device kernel code and invoke the kernel
	
	Matrix M_d = AllocateDeviceMatrix(M);
	Matrix N_d = AllocateDeviceMatrix(N);

	Matrix P_d = AllocateDeviceMatrix(P);

	CopyToDeviceMatrix(M_d, M);
	CopyToDeviceMatrix(N_d, N);

	//cudaDeviceSynchronize(); 
	//not necessary? all these calls should be on the same stream

	//instructions say 1 thread block is enough for entire matrix
	//so, since this is naive approach, we'll just make the 1 block
	//with heightXwidth threads
	
	dim3 dimBlock(P.width, P.height, 1); //assume P is already the right size

	MatrixMulKernel<<<1, dimBlock>>>(M_d, N_d, P_d);

	//cudaDeviceSynchronize();
	//again, shouldn't be necessary

	//P_d should now be populated with the solution
	CopyFromDeviceMatrix(P, P_d);

	//what...no free functions?
	cudaFree(M_d.elements);
	cudaFree(N_d.elements);
	cudaFree(P_d.elements); 
}
