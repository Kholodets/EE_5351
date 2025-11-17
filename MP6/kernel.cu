#include <stdio.h>
#define BLOCK_SIZE 1024

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
		unsigned int *csrColIdx, float *csrData, float *inVector, 
		float *outVector) {


	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < dim) {
		float dot = 0;
		int row_start = csrRowPtr[row];
		int row_end = csrRowPtr[row+1];
		for (int i = row_start; i < row_end; ++i) {
			dot += csrData[i] * inVector[csrColIdx[i]];
		}
		outVector[row] += dot;
	}
}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
		unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
		unsigned int *jdsColIdx, float *jdsData, float* inVector,
		float *outVector) {

	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < dim) {
		float dot = 0;
		for (int i = 0; i < jdsRowNNZ[row]; ++i) {
			int k = jdsColStartIdx[i] + row;
			dot += jdsData[k] * inVector[jdsColIdx[k]];
		}
		outVector[jdsRowPerm[row]] += dot;
	}
	
}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
		float *csrData, float *inVector, float *outVector) {

	int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
	spmv_csr_kernel<<<blocks, BLOCK_SIZE>>>(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector);
}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
		unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
		float* inVector, float *outVector) {

	int blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
	spmv_jds_kernel<<<blocks, BLOCK_SIZE>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector);

}






