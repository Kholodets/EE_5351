/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

// includes, kernels
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
bool CompareMatrices(Matrix A, Matrix B);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
int ReadParamsFile(int* params, char* file_name, int num_params);

#define ASIZE 10000

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	int errorM = 0, errorN = 0;
	
	srand(52);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		
		M  = AllocateMatrix(rand() % ASIZE, rand() % ASIZE, 1);
		N  = AllocateMatrix(M.width, rand() % ASIZE, 1);
		P  = AllocateMatrix(M.height, N.width, 0);
		
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*)malloc(3 * sizeof(int));
		unsigned data_read = ReadParamsFile(params, argv[1], 3);
		if(data_read != 3){
			printf("Error reading parameter file\n");
			return 1;
		}

		M  = AllocateMatrix(params[0], params[1], 0);
		N  = AllocateMatrix(params[1], params[2], 0);		
		P  = AllocateMatrix(params[0], params[2], 0);
		unsigned sizeM = ReadFile(&M, argv[2]);
		unsigned sizeN = ReadFile(&N, argv[3]);
		if( (sizeM != M.height * M.width) || (sizeN != N.height * N.width) )
		{
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}

	// M * N on the device
    MatrixMulOnDevice(M, N, P);
    
    printf("GPU computation complete\n");
    // compute the matrix multiplication on the CPU for comparison
    //Matrix reference = AllocateMatrix(P.height, P.width, 0);
    //computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
        
    printf("CPU computation complete\n");
    // check if the device result is equivalent to the expected solution
    //bool res = CompareMatrices(reference, P);
    //printf("Test %s\n", res ? "PASSED" : "FAILED");
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int data_read = M->width * M->height;
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < data_read; i++) 
        fscanf(input, "%f", &(M->elements[i]));
    return data_read;
}

// Read params of input matrices
int ReadParamsFile(int* params, char* file_name, int num_params)
{
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < num_params; i++) 
        fscanf(input, "%d", &(params[i]));
    return num_params;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    unsigned int size = M.width * M.height;
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < size; i++) {
        fprintf(output, "%f ", M.elements[i]);
    }
}

// returns true iff A and B have same elements in same order
bool CompareMatrices(Matrix reference, Matrix compare) {
    unsigned int size = reference.width * reference.height;

    if ((reference.width != compare.width) || (reference.height != compare.height))
        return false;

    float* ref = reference.elements;
    float* comp = compare.elements;

    float errTol = 0.001f;

    for (unsigned int i = 0; i < size; i++) {
        float diff = abs(ref[i] - comp[i]);
        bool small = abs(ref[i]) < 1.0f;
        if ((small && diff > errTol) || (!small && abs(diff / ref[i]) > errTol)) {
            return false;
        }
    }
    return true;
}
