/* Vector addition: C = A + B.
 * Host code.
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

#include "vectoradd.h"

#define MAXLINE 100000

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

// The computeGold function computes the "golden" solution on the CPU, which is compared against the result you compute on the GPU
extern "C"
void computeGold(float*, float*, float*, unsigned int);

void InitVector(float* V, int length);
int ReadFile(float* V, char* file_name);
void WriteFile(float* V, char* file_name);

void VectorAddOnDevice(float* A, float* B, float* C);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  // Vectors for the program
  float *A_h, *B_h, *C_h;
  
  // Number of elements in the vectors
  int errorA = 0, errorB = 0;
  
  srand(2012);
  // Allocate memory for the vectors

  A_h = (float*) malloc(VSIZE*sizeof(float));
  B_h = (float*) malloc(VSIZE*sizeof(float));
  C_h = (float*) malloc(VSIZE*sizeof(float));
  
  // Check command line for input vector files
  if(argc != 3 && argc != 4) 
  {
    // No inputs provided
    // Randomly initialize the vectors A,B
    InitVector(A_h, VSIZE);
    InitVector(B_h, VSIZE);        
  }
  else
  {
    // Inputs provided
    // read source vectors from file
    errorA = ReadFile(A_h, argv[1]);
    errorB = ReadFile(B_h, argv[2]);
    // check for read errors
    if(errorA != VSIZE || errorB != VSIZE)
    {
      printf("Error reading input files %d, %d\n", errorA, errorB);
      return 1;
    }
  }

  // A + B on the device
  VectorAddOnDevice(A_h, B_h, C_h);
  
  // compute the vector addition golden result on the CPU for comparison
  float* reference = (float*) malloc(VSIZE*sizeof(float));
  computeGold(reference, A_h, B_h, VSIZE);
        
  // check if the device result is equivalent to the expected solution
  //CUTBoolean res = cutComparefe(reference, VSIZE, 0.0001f);
  unsigned int res = 1;

  float errTol = 0.001f;
  
  for (unsigned int i = 0; i < VSIZE; i++){
    float diff = abs(reference[i] - C_h[i]);
    bool small = abs(reference[i]) < 1.0e-2f;
  	if ((small && diff > errTol) || (!small && abs(diff / reference[i]) > errTol)) {
      printf("%d: %f, %f\n",i,reference[i],C_h[i]);
      res = 0;
    }
  }
  
  printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  
  // output result if output file is requested
  if(argc == 4)
  {
    WriteFile(C_h, argv[3]);
  }
  else if(argc == 2)
  {
    WriteFile(C_h, argv[1]);
  }    

  // Free host matrices
  free(A_h);
  A_h = NULL;
  free(B_h);
  B_h = NULL;
  free(C_h);
  C_h = NULL;
  return 0;
}


// Initialize vector with random values
void InitVector(float* V, int length)
{		
  for(unsigned int i = 0; i < length; i++)
  {
    V[i] = rand() / (float)RAND_MAX;
  }
}	

// Read a floating point vector in from file
int ReadFile(float* V, char* file_name)
{
  unsigned int data_read = VSIZE;
  FILE* input = fopen(file_name, "r");
  char vector_string[MAXLINE];
  fgets(vector_string, MAXLINE, input);
  char* part = strtok(vector_string, " ");
  for (unsigned i = 0; i < VSIZE && part != NULL; i++) {
    V[i] = atof(part);
    part = strtok(NULL, " ");
  }
  return data_read;
}

// Write a floating point vector to file
void WriteFile(float* V, char* file_name)
{
  FILE* output = fopen(file_name, "w");
  for (unsigned i = 0; i < VSIZE; i++) {
    fprintf(output, "%f ", V[i]);
  }
}
