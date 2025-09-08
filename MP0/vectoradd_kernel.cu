/* Vector Addition: C = A + B.
 * Device code.
 */

#include <stdio.h>
#include "vectoradd.h"

// Vector addition kernel thread specification
__global__ void VectorAddKernel(float* A, float* B, float* C)
{
  // INSERT CODE to add the two vectors

}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void VectorAddOnDevice(float* A, float* B, float* C)
{
	//Interface host call to the device kernel code and invoke the kernel

	// steps:
	// 1. declare and allocate device vectors A_d, B_d and C_d with length same as input vectors

	// 2. copy A to A_d, B to B_d

	// 3. launch kernel to compute C_d = A_d + B_d

	// 4. copy C_d back to host vector C

	// 5. synchronize host and device to ensure that transfer is finished

	// 6. free device vectors A_d, B_d, C_d

}