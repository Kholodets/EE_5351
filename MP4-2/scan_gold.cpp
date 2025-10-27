#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Output i is the sum of all inputs up to and including input i
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len) 
{
  reference[0] = idata[0];
  unsigned int total_sum = idata[0];
  for( unsigned int i = 1; i < len; ++i) 
  {
      total_sum += idata[i];
      reference[i] = reference[i-1] + idata[i];
  }
  if (total_sum != reference[len-1])
      printf("Warning: exceeding single-precision accuracy.  Scan will have reduced precision.\n");
  
}

