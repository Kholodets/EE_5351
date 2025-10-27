extern const int NUM_BANKS = 16;
extern const int LOG_NUM_BANKS = 4;
extern const int TILE_SIZE = 1024;
// You can use any other block size you wish.
extern const int BLOCK_SIZE = 256;



// Device Functions






// Kernel Functions





// Host Helper Functions (allocate your own data structure...)

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, unsigned int* blockSums, int numElements)
{   //all pointers are to device memory regions



}
// **===-----------------------------------------------------------===**

// Use the function to allocate your block sums here
unsigned int* preallocBlockSums(int num_elements)
{
	unsigned int* bs_d = 0; //assign your device memory pointer to this variable
	// =========================================



	// =========================================
	return bs_d;
}

// Use the function to deallocate (free) your block sums
void deallocBlockSums(unsigned int* bs_d)
{
	
}
