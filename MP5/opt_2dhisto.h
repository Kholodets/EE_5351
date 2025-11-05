#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto( uint32_t *in_d, int n, uint32_t *bins_d);

/* Include below the function headers of any other functions that you implement */

void prep_hist(uint32_t *input[], int iw, int ih, uint32_t **in_d, uint32_t **bins_d);
void fin_hist(uint32_t *in_d, uint32_t *bins_d, uint8_t *bins_h);

#endif
