double ce_getcost(double *in_array, size_t size, double threshold, char costfn, int step);
double ce_thresnode(double *in_array, double *out_array, size_t size, double threshold, char type);
int ce_thresnode2(double *in_array, size_t size, double threshold, int type);
void ce_energycurve(double *arrE, double *arrC, size_t N, int M);
void ce_sumsquares(double *arr, const size_t arrs, const int W, double *besttau, const double thr);
// FOR WINDOWS:
// int upsampling_convolution_valid_sf(const double * const input, const size_t N, const double * const filter, const size_t F, double * const output, const size_t O);
int upsampling_convolution_valid_sf(const double * const input, const size_t N, const double * const filter, const size_t F, double * const output, const size_t O);
