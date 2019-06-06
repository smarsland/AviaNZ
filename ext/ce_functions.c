#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// given array, calculates its cost as
// sum of elements above threshold,
// entropy, or SURE (some soft cost fn??)
double ce_getcost(double *in, size_t size, double threshold, char costfn, int step)
{
        double cost=0.0;
        if(costfn=='t'){
                // Threshold
                for(size_t i=0; i<size; i+=step){
                        if(fabs(in[i]) > threshold){
                                cost++;
                        }
                }
        } else if(costfn=='e'){
                // Entropy
                for(size_t i=0; i<size; i+=step){
                        if(in[i]!=0){
                                cost -= in[i] * in[i] * log(in[i] * in[i]);
                        }
                }
        } else {
                // SURE
                for(size_t i=0; i<size; i+=step){
                        double in2 = in[i]*in[i];
                        if(in2 <= threshold*threshold){
                                // d * (d<=t2): sum d which <=t2
                                cost += in2;
                        } else {
                                // 2 * ds + t2 * ds
                                cost += threshold*threshold + 2;
                        }
                }
                cost -= size;
        }

        return cost;
}

// Threshold a node in a wp tree and put output in a new wp tree
void ce_thresnode(double *in_array, double *out_array, size_t size, double threshold, char type)
{
        if(type=='h'){
                // Hard thresholding
                for(size_t i=0; i<size; i++){
                        if(fabs(in_array[i]) < threshold){
                                out_array[i] = 0.0;
                        } else {
                                out_array[i] = in_array[i];
                        }
                }
        } else if(type=='s'){
                // Soft thresholding
                for(size_t i=0; i<size; i++){
                        double tmp = fabs(in_array[i]) - threshold;
                        if(tmp<0){
                                out_array[i] = 0.0;
                        } else if(in_array[i]<0){
                                out_array[i] = -tmp;
                        } else {
                                out_array[i] = tmp;
                        }
                }
        }
}

// Threshold a node in a wp tree. Note: works inplace, unlike earlier
int ce_thresnode2(double *in_array, size_t size, double threshold, int type)
{
        if(type==2){
                // Hard thresholding
                for(size_t i=0; i<size; i++){
                        if(fabs(in_array[i]) < threshold){
                                in_array[i] = 0.0;
                        } else {
                                in_array[i] = in_array[i];
                        }
                }
                return 0;
        } else if(type==1){
                // Soft thresholding
                for(size_t i=0; i<size; i++){
                        double tmp = fabs(in_array[i]) - threshold;
                        if(tmp<0){
                                in_array[i] = 0.0;
                        } else if(in_array[i]<0){
                                in_array[i] = -tmp;
                        } else {
                                in_array[i] = tmp;
                        }
                }
                return 0;
        } else {
                return -1;
        }
}

// Main loop for "energy curve" - expanding envelope around waveform
void ce_energycurve(double *arrE, double *arrC, size_t N, int M)
{
        for(int i=M+1; i<N-M; i++){
                arrE[i] = arrE[i-1] - arrC[i - M - 1] + arrC[i + M];
        }
        // Normalize for M
        for(size_t i=0; i<N; i++){
                arrE[i] = arrE[i] / (2 * M);
        }
}

// Sum-of-squared differences loop for Yin's fund. freq. calculation
void ce_sumsquares(double *arr, int W, double *out){
        double diff;
        for(int tau=1; tau<W; tau++){
                for(int j=0; j<W; j++){
                        diff = arr[j] - arr[tau + j];
                        out[tau] += diff * diff;
                }
        }
}

/*
 *  performs IDWT for all modes
 *  
 *  The upsampling is performed by splitting filters to even and odd elements
 *  and performing 2 convolutions.
*/
// WINDOWS header version:
/*int upsampling_convolution_valid_sf(const double * const input, const int N,
                const double * const filter, const int F,
                double * const output, const int O){*/
int upsampling_convolution_valid_sf(const double * const restrict input, const int N,
                const double * const restrict filter, const int F,
                double * const restrict output, const int O){

        size_t o, i;

        // output_len expected to be 2*input_len - wavelet.rec_len + 2
        if(O != 2*N-F+2)
                return -1;
        if((F%2) || (N < F/2))
                return -1;
        
        // Perform only stage 2 - all elements in the filter overlap an input element.
        {
                for(o = 0, i = F/2 - 1; i < N; ++i, o += 2){
                        double sum_even = 0;
                        double sum_odd = 0;
                        size_t j;
                        // trick: skipping f[odd]*0 and f[even]*0 terms, due to 0s that come from upsampling
                        for(j = 0; j < F/2; ++j){
                                sum_even += filter[j*2] * input[i-j];
                                sum_odd += filter[j*2+1] * input[i-j];
                        }
                        output[o] += sum_even;
                        output[o+1] += sum_odd;
                }
        }
        return 0;
}
