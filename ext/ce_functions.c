#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// given array, calculates its cost as
// sum of elements above threshold,
// entropy, or SURE (some soft cost fn??)
double ce_getcost(double *in, int size, double threshold, char costfn, int step)
{
        double cost=0.0;
        if(costfn=='t'){
                // Threshold
                for(int i=0; i<size; i+=step){
                        if(fabs(in[i]) > threshold){
                                cost++;
                        }
                }
        } else if(costfn=='e'){
                // Entropy
                for(int i=0; i<size; i+=step){
                        if(in[i]!=0){
                                cost -= in[i] * in[i] * log(in[i] * in[i]);
                        }
                }
        } else {
                // SURE
                for(int i=0; i<size; i+=step){
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
void ce_thresnode(double *in_array, double *out_array, int size, double threshold, char type)
{
        if(type=='h'){
                // Hard thresholding
                for(int i=0; i<size; i++){
                        if(fabs(in_array[i]) < threshold){
                                out_array[i] = 0.0;
                        } else {
                                out_array[i] = in_array[i];
                        }
                }
        } else if(type=='s'){
                // Soft thresholding
                for(int i=0; i<size; i++){
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
void ce_thresnode2(double *in_array, int size, double threshold, char type)
{
        if(type=='h'){
                // Hard thresholding
                for(int i=0; i<size; i++){
                        if(fabs(in_array[i]) < threshold){
                                in_array[i] = 0.0;
                        } else {
                                in_array[i] = in_array[i];
                        }
                }
        } else if(type=='s'){
                // Soft thresholding
                for(int i=0; i<size; i++){
                        double tmp = fabs(in_array[i]) - threshold;
                        if(tmp<0){
                                in_array[i] = 0.0;
                        } else if(in_array[i]<0){
                                in_array[i] = -tmp;
                        } else {
                                in_array[i] = tmp;
                        }
                }
        }
}

// Main loop for "energy curve" - expanding envelope around waveform
void ce_energycurve(double *arrE, double *arrC, int N, int M)
{
        for(int i=M+1; i<N-M; i++){
                arrE[i] = arrE[i-1] - arrC[i - M - 1] + arrC[i + M];
        }
        // Normalize for M
        for(int i=0; i<N; i++){
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
