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
// Args: input audiodata, its size, window size, output array, threshold for accepting fund freq
void ce_sumsquares(double *arr, const size_t arrs, const int W, double *besttau, const double thr){
	double diff;
	double currtau;
	double partial2;
	// unfortunately, WIN complains:
	// double partial[W];
	// double out[W];
	double *out = malloc(sizeof(double) * W);
	double *partial = malloc(sizeof(double) * W);

	// start positions shift by half a window.
	// Hence, we can precalculate and re-use half of the internal stuff.
	// For start=0, initial half window:
	for(int tau=1; tau<W; tau++){
		partial[tau] = 0;
		for(int j=0; j<W/2; j++){
			diff = arr[j] - arr[tau + j];
			partial[tau] += diff * diff;
		}
	}

	int Wnum = 0;
	for(size_t start=0; start<arrs-2*W; start+=W/2){
		// first, compute the modified auto-correlation (Yin estimator)
		double cumsum = 0;
		out[0] = 1;
		for(int tau=1; tau<W; tau++){
			// half of the window is precalculated, so do the second half
			partial2 = 0;
			for(int j=W/2; j<W; j++){
				diff = arr[j + start] - arr[tau + j + start];
				partial2 += diff * diff;
			}
			// add both halves
			currtau = partial[tau] + partial2;
			// store the second half as the first half for next winodw
			partial[tau] = partial2;

			// accumulate unnormalized corrs
			cumsum += currtau;
			// store diff normalized by cumulative mean
			out[tau] = currtau / cumsum * tau;
		}

		// now find the best tau for this start
		int found = 0;
		besttau[Wnum] = -1;
		for(int tau=1; tau<W; tau++){
			// if we're in a trough and it starts rising, break
			if(found==1 && out[tau] >= out[tau-1]){
				// minimum is at tau-1.
				// Parabolic interpolation to improve the estimate
				double s0 = out[tau-2];
				double s2 = out[tau];
				double divisor = 2*out[tau-1] - s0 - s2;
				if(out[tau-1] >= thr || divisor==0){
					// autocorr too weak or numeric error due to div 0
					besttau[Wnum] = -1;
					break;
				} else {
					found = 2;
					besttau[Wnum] = tau-1 + (s2 - s0) / (2 * divisor);
					break;
				}
			}
			if(found==0 && out[tau] < thr){
				// found beginning of trough
				found = 1;
			}
		}
		Wnum++;
	}
	free(out);
	free(partial);

	// ALTERNATIVE APPROACH: save 25 % by skipping after best tau
	// we only look for the first smallest trough & stop if it's found - this marks it
	/* for(int tau=1; tau<W; tau++){
		if(found==2){
			// skip processing
			out[tau] = 99999.9;
		} else {
			currtau = 0;
			for(int j=0; j<W; j++){
				diff = arr[j] - arr[tau + j];
				currtau += diff * diff;
			}
			// accumulate unnormalized corrs
			cumsum += currtau;
			// output diff normalized by cumulative mean
			out[tau] = currtau / cumsum * tau;

			// if we're in a trough and it starts rising, break
			if(found==1 && out[tau] >= out[tau-1]){
				// minimum is at tau-1
				found = 2;
			}
			if(found==0 && out[tau] < thr){
				// found beginning of trough
				found = 1;
			}
		}
	}*/
}

/*
 *  performs IDWT for all modes
 *  
 *  The upsampling is performed by splitting filters to even and odd elements
 *  and performing 2 convolutions.
*/
int upsampling_convolution_valid_sf(const double * const input, const int N,
		const double * const filter, const int F,
		double * const output, const int O){

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
