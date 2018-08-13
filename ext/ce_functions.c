#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// given array, calculates its cost as
// sum of elements above threshold,
// entropy, or SURE (some soft cost fn??)
double ce_getcost(double *in, int size, double threshold, char costfn)
{
	int cost=0;
	if(costfn=='t'){
		// Threshold
		for(int i=0; i<size; i++){
			if(abs(in[i]) > threshold){
				cost++;
			}
		}
	} else if(costfn=='e'){
		// Entropy
		for(int i=0; i<size; i++){
			if(in[i]!=0){
				cost -= in[i] * in[i] * log(in[i] * in[i]);
			}
		}
	} else {
		// SURE
		for(int i=0; i<size; i++){
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

// Threshold a node in a wp tree and put output to new_wp tree
void ce_thresnode(double *in_array, double *out_array, int size, double threshold, char type)
{
	if(type=='h'){
		// Hard thresholding
		for(int i=0; i<size; i++){
			if(abs(in_array[i]) < threshold){
				out_array[i] = 0.0;
			} else {
				out_array[i] = in_array[i];
			}
		}
	} else if(type=='s'){
		// Soft thresholding
		for(int i=0; i<size; i++){
			double tmp = abs(in_array[i]) - threshold;
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

