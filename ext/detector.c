#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "detector.h"

#define LOG2PI 1.8378770664093453

struct ChpList {
    // size_t prev;
    size_t start;
    size_t end;
};

// Square cost functions (for use w/ change in mean)
// cost of points from background:
double cost0sq(const double x, const double wt){
    return pow(x - wt, 2);
}

// cost of points from segment, non-recursive:
double cost1sq(const double xs[], const size_t len){
    // get E(X)*n and E(X^2)*n
    double m=0;
    double m2=0;
    for(size_t i=0; i<len; i++){
        m += xs[i];
        m2 += xs[i]*xs[i];
    }
    
    // get the cost (variance*n)
    double cost=0;
    cost = (m2 - m*m/len);
    return cost;
}

// Cost functions for use w/ change in variance of N
// (these are -2 log L)
double cost0var(const double x2, const double s2t){
    return LOG2PI + log(s2t) + x2/s2t;
}

// double cost1var(const double x2s[], const size_t len){
//     return len*(LOG2PI + 1 + log(var(x2s)));
// }

void findmincost(const double cs[], const double Fs[], const size_t starts[], const size_t stlen, double *outcost, size_t *outstart, double segcosts[]){
    for(size_t i=0; i<stlen; i++){
        size_t t2 = starts[i];
        segcosts[t2] = Fs[t2] + cs[t2+1];
        // printf("--parts: F(i)=%.3f, cost1=%.3f\n", F[t2], cost1sq(xs+t2+1, t-t2));
        // keep track of min and argmin F_S
        if(segcosts[t2]<*outcost){
            *outcost = segcosts[t2];
            *outstart = t2;
        }
    }
}


// main loop over data - for change in VARIANCE
int alg1_var(double xs[], const size_t n, const double mu0, const double penalty){
    // center and square x
    double x2s[n];
    for(size_t i=0; i<n; i++){
        xs[i] = (xs[i]-mu0);
        x2s[i] = xs[i]*xs[i];
    }
    // precomputed segment costs
    // TODO add an initialization marker to spot cache misses
    double cs[n];
    double len;
    double m2;

    // output: estimate sequence
    double wts[n];
    wts[0] = x2s[0];

    // output: segments (list of starts and ends)
    // Could replace with two bare size_t[]
    struct ChpList chps[n];

    // output: history of best costs
    double F[n];
    F[0] = 0;

    // acceptable start positions to keep track of pruning
    size_t possiblestarts[n];
    possiblestarts[0] = 0;
    for(size_t i=1; i<n-1; i++){
        possiblestarts[i] = 0;
    }
    size_t numpossiblestarts = 1;
    size_t minstart = 0;

    // main DP loop:
    double bgcost;
    double segcosts[n];
    size_t bgsizes[n];
    bgsizes[0] = 1;
    for(size_t t=1; t<n; t++){
        // printf("Cycle %zu/%zu\n", t+1, n);
        // precompute costs for all possible segments ending here at t
        m2 = 0;
        for(size_t start=t; start>minstart; start--){
            // m += xs[start];
            m2 += x2s[start];
            len = t-start+1;
            // SSE = Var(x_i) * n
            cs[start] = len*(LOG2PI + 1 + log(m2/len));
        }
        // Could be good to check if any of the costs are -Inf given that logs are involved.

        // F_B = F(t-1) + C0(t)
        bgcost = F[t-1] + cost0var(x2s[t], wts[t-1]);

        // F_S = min F(t-k) + C(t-k+1:t) + beta
        double bestsegcost = INFINITY;
        size_t bestsegstart = -1;
        findmincost(cs, F, possiblestarts, numpossiblestarts, &bestsegcost, &bestsegstart, segcosts);
        // printf("bgcost: %.2f, segcost: %.2f + %.2f at t=%zu\n", bgcost, bestsegcost, penalty, bestsegstart+1);

        // determine best F = min(F_B, F_S)
        bestsegcost += penalty;
        if(bgcost < bestsegcost){
            // update cost
            F[t] = bgcost;
            // update wt
            bgsizes[t] = bgsizes[t-1]+1;
            wts[t] = wts[t-1] - (wts[t-1] - x2s[t])/bgsizes[t];
            // to avoid storing a lot of dynamically growing chp lists,
            // just store a pointer to the last list, plus at most one new chp pair
            // using 0 to indicate "not a chp", as the 0-th point is auto-set to background
            chps[t].start = 0;
            chps[t].end = 0;
        } else {
            // update cost
            F[t] = bestsegcost;
            // reset wt to wt[beststart]
            bgsizes[t] = bgsizes[bestsegstart];
            wts[t] = wts[bestsegstart];
            // store a pointer to the last list, plus at most one new chp pair - don't think this is needed
            // attach the new chp pair
            chps[t].start = bestsegstart+1;
            chps[t].end = t;
            // printf("! Ch.p. detected at %zu-%zu\n", bestsegstart+1, t);
        }

        // update and prune possible segment starts
        minstart = t; // useful for reducing the set for cost precomputing
        for(size_t i=0; i<numpossiblestarts; i++){
            if(F[t] <= segcosts[possiblestarts[i]]){
                // prune out this start time
                possiblestarts[i] = possiblestarts[--numpossiblestarts];
            }
            if(possiblestarts[i]<minstart){
                minstart = possiblestarts[i];
            }
        }
        possiblestarts[numpossiblestarts++] = t;
        // printf("Current wt: %f, F(t): %.2f\n", wts[t], F[t]);
    }
    printf("Final cost: %.3f\n", F[n-1]);

    // extract changepoints
    size_t i = n-1;
    printf("Detected segments:\n");
    while(i>0){
        if(chps[i].start>0){
            printf("* %zu-%zu\n", chps[i].start, chps[i].end);
            if(chps[i].start - 1 >= i){
                printf("ERROR: potential infinite recursion in changepoints\n");
                return 1;
            }
            i = chps[i].start - 1;
        } else {
            i--;
        }
    }
    return 0;
}



// main loop over data - for change in MEAN
int alg1_mean(double xs[], const size_t n, const double sd, const double penalty){
    // standardize x
    for(size_t i=0; i<n; i++){
        xs[i] = xs[i]/sd;
    }
    // precomputed segment costs
    // TODO add an initialization marker to spot cache misses
    // TODO move this back to stack?
    double *cs = malloc(n*sizeof(double));
    if(!cs){
        printf("ERROR: failed to allocate %ld bytes of memory\n", n*sizeof(double));
        return 1;
    }
    double len;
    double m;
    double m2;

    // output: estimate sequence
    double wts[n];
    wts[0] = xs[0];

    // output: segments (list of starts and ends)
    // Could replace with two bare size_t[]
    struct ChpList chps[n];

    // output: history of best costs
    double F[n];
    F[0] = 0;

    // acceptable start positions to keep track of pruning
    size_t possiblestarts[n];
    possiblestarts[0] = 0;
    for(size_t i=1; i<n-1; i++){
        possiblestarts[i] = 0;
    }
    size_t numpossiblestarts = 1;
    size_t minstart = 0;

    // main DP loop:
    double bgcost;
    double segcosts[n];
    size_t bgsizes[n];
    bgsizes[0] = 1;
    for(size_t t=1; t<n; t++){
        // printf("Cycle %zu/%zu\n", t+1, n);
        // precompute costs for all possible segments ending here at t
        m = 0;
        m2 = 0;
        for(size_t start=t; start>minstart; start--){
            m += xs[start];
            m2 += xs[start]*xs[start];
            len = t-start+1;
            cs[start] = (m2 - m*m/len);
        }

        // F_B = F(t-1) + C0(t)
        bgcost = F[t-1] + cost0sq(xs[t], wts[t-1]);

        // F_S = min F(t-k) + C(t-k+1:t) + beta
        double bestsegcost = INFINITY;
        size_t bestsegstart = -1;
        findmincost(cs, F, possiblestarts, numpossiblestarts, &bestsegcost, &bestsegstart, segcosts);
        // printf("bgcost: %.2f, segcost: %.2f + %.2f at t=%zu\n", bgcost, bestsegcost, penalty, bestsegstart+1);

        // determine best F = min(F_B, F_S)
        bestsegcost += penalty;
        if(bgcost < bestsegcost){
            // update cost
            F[t] = bgcost;
            // update wt
            bgsizes[t] = bgsizes[t-1]+1;
            wts[t] = wts[t-1] - (wts[t-1] - xs[t])/bgsizes[t];
            // to avoid storing a lot of dynamically growing chp lists,
            // just store a pointer to the last list, plus at most one new chp pair
            // chps[t].prev = t-1;
            // using 0 to indicate "not a chp", as the 0-th point is auto-set to background
            chps[t].start = 0;
            chps[t].end = 0;
        } else {
            // update cost
            F[t] = bestsegcost;
            // reset wt to wt[beststart]
            bgsizes[t] = bgsizes[bestsegstart];
            wts[t] = wts[bestsegstart];
            // store a pointer to the last list, plus at most one new chp pair - don't think this is needed
            // chps[t].prev = bestsegstart;
            // attach the new chp pair
            chps[t].start = bestsegstart+1;
            chps[t].end = t;
            // printf("! Ch.p. detected at %zu-%zu\n", bestsegstart+1, t);
        }

        // update and prune possible segment starts
        minstart = t; // useful for reducing the set for cost precomputing
        for(size_t i=0; i<numpossiblestarts; i++){
            if(F[t] <= segcosts[possiblestarts[i]]){
                // prune out this start time
                possiblestarts[i] = possiblestarts[--numpossiblestarts];
            }
            if(possiblestarts[i]<minstart){
                minstart = possiblestarts[i];
            }
        }
        possiblestarts[numpossiblestarts++] = t;
        // printf("Current wt: %f, F(t): %.2f\n", wts[t], F[t]);
    }
    free(cs);
    printf("Final cost: %.3f\n", F[n-1]);

    // extract changepoints
    size_t i = n-1;
    printf("Detected segments:\n");
    while(i>0){
        if(chps[i].start>0){
            printf("* %zu-%zu\n", chps[i].start, chps[i].end);
            if(chps[i].start - 1 >= i){
                printf("ERROR: potential infinite recursion in changepoints\n");
                return 1;
            }
            i = chps[i].start - 1;
        } else {
            i--;
        }
    }
    return 0;
}


// for testing separately:
// int main(int argc, char *argv[]){
//     FILE *fp;
//     size_t n = 10;
//     double xs[n];
// 
//     if((fp = fopen("/home/julius/Documents/gitrep/birdscape_main/tests/ce_seg_series.txt", "r"))==NULL){
//         printf("ERROR: could not open file");
//         return 1;
//     }
//     
//     for(size_t i=0; i<n; i++){
//         fscanf(fp, "%lf", xs+i);
//     }
//     fclose(fp);
// 
//     double sd = 1;
//     double penalty = 3*1.1*log(n);
//     printf("Detected settings: %zu datapoints, SD=%f, beta=%f\n", n, sd, penalty);
// 
//     alg1(xs, n, sd, penalty);
// }


