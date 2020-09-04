#include <math.h>
#include <stdio.h>
#include <string.h>
#include "detector.h"

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


// main loop over data
int alg1(double xs[], const size_t n, const double sd, const double penalty){
    // standardize x
    for(size_t i=0; i<n; i++){
        xs[i] = xs[i]/sd;
    }
    // precompute segment costs
    double cs[n][n];
    for(size_t start=1; start<n; start++){
        double m = 0;
        double m2 = 0;
        size_t len;
        for(size_t end=start; end<n; end++){
            m += xs[end];
            m2 += xs[end]*xs[end];
            len = end-start+1;
            cs[start][end] = (m2 - m*m/len);
        }
    }

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
    int possiblestarts[n];
    possiblestarts[0] = 0;
    for(size_t i=1; i<n-1; i++){
        possiblestarts[i] = 0;
    }
    size_t numpossiblestarts = 1;

    // main DP loop:
    double bgcost;
    double segcosts[n];
    size_t bgsizes[n];
    bgsizes[0] = 1;
    for(size_t t=1; t<n; t++){
        // printf("Cycle %zu/%zu\n", t+1, n);
        // F_B = F(t-1) + C0(t)
        bgcost = F[t-1] + cost0sq(xs[t], wts[t-1]);

        // F_S = min F(t-k) + C(t-k+1:t) + beta
        double bestsegcost = INFINITY;
        size_t bestsegstart = -1;
        for(size_t i=0; i<numpossiblestarts; i++){
            int t2 = possiblestarts[i];
            segcosts[t2] = F[t2] + cs[t2+1][t];
            // or, without precomputing:
            // segcosts[t2] = F[t2] + cost1sq(xs + t2 + 1, t-t2);
            // printf("--parts: F(i)=%.3f, cost1=%.3f\n", F[t2], cost1sq(xs+t2+1, t-t2));
            // keep track of min and argmin F_S
            if(segcosts[t2]<bestsegcost){
                bestsegcost = segcosts[t2];
                bestsegstart = t2;
            }
        }
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
        for(size_t i=0; i<numpossiblestarts; i++){
            if(F[t] <= segcosts[possiblestarts[i]]){
                // prune out this start time
                possiblestarts[i] = possiblestarts[--numpossiblestarts];
            }
        }
        possiblestarts[numpossiblestarts++] = t;
        // printf("Current wt: %f, F(t): %.2f\n", wts[t], F[t]);
    }

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


int main(int argc, char *argv[]){
    FILE *fp;
    size_t n = 10;
    double xs[n];

    if((fp = fopen("/home/julius/Documents/gitrep/birdscape_main/tests/ce_seg_series.txt", "r"))==NULL){
        printf("ERROR: could not open file");
        return 1;
    }
    
    for(size_t i=0; i<n; i++){
        fscanf(fp, "%lf", xs+i);
    }
    fclose(fp);

    double sd = 1;
    double penalty = 3*1.1*log(n);
    printf("Detected settings: %zu datapoints, SD=%f, beta=%f\n", n, sd, penalty);

    alg1(xs, n, sd, penalty);
}



