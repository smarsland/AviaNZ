#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "detector.h"

#define LOG2PI 1.8378770664093453

// typedef enum {
//     NUIS,
//     SIG
// } chptype;

struct ChpList {
    // size_t prev;
    size_t start;
    // size_t end; // equals the time when this chp was stored
    // chptype type; // can be determined from end-start
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
// (these are -2 log L - LOG2PI)
double cost0var(const double x2, const double s2t){
    return log(s2t) + x2/s2t;
}

double estsigma(const double x2s[], const size_t first, const size_t last){
    double sigma = 0;
    for(size_t j=first; j<=last; j++){
        sigma += x2s[j];
    }
    sigma /= last-first+1;
    sigma = sqrt(sigma);
    return sigma;
}


// double cost1var(const double x2s[], const size_t len){
//     return len*(LOG2PI + 1 + log(var(x2s)));
// }

void findmincost(const double cs[], const double Fs[], const size_t starts[], const size_t stlen, double *outcost, size_t *outstart, double segcosts[]){
    for(size_t i=0; i<stlen; i++){
        size_t t2 = starts[i];
        segcosts[t2] = Fs[t2] + cs[t2+1];
        // printf("--parts: F(i)=%.3f, cost1=%.3f\n", Fs[t2], cs[t2+1]);
        // keep track of min and argmin F_S
        if(segcosts[t2]<*outcost){
            *outcost = segcosts[t2];
            *outstart = t2;
        }
    }
}


// main loop over data - for change in VARIANCE
int alg1_var(double xs[], const size_t n, const size_t maxlb, const double mu0, const double penalty, int outstarts[], int outends[], char outtypes[]){
    // center and square x
    double *x2s  = malloc(n * sizeof(double));
    for(size_t i=0; i<n; i++){
        xs[i] = (xs[i]-mu0);
        x2s[i] = xs[i]*xs[i];
    }
    // precompute segment costs
    // TODO add an initialization marker to spot cache misses
    double *cs = malloc(n * sizeof(double));
    double len;
    double m2;

    // output: estimate sequence
    double *wts = malloc(n * sizeof(double));
    wts[0] = x2s[0];

    // output: segments (list of starts and ends)
    // Could replace with two bare size_t[]
    struct ChpList *chps = malloc(n * sizeof(struct ChpList));

    // output: history of best costs
    double *F = malloc(n * sizeof(double));
    F[0] = 0;

    // acceptable start positions to keep track of pruning
    size_t *possiblestarts = malloc(n * sizeof(size_t));
    possiblestarts[0] = 0;
    for(size_t i=1; i<n-1; i++){
        possiblestarts[i] = 0;
    }
    size_t numpossiblestarts = 1;
    size_t minstart = 0;

    // main DP loop:
    double bgcost;
    double *segcosts = malloc(n * sizeof(double));
    size_t *bgsizes = malloc(n * sizeof(size_t));
    bgsizes[0] = 1;
    for(size_t t=1; t<n; t++){
        // printf("----Cycle %zu/%zu----\n", t+1, n);
        // precompute costs for all possible segments ending here at t
        m2 = 0;
        for(size_t start=t; start>minstart; start--){
            // m += xs[start];
            m2 += x2s[start];
            len = t-start+1;
            // SSE = Var(x_i) * n
            cs[start] = len*(1 + log(m2/len));
        }
        // Could be good to check if any of the costs are -Inf given that logs are involved.

        // F_B = F(t-1) + C0(t)
        bgcost = F[t-1] + cost0var(x2s[t], wts[t-1]);

        // F_S = min F(t-k) + C(t-k+1:t) + beta
        double bestsegcost = INFINITY;
        size_t bestsegstart;
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
            // chps[t].end = 0;
        } else {
            // update cost
            F[t] = bestsegcost;
            // reset wt to wt[beststart]
            bgsizes[t] = bgsizes[bestsegstart];
            wts[t] = wts[bestsegstart];
            // store a pointer to the last list, plus at most one new chp pair - don't think this is needed
            // attach the new chp pair
            chps[t].start = bestsegstart+1;
            // chps[t].end = t;
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
    printf("Final cost: %.2f, final wt: %.4f\n", F[n-1], wts[n-1]);
    free(x2s);
    free(cs);
    free(wts);
    free(F);
    free(possiblestarts);
    free(segcosts);
    free(bgsizes);

    // extract changepoints
    size_t i = n-1;
    size_t outnum = 0;
    printf("Detected segments:\n");
    while(i>0){
        if(chps[i].start>0){
            // no need to store chp end separately, as i is exactly that
            printf("* %zu-%zu\n", chps[i].start, i);
            outstarts[outnum] = chps[i].start;
            outends[outnum] = i;
            // no need to store the type since it can be recreated from length
            char thischptype = i-chps[i].start<=maxlb ? 's' : 'n';
            outtypes[outnum] = thischptype;
            outnum++;
            if(chps[i].start - 1 >= i){
                printf("ERROR: potential infinite recursion in changepoints\n");
                return 1;
            }
            i = chps[i].start - 1;
        } else {
            i--;
        }
    }
    free(chps);
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
    double *cs = malloc(n * sizeof(double));
    double len;
    double m;
    double m2;

    // output: estimate sequence
    double *wts = malloc(n * sizeof(double));
    wts[0] = xs[0];

    // output: segments (list of starts and ends)
    // Could replace with two bare size_t[]
    struct ChpList *chps = malloc(n * sizeof(struct ChpList));

    // output: history of best costs
    double *F = malloc(n * sizeof(double));
    F[0] = 0;

    // acceptable start positions to keep track of pruning
    size_t *possiblestarts = malloc(n * sizeof(size_t));
    possiblestarts[0] = 0;
    for(size_t i=1; i<n-1; i++){
        possiblestarts[i] = 0;
    }
    size_t numpossiblestarts = 1;
    size_t minstart = 0;

    // main DP loop:
    double bgcost;
    double *segcosts = malloc(n * sizeof(double));
    size_t *bgsizes = malloc(n * sizeof(size_t));
    bgsizes[0] = 1;
    for(size_t t=1; t<n; t++){
        // printf("----Cycle %zu/%zu----\n", t+1, n);
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
        size_t bestsegstart;
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
            // chps[t].end = 0;
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
            // chps[t].end = t;
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
    printf("Final cost: %.2f, final wt: %.4f\n", F[n-1], wts[n-1]);
    free(cs);
    free(wts);
    free(F);
    free(possiblestarts);
    free(segcosts);
    free(bgsizes);
    
    // extract changepoints
    size_t i = n-1;
    printf("Detected segments:\n");
    while(i>0){
        if(chps[i].start>0){
            printf("* %zu-%zu \n", chps[i].start, i);
            if(chps[i].start - 1 >= i){
                printf("ERROR: potential infinite recursion in changepoints\n");
                return 1;
            }
            i = chps[i].start - 1;
        } else {
            i--;
        }
    }
    free(chps);
    return 0;
}


int alg2_var(double xs[], const size_t nn, const size_t maxlb, const double sigma0, const double penalty_s, const double penalty_n, int outstarts[], int outends[], char outtypes[], const size_t printing){
    // xs, nn: centered data (nn points)
    // sigma0: sigma^2_0, given
    // output: segments (list of starts and ends and types)
    struct ChpList *chps = malloc(nn * sizeof(struct ChpList));

    // square x
    double *x2s = malloc(nn * sizeof(double));
    for(size_t i=0; i<nn; i++){
        x2s[i] = xs[i]*xs[i];
    }

    double *F  = malloc(nn * sizeof(double));
    F[0] = 0;

    // arrays of outputs and parameters for each detector
    // each detector gets a slice of nn elements starting from i*nn
    double *detwts = malloc(nn * nn * sizeof(double));
    double *detFs = malloc(nn * nn * sizeof(double));
    double *detsegcosts = malloc(nn * nn * sizeof(double));
    size_t *detbgsizes = malloc(nn * nn * sizeof(size_t));
    size_t *detpostarts = malloc(nn * nn * sizeof(size_t));
    if(!detpostarts || !detbgsizes || !detsegcosts || !detFs || !detwts){
        printf("ERROR: could not allocate memory of size %zu x %zu x %zu bytes\n", nn, nn, sizeof(double));
        return 1;
    }
    struct ChpList *detchps = malloc(nn * nn * sizeof(struct ChpList));
    if(!detchps){
        printf("ERROR: could not allocate memory of size %zu x %zu x %zu bytes\n", nn, nn, sizeof(struct ChpList));
        return 1;
    }
    for(size_t i=0; i<nn*nn; i++){
        detpostarts[i] = 0;
        detwts[i] = 9999; // cache miss marker. Could remove later
    }
    for(size_t i=0; i<nn; i++){
        detwts[i*nn + i] = x2s[i];
        detFs[i*nn + i] = 1 + log(x2s[i]);
        detbgsizes[i*nn + i] = 1;
    }

    // other detector-specific values
    size_t *detnumpostarts = malloc(nn * sizeof(size_t));
    for(size_t i=0; i<nn; i++){
        detnumpostarts[i] = 0;
    }

    // for precomputing costs
    double m2;
    double len;
    double *cs = malloc(nn * sizeof(double));
    double *vars = malloc(nn * sizeof(double));

    // acceptable start positions (for S) to keep track of pruning
    size_t *possiblestarts = malloc(nn * sizeof(size_t));
    possiblestarts[0] = 0;
    for(size_t i=1; i<nn-1; i++){
        possiblestarts[i] = 0;
    }
    size_t numpossiblestarts = 1;

    // acceptable start positions (for N)
    size_t *possiblestarts_n = malloc(nn * sizeof(size_t));
    possiblestarts_n[0] = 0;
    for(size_t i=1; i<nn-1; i++){
        possiblestarts_n[i] = 0;
    }
    size_t numpossiblestarts_n = 1;

    // minstart only tracks nuis starts as they are longer anyway
    size_t minstart = 0;

    // main DP loop
    double bgcost;
    double *segcosts = malloc(nn * sizeof(double));
    for(size_t tt=1; tt<nn; tt++){
        // printf("----Cycle %zu/%zu----\n", tt+1, nn);
        // precompute costs for all possible segments ending here at t
        m2 = 0;
        // Note: minstart fixed for now. Cannot really prune this,
        // b/c burnin iterations do not prune possible S starts.
        minstart = tt>maxlb ? tt-maxlb : 0;
        // printf("DEBUG will precompute C down to %zu\n", minstart);
        for(size_t start=tt; start>minstart; start--){
            m2 += x2s[start];
            len = tt-start+1;
            // when we want to limit the S+N directions, also need to store seg. variance:
            vars[start] = m2/len;
            cs[start] = len*(1 + log(m2/len));
        }

        // F_B = F(t-1) + C0(t)
        bgcost = F[tt-1] + cost0var(x2s[tt], sigma0);

        // F_S = min F(t-k) + C(t-k+1:t) + beta
        // this will loop over all possible sig. segment starts:
        double bestsegcost = INFINITY;
        size_t bestsegstart = 0;
        findmincost(cs, F, possiblestarts, numpossiblestarts, &bestsegcost, &bestsegstart, segcosts);
        // printf("bgcost: %.2f, segcost: %.2f + %.2f at t=%zu\n", bgcost, bestsegcost, penalty_s, bestsegstart+1);

        // loop over all possible nuis. segment starts:
        // basically, all points 1:maxlb and maxlb: are used in the loop (for doing detector.step),
        // but only points maxlb+1: are treated as real nuis candidates and get pruned
        double bestnuiscost = INFINITY;
        size_t bestnuisstart = 0;
        for(size_t i=0; i<numpossiblestarts_n; i++){
            double detbgcost;
            size_t t2 = possiblestarts_n[i];
            // printf("DEBUG Working on potential nuisance at %zu : %zu\n", t2+1, tt);
            // start of the nn-size block alloc'd for this detector
            // (in row-column notation, 0-based indices on the matrices are:
            //  row=first element in the detector
            //  column=last element in the detector
            // So it is upper triangular.
            size_t t2det = (t2+1)*nn;

            // if tested seg l=1, don't test segments etc.
            // (detector F, wt values are already initialized with 0)
            if(t2+1 < tt){
                detbgcost = detFs[t2det + tt-1] + cost0var(x2s[tt], detwts[t2det + tt-1]);

                // step the detector (which now contains points t2-tt),
                // i.e. loop over last (S) segment starts for that detector
                double detbestsegcost = INFINITY;
                size_t detbestsegstart = 0;
                for(size_t j=0; j<detnumpostarts[t2+1]; j++){
                    size_t t3 = detpostarts[t2det + j];
                    // printf("- possible start: %zu (%zu total)\n", t3, detnumpostarts[t2+1]);
                    if(t3<minstart){
                        printf("ERROR: fatal cache miss for precomputed cost at %zu < %zu, detector %zu\n", t3, minstart, t2);
                        return(1);
                    }
                    // cs: precomputed cost for segs ending at tt
                    detsegcosts[t2det + t3] = detFs[t2det + t3] + cs[t3+1];

                    // If we want to limit effect directions to theta_S > theta_N > theta_0 or vice versa:
                    // for any segment which had theta_0 < theta_S < theta_N, the cost of segment cannot be minimal
                    // (b/c it equals C(no segment)+beta). But don't set the cost to inf to avoid pruning it.
                    if((detwts[t2det+tt-1]>sigma0 && vars[t3+1] <= detwts[t2det+tt-1]) || \
                            (detwts[t2det+tt-1]<sigma0 && vars[t3+1] >= detwts[t2det+tt-1])){
                        continue;
                    }

                    // keep track to get min and argmin F_S for that detector
                    if(detsegcosts[t2det + t3] < detbestsegcost){
                        detbestsegcost = detsegcosts[t2det + t3];
                        detbestsegstart = t3;
                    }
                }
                // printf("- chosen seg start for this N: %zu\n", detbestsegstart);

                // determine best F = min(F_B, F_S)
                detbestsegcost += penalty_s;
                if(detbgcost < detbestsegcost){
                    // update cost
                    detFs[t2det + tt] = detbgcost;
                    // update wt
                    detbgsizes[t2det + tt] = detbgsizes[t2det + tt-1]+1;
                    detwts[t2det + tt] = detwts[t2det+tt-1] - (detwts[t2det+tt-1] - x2s[tt])/detbgsizes[t2det + tt];
                    // using 0 to indicate "not a chp", as the 0-th point is auto-set to background
                    detchps[t2det + tt].start = 0;
                } else {
                    // update cost
                    detFs[t2det + tt] = detbestsegcost;
                    // reset wt to wt[beststart]
                    detbgsizes[t2det + tt] = detbgsizes[t2det + detbestsegstart];
                    detwts[t2det + tt] = detwts[t2det + detbestsegstart];
                    // attach the new chp pair
                    detchps[t2det + tt].start = detbestsegstart+1;
                    // detchps[t2det + tt].end = tt;
                    // printf("! Ch.p. detected at %zu-%zu\n", detbestsegstart+1, tt);
                }
            }

            // update and prune possible segment starts
            // btw note the lit loop syntax
            for(size_t j=detnumpostarts[t2+1]; j-- > 0; ){
                if(detFs[t2det + tt] <= detsegcosts[t2det + detpostarts[t2det + j]]){
                    // prune out this start time
                    // printf("DEBUG - pruning out %zu, replacing with %zu\n", detpostarts[t2det + j], detpostarts[t2det + detnumpostarts[t2+1]-1]);
                    detpostarts[t2det + j] = detpostarts[t2det + --detnumpostarts[t2+1]];
                } else if(tt+1-detpostarts[t2det + j] > maxlb){
                    // remove one segment start to limit lookback
                    // (not super efficient but can't do better w/o sorting)
                    // printf("DEBUG - start %zu for detector %zu exceeds maxlb = %zu\n", detpostarts[t2det+j], t2, maxlb);
                    detpostarts[t2det + j] = detpostarts[t2det + --detnumpostarts[t2+1]];
                }
            }
            detpostarts[t2det + detnumpostarts[t2+1]++] = tt;
            // printf("- Current wt: %f, F(t): %.2f\n", detwts[t2det + tt], detFs[t2det + tt]);
            // printf("- total number of possible det. starts: %zu, last: %zu\n", detnumpostarts[t2+1], detpostarts[t2det + detnumpostarts[t2+1]-1]);
            if(detwts[t2det+tt]==9999){
                printf("ERROR: fatal cache miss in cycle %zu\n", tt);
                return 1;
            }

            // check if this detector is the best nuisance
            // (nuisances of length 1:maxlb are only for burn-in and not real candidates)
            if(tt-t2 > maxlb){
                if(F[t2] + detFs[t2det + tt] < bestnuiscost){
                    bestnuiscost = F[t2] + detFs[t2det + tt];
                    bestnuisstart = t2;
                }
            }
        }
            
        // printf("bgcost: %.2f, segcost: %.2f + %.2f at t=%zu, nuiscost: %.2f + %.2f at t=%zu\n", bgcost, bestsegcost, penalty_s, bestsegstart+1, bestnuiscost, penalty_n, bestnuisstart+1);

        // determine best F = min(F_B, F_S, F_N)
        bestsegcost += penalty_s;
        bestnuiscost += penalty_n;

        if(bgcost < bestsegcost && bgcost < bestnuiscost){
            // update cost
            F[tt] = bgcost;
            // using 0 to indicate "not a chp", as the 0-th point is auto-set to background
            chps[tt].start = 0;
        } else if (bestsegcost<bestnuiscost){
            // update cost
            F[tt] = bestsegcost;
            // attach the new chp pair
            chps[tt].start = bestsegstart+1;
            // printf("! Ch.p. detected at %zu-%zu, type SIGNAL\n", bestsegstart+1, tt);
        } else {
            // update cost
            F[tt] = bestnuiscost;
            // attach the new chp pair
            chps[tt].start = bestnuisstart+1;
            // printf("! Ch.p. detected at %zu-%zu, type NUIS\n", bestnuisstart+1, tt);
        }

        // update and prune possible segment starts
        // minstart = tt; // useful for reducing the set for cost precomputing
        // minstart = tt>maxlb ? tt-maxlb : 0;
        for(size_t i=numpossiblestarts; i-- > 0;){
            if(F[tt] <= segcosts[possiblestarts[i]]){
                // prune out this start time
                possiblestarts[i] = possiblestarts[--numpossiblestarts];
            } else if(tt+1-possiblestarts[i] > maxlb){
                // remove one segment start to limit lookback
                // (not super efficient but can't do better w/o sorting)
                possiblestarts[i] = possiblestarts[--numpossiblestarts];
            }
            // if(possiblestarts[i]<minstart){
            //     minstart = possiblestarts[i];
            // }
        }
        // printf("After pruning signal starts, %zu remain, starting at %zu\n", numpossiblestarts, minstart);

        // add one segment start
        possiblestarts[numpossiblestarts++] = tt;

        // update and prune possible nuisance starts
        for(size_t i=numpossiblestarts_n; i-- >0;){
            size_t t2 = possiblestarts_n[i];
            // nuisances that will be no longer than maxlb in the next cycle
            // are just burnins (not used for F_N), so shouldn't be pruned
            if(tt-t2 > maxlb){
                if(F[tt] <= F[t2] + detFs[(t2+1)*nn + tt]){
                    // prune out this start time
                    possiblestarts_n[i] = possiblestarts_n[--numpossiblestarts_n];
                }
            }
        }
        possiblestarts_n[numpossiblestarts_n++] = tt;
        // printf("After pruning nuisance starts, %zu remain\n", numpossiblestarts_n);
    }
    // printf("Final cost: %.3f\n", F[nn-1]);

    free(detwts);
    free(detFs);
    free(detsegcosts);
    free(detbgsizes);
    free(detpostarts);
    free(detnumpostarts);
    free(x2s);
    free(F);
    free(cs);
    free(vars);
    free(segcosts);
    free(possiblestarts);
    free(possiblestarts_n);

    // extract changepoints
    size_t i = nn-1;
    if(printing){
         printf("Detected segments:\n");
    }
    size_t outnum = 0;
    while(i>0){
        if(chps[i].start>0){
            // return this segment
            outstarts[outnum] = (int) chps[i].start;
            outends[outnum] = (int) i;
            // no need to store the type since it can be recreated from length
            char thischptype = i-chps[i].start<=maxlb ? 's' : 'n';
            outtypes[outnum] = thischptype;
            if(printing){
                printf("* %zu. %zu-%zu %s   \t\t", outnum, chps[i].start, i, thischptype=='s'?"SIG ":"NUIS");
                printf("   * theta: %.2f\n", estsigma(x2s, chps[i].start, i));
            }
            outnum++;

            // just in case
            if(chps[i].start - 1 >= i){
                printf("ERROR: potential infinite recursion in changepoints\n");
                return 1;
            }

            // if this is a nuis, look for overlapping signal segments:
            if(thischptype=='n'){
                size_t i2 = i;
                size_t i2det = chps[i].start*nn;
                while(i2>chps[i].start){
                    if(detchps[i2det + i2].start>0){
                        if(printing){
                            printf("* %zu. %zu-%zu %s \t", outnum, detchps[i2det + i2].start, i2, "SIG on NUIS");
                            printf("   * theta: %.2f\n", estsigma(x2s, detchps[i2det + i2].start, i2));
                        }
                        outstarts[outnum] = (int) detchps[i2det + i2].start;
                        outends[outnum] = (int) i2;
                        outtypes[outnum] = 'o';
                        outnum++;

                        // just in case
                        if(detchps[i2det + i2].start - 1 >= i2){
                            printf("ERROR: potential infinite recursion in changepoints\n");
                            return 1;
                        }
                        i2 = detchps[i2det + i2].start - 1;
                    } else {
                        i2--;
                    }
                    i2--;
                }
            }
            i = chps[i].start - 1;
        } else {
            i--;
        }
    }
    free(detchps);
    free(chps);
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


