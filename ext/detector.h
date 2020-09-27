#include <math.h>

double cost0sq(double x, double wt);
double cost1sq(const double xs[], size_t len);
void findmincost(const double cs[], const double Fs[], const size_t starts[], size_t stlen, double *outcost, size_t *outstart, double segcosts[]);
double estsigma(const double x2s[], const size_t first, const size_t last);
int alg1_var(double xs[], size_t n, double mu0, double penalty, int outstarts[], int outends[], char outtypes[]);
int alg1_mean(double xs[], size_t n, double sd, double penalty);
int alg2_var(double xs[], size_t nn, size_t maxlb, double sigma0, double penalty_s, double penalty_n, int outstarts[], int outends[], char outtypes[]);
