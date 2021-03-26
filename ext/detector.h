#include <math.h>

double cost0sq(const double x, const double wt);
double cost1sq(const double xs[], const size_t len);
void findmincost(const double cs[], const double Fs[], const size_t starts[], const size_t stlen, double *outcost, size_t *outstart, double segcosts[]);
double estsigma(const double x2s[], const size_t first, const size_t last);
int alg1_var(double xs[], const size_t n, const size_t maxlb, const double mu0, const double penalty, int outstarts[], int outends[], char outtypes[]);
int alg1_mean(double xs[], const size_t n, const double sd, const double penalty);
int alg2_var(double xs[], const size_t nn, const size_t maxlb, const double sigma0, const double penalty_s, const double penalty_n, int outstarts[], int outends[], char outtypes[], const size_t printing);
