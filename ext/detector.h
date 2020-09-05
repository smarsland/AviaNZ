#include <math.h>

double cost0sq(double x, double wt);
double cost1sq(const double xs[], size_t len);
void findmincost(const double cs[], const double Fs[], const size_t starts[], size_t stlen, double *outcost, size_t *outstart, double segcosts[]);
int alg1_var(double xs[], size_t n, double mu0, double penalty);
int alg1_mean(double xs[], size_t n, double sd, double penalty);
