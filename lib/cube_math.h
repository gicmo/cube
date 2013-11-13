#ifndef CUBE_MATH_H
#define CUBE_MATH_H

#include <cube.h>

#ifdef __cplusplus
extern "C" {
#endif

  void cube_host_dgesvd (char jobu, char jobvt, int m, int n, double *A, int lda, double *s, double *U, int ldu, double *Vt, int ldvt, int *info);
  void cube_host_dgemm (cube_t *ctx, char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);

  void cube_host_dvmean (cube_t *ctx, int n, double *v, int incv, double *mean);
  void cube_host_dvizm (cube_t *ctx, int n, double *v, int incv);
  void cube_host_dvstd (cube_t *ctx, int n, double *v, int incv, double *std);
  void cube_host_dvvar (cube_t *ctx, int n, double *v, int incv, double *var);

  void cube_host_dscal (cube_t *ctx, int n, double a, double *v, int incv);
  void cube_host_sscal (cube_t *ctx, int n, float a, float *v, int incv);
  void cube_host_isamax (cube_t *ctx, int n, float *v, int incv, int *idx);

  void cube_gpu_dgesdd (cube_t *ctx, char jobu, int m, int n, double *A, int lda, double *s, double *U, int ldu, double *Vt, int ldvt);

  #ifndef min
  #define min(a,b) ((a)>(b)?(b):(a))
  #endif

  #ifndef max
  #define max(a,b) ((a)>(b)?(a):(b))
  #endif

#ifdef __cplusplus
}
#endif

#endif
