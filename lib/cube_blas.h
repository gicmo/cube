
#ifndef CUBE_BLAS_H
#define CUBE_BLAS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cube.h"

enum _cube_blas_op_t {
  CUBE_BLAS_OP_N = 0,
  CUBE_BLAS_OP_T = 1,
  CUBE_BLAS_OP_C = 2
};

typedef enum _cube_blas_op_t cube_blas_op_t;

void cube_blas_d_iamax (cube_t       *ctx,
			int           n,
			const double *x, int incx,
			int          *result);

void cube_blas_d_gemm (cube_t *ctx,
		       cube_blas_op_t transa, cube_blas_op_t transb,
		       int m, int n, int k,
		       const double *alpha,
		       const double *A, int lda,
		       const double *B, int ldb,
		       const double *beta,
		       double *C, int ldc);

void cube_blas_d_axpy (cube_t       *ctx,
		       int           n,
		       const double *alpha,
		       const double *x, int incx,
		       double       *y, int incy);

void cube_blas_d_copy (cube_t       *ctx,
		       int           n,
		       const double *x, int incx,
		       double       *y, int incy);

void cube_blas_d_scal (cube_t       *ctx,
		       int           n,
		       const double *alpha,
		       double       *x, int incx);



#ifdef __cplusplus
}
#endif

#endif
