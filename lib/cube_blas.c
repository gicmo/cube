// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"

void
cube_blas_d_iamax (cube_t       *ctx,
		   int           n,
		   const double *x, int incx,
		   int          *result)
{
  cublasStatus_t status;
  
  if (! cube_context_check (ctx))
    return;

  status = cublasIdamax (ctx->h_blas,
			 n,
			 x, incx,
			 result);

  cube_blas_check (ctx, status);
}


void
cube_blas_d_gemm (cube_t *ctx,
		  cube_blas_op_t transa, cube_blas_op_t transb,
		  int m, int n, int k,
		  const double *alpha,
		  const double *A, int lda,
		  const double *B, int ldb,
		  const double *beta,
		  double *C, int ldc)
{
  cublasStatus_t status;
  cublasOperation_t ta,  tb;
  
  if (! cube_context_check (ctx))
    return;

  ta = (cublasOperation_t) transa;
  tb = (cublasOperation_t) transb;

  status = cublasDgemm (ctx->h_blas,
			ta, tb,
			m, n, k,
			alpha,
			A, lda,
			B, ldb,
			beta,
			C, ldc);

  cube_blas_check (ctx, status);
}


void
cube_blas_d_axpy (cube_t       *ctx,
		  int           n,
		  const double *alpha,
		  const double *x, int incx,
		  double       *y, int incy)
{
  cublasStatus_t status;
  
  if (! cube_context_check (ctx))
    return;

  status = cublasDaxpy (ctx->h_blas,
			n,
			alpha,
			x, incx,
			y, incy);

  cube_blas_check (ctx, status);
}


void
cube_blas_d_copy (cube_t       *ctx,
		  int           n,
		  const double *x, int incx,
		  double       *y, int incy)
{
  cublasStatus_t status;
  
  if (! cube_context_check (ctx))
    return;

  status = cublasDcopy (ctx->h_blas,
			n,
			x, incx,
			y, incy);

  cube_blas_check (ctx, status);
}

void
cube_blas_d_scal (cube_t *ctx,
		  int     n,
		  const double *alpha,
		  double *x, int incx)
{
  cublasStatus_t status;

  if (! cube_context_check (ctx))
    return;

  status = cublasDscal (ctx->h_blas,
			n,
			alpha,
			x, incx);

  
  cube_blas_check (ctx, status);
}
