// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include <cube_math.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#elif HAVE_ACML
#include <acml.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#endif

#ifdef HAVE_CULA
#include <cula_lapack_device.h>
#endif

#include "cube_private.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define min(a,b) ((a)>(b)?(b):(a))

void
cube_host_dgesvd (char jobu, char jobvt, int m, int n, double *A, int lda, double *s, double *U, int ldu, double *Vt, int ldvt, int *info)
{
#ifdef __APPLE__
  double  cwork;
  double *work;
  int     lwork;

  dgesvd_(&jobu, &jobvt,
	  &m, &n, A, &lda,
	  s,
	  U, &ldu,
	  Vt, &ldvt,
	  &cwork, &lwork, info);

  lwork = (int) cwork;

  work = malloc (lwork * sizeof (double));

  dgesvd_(&jobu, &jobvt,
	  &m, &n, A, &lda,
	  s,
	  U, &ldu,
	  Vt, &ldvt,
	  work, &lwork, info);

  free (work);
#elif defined (HAVE_ACML)
  dgesvd (jobu, jobvt, m, n, A, lda, s, U, ldu, Vt, ldvt, info);
#else
  double superb[min(m,n)];
  *info = LAPACKE_dgesvd (LAPACK_COL_MAJOR, jobu, jobvt, m, n, A, lda,
			  s, U, ldu, Vt, ldvt, superb);
#endif
}

void
cube_host_dgesdd (char jobz, int m, int n, double *A, int lda, double *s, double *U, int ldu, double *Vt, int ldvt, int *info)
{
#ifdef __APPLE__
  double  cwork;
  double *work;
  int     lwork;
  int    *iwork;

  iwork = malloc (sizeof(int) * 8 * min(m, n));
  lwork = -1;

  dgesdd_(&jobz,
	  &m, &n, A, &lda,
	  s,
	  U, &ldu,
	  Vt, &ldvt,
	  &cwork, &lwork, iwork, info);

  if (*info != 0)
    {
      fprintf (stderr, "dgesdd_: error during workspace estimation %d\n", *info);
      return;
    }

  lwork = (int) cwork;
  work = malloc (lwork * sizeof(double));
  
  dgesdd_(&jobz,
	  &m, &n, A, &lda,
	  s,
	  U, &ldu,
	  Vt, &ldvt,
	  work, &lwork, iwork, info);

  free (work);
  free (iwork);
#elif defined (HAVE_ACML)
  dgesdd (jobz, m, n, A, lda, s, U, ldu, Vt, ldvt, info);
#else
  *info = LAPACKE_dgesdd (LAPACK_COL_MAJOR, jobz, m, n, A, lda, s,
			  U, ldu, Vt, ldvt);
#endif
}

void
cube_gpu_dgesdd (cube_t *ctx, char jobs, int m, int n, double *A, int lda, double *s, double *U, int ldu, double *Vt, int ldvt)
{
  double *hA, *hU, *hVt, *hs;
  int info;
  int k;

  if (! cube_context_check (ctx))
    return;

#ifdef HAVE_CULA
  if (m > 3000 && n > 3000)
    {
      culaStatus status;

      status = culaDeviceDgesdd (jobs, m, n, A, lda, s, U, ldu, Vt, ldvt);
      if (status != culaNoError)
	{
	  printf ("WARNING: cula error\n");
	  ctx->status = CUBE_ERROR_FAILED;
	}
      return;
    }
#endif

  k = min (m, n);

  hA  = malloc (m * n * sizeof (double));
  hs  = malloc (k * sizeof (double));
  hU  = malloc (ldu * m * sizeof (double));
  hVt = malloc (ldvt * n * sizeof (double));

  cube_memcpy (ctx, hA, A, m * n * sizeof (double), CMK_DEVICE_2_HOST);

  cube_host_dgesdd (jobs, m, n, hA, lda, hs, hU, ldu, hVt, ldvt, &info);

  /* FIXME: check info */

  cube_memcpy (ctx, s, hs, k * sizeof (double), CMK_HOST_2_DEVICE);
  cube_memcpy (ctx, U, hU, ldu * m * sizeof (double), CMK_HOST_2_DEVICE);
  cube_memcpy (ctx, Vt, hVt, ldvt * n * sizeof (double), CMK_HOST_2_DEVICE);

  free (hA);
  free (hs);
  free (hU);
  free (hVt);
}

void
cube_host_dgemm (cube_t *ctx,
		 char transa,
		 char transb,
		 int m,
		 int n,
		 int k,
		 double alpha,
		 double *a,
		 int lda,
		 double *b,
		 int ldb,
		 double beta,
		 double *c,
		 int ldc)
{
  enum CBLAS_TRANSPOSE transA, transB;

  if (! cube_context_check (ctx))
    return;

  transA = transa == 'N' ? CblasNoTrans : CblasTrans;
  transB = transb == 'N' ? CblasNoTrans : CblasTrans;

  cblas_dgemm (CblasColMajor, transA, transB, m, n, k, alpha,
	      a, lda, b, ldb, beta, c, ldc);

#if 0
  dgemm (transa, transb,
	 m, n, k,
	 alpha, a, lda,
	 b, ldb,
	 beta, c, ldc);
#endif
}

void
cube_host_dscal (cube_t *ctx, int n, double a, double *v, int incv)
{
  cblas_dscal (n, a, v, incv);
}

void
cube_host_sscal (cube_t *ctx, int n, float a, float *v, int incv)
{
  cblas_sscal (n, a, v, incv);
}

void
cube_host_isamax (cube_t *ctx, int n, float *v, int incv, int *idx)
{
  int r;

  r = cblas_isamax (n, v, incv);

  if (idx)
    *idx = r;
}

void
cube_host_dvmean(cube_t *ctx, int n, double *v, int incv, double *mean)
{
  long double s;
  int i;
  int vx;

  s = 0.0;
  vx = n * incv;

  for (i = 0; i < vx; i += incv)
    s += v[i];

  *mean = s/n;
}

void
cube_host_dvizm (cube_t *ctx, int n, double *v, int incv)
{
  double mean;
  int i;
  int vx;

  cube_host_dvmean (ctx, n, v, incv, &mean);

  vx = n*incv;
  for (i = 0; i < vx; i += incv)
    v[i] -= mean;
}

void
cube_host_dvvar (cube_t *ctx, int n, double *v, int incv, double *var)
{
  double mean;
  double s;
  int i;
  int vx;

  cube_host_dvmean (ctx, n, v, incv, &mean);

  s = 0;
  vx = n*incv;
  for (i = 0; i < vx; i += incv)
    {
      double x;
      x = v[i] - mean;
      s += pow (x, 2.0);
    }

  *var = s/((double) n - 1.0);
}

void
cube_host_dvstd (cube_t *ctx, int n, double *v, int incv, double *std)
{
  double var;

  cube_host_dvvar (ctx, n, v, incv, &var);
  *std = sqrt (var);
}
