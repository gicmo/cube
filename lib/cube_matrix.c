
#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_matrix.h"
#include "cube_math.h"
#include "cube_kernels.h"

#include <stdio.h>
#include <string.h>



//#define IDX2F(i, j, ld) ((((j)-1)∗(ld))+((i)-1))
#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

static cube_matrix_t *
cube_matrix_new (cube_t *ctx)
{
  cube_matrix_t *ma;

  if (! cube_context_check (ctx))
    return NULL;

  ma = malloc (sizeof (cube_matrix_t));

  if (ma)
    memset (ma, 0, sizeof (cube_matrix_t));

  return ma;
}

cube_matrix_t *
cube_matrix_create (cube_t *ctx,
		    int     m,
		    int     n)
{
  cube_matrix_t *ma;

  ma = cube_matrix_new (ctx);

  if (ma)
    {
      ma->data = malloc (sizeof (double) * m * n);
      ma->m = m;
      ma->n = n;
    }

  return ma;
}

cube_matrix_t *
cube_matrix_new_from_data (cube_t       *ctx,
			   int           m,
			   int           n,
			   double       *data,
			   char          order)
{
  cube_matrix_t *ma;

  ma = cube_matrix_new (ctx);

  if (ma == NULL)
    return NULL;

  ma->data = data;
  ma->m = m;
  ma->n = n;
  ma->order = order;
  ma->dev_ptr = NULL;

  return ma;
}

cube_matrix_t *
cube_matrix_new_fill (cube_t *ctx,
		      int m,
		      int n,
		      double a)
{
  cube_matrix_t *ma;

  ma = cube_matrix_new (ctx);

  if (ma)
    {
      double *data;
      int i;

      ma->data = data = malloc (sizeof (double) * m * n);
      ma->m = m;
      ma->n = n;


      for (i = 0; i < (m*n); i++)
        data[i] = a;
    }

  return ma;
}

cube_matrix_t *
cube_matrix_new_ones (cube_t *ctx,
                      int     m,
                      int     n)
{
  cube_matrix_t *ma;

  ma = cube_matrix_new_fill (ctx, m, n, 1.0);

  return ma;

}

void
cube_matrix_destroy (cube_t        *ctx,
		     cube_matrix_t *matrix)
{
  if (! cube_context_check (ctx) || matrix == NULL)
    return;

  if (matrix->dev_ptr != NULL)
    cube_free_device (ctx, matrix->dev_ptr);

  free (matrix);
}


void
cube_matrix_sync (cube_t          *ctx,
		  cube_matrix_t   *matrix,
		  cube_sync_dir_t  direction)
{
  size_t size;

  if (! cube_context_check (ctx) || matrix == NULL)
    return;

  if (ctx->gpu == 0)
    return;

  size = matrix->m * matrix->n * sizeof (double);

  if (direction == CUBE_SYNC_DEVICE)
    {
      if (matrix->dev_ptr == NULL)
	matrix->dev_ptr = cube_malloc_device (ctx, size);

      cube_memcpy (ctx,
		   matrix->dev_ptr,
		   matrix->data,
		   size,
		   CMK_HOST_2_DEVICE);

    }
  else
    {
      cube_memcpy (ctx,
		   matrix->data,
		   matrix->dev_ptr,
		   size,
		   CMK_DEVICE_2_HOST);
    }
}

int
cube_matrix_get_m (cube_matrix_t *matrix)
{
  if (matrix == NULL)
    return 0;

  return matrix->m;
}


int
cube_matrix_get_n (cube_matrix_t *matrix)
{
  if (matrix == NULL)
    return 0;

  return matrix->n;
}


double *
cube_matrix_get_data (cube_matrix_t *matrix)
{
  if (matrix == NULL)
    return NULL;

  return (matrix->data);
}


void
cube_matrix_iamax (cube_t              *ctx,
		   const cube_matrix_t *A,
		   int                 *result)
{
  int size;

  if (! cube_context_check (ctx))
    return;

  size = A->m * A->n;

  cube_blas_d_iamax (ctx,
		     size,
		     A->dev_ptr, 1,
		     result);
}

void
cube_matrix_amax (cube_t              *ctx,
		  const cube_matrix_t *A,
		  void                *result)
{
  if (! cube_context_check (ctx))
    return;
  //FIXME: implement me
}

void
cube_matrix_gemm (cube_t *ctx,
		  cube_blas_op_t transa, cube_blas_op_t transb,
		  const double *alpha,
		  const cube_matrix_t *A,
		  const cube_matrix_t *B,
		  const double *beta,
		  cube_matrix_t *C)
{
  void *devA, *devB, *devC;
  int m, n, k;
  int x, y, z;

  if (! cube_context_check (ctx))
    return;

  m = C->m;
  n = C->n;

  x = A->m;
  k = A->n;

  y = B->m;
  z = B->n;

  devA = A->dev_ptr;
  devB = B->dev_ptr;
  devC = C->dev_ptr;

  cube_blas_d_gemm (ctx,
		    transa, transb,
		    m, n, k,
		    alpha,
		    devA, x,
		    devB, y,
		    beta,
		    devC, m);

}

cube_matrix_t *
cube_matrix_new_on_device (cube_t       *ctx,
			   int           m,
			   int           n)
{
  cube_matrix_t *ma;
  size_t         size;

  ma = cube_matrix_new (ctx);

  if (ma == NULL)
    return NULL;

  size = m * n * sizeof (double);

  ma->m = m;
  ma->n = n;
  ma->dev_ptr = cube_malloc_device (ctx, size);
  //ma->data = malloc (size);

  if (ma->dev_ptr == NULL)
    {
      free (ma);
      ma = NULL;
    }

  return ma;
}

void
cube_matrix_copy         (cube_t          *ctx,
			  cube_matrix_t   *x,
			  cube_matrix_t   *y,
			  cube_sync_dir_t  where)
{
  int n;

  if (! cube_context_check (ctx))
    return;

   n = x->m * x->n;

   cube_blas_d_copy (ctx,
		     n,
		     x->dev_ptr, 1,
		     y->dev_ptr, 1);

}

void
cube_matrix_scale (cube_t          *ctx,
		   cube_matrix_t   *x,
		   const double    *alpha)
{
  int n;

  if (! cube_context_check (ctx))
    return;

  n = x->m * x->n;

  cube_blas_d_scal (ctx, n,
		    alpha,
		    x->dev_ptr, 1);
}

void
cube_matrix_host_pinv (cube_t        *ctx,
		       cube_matrix_t *mA,
		       cube_matrix_t *mAi)
{
  double *A, *Ai, *Ac, *s, *U, *Vt, *Sw;
  double *X;
  char jobu, jobvt;
  int lda, ldu, ldvt, info;
  int m, n, k;
  int i;

  X = Sw = NULL;

  m = cube_matrix_get_m (mA);
  n = cube_matrix_get_n (mA);

  jobu  = 'A';
  jobvt = 'A';

  lda  = m;
  ldu  = m;
  ldvt = n;

  k = min (m, n);

  //cube_matrix_sync (ctx, mA, CUBE_SYNC_HOST);

  A  = malloc (m * n * sizeof (double));
  s  = malloc (k * sizeof (double));
  U  = malloc (ldu * m * sizeof (double));
  Vt = malloc (ldvt * n * sizeof (double));

  Ac = cube_matrix_get_data (mA);
  memcpy (A, Ac, m * n * sizeof (double));

  cube_host_dgesvd (jobu, jobvt, m, n, A, lda, s, U, ldu, Vt, ldvt, &info);

  if (info != 0)
    {
      fprintf (stderr, "Error in cube_matrix_host_pinv (info != 0)!");
      goto out;
    }

  Sw = malloc (k * k * sizeof (double));
  memset (Sw, 0, (k * k * sizeof (double)));

  for (i = 0; i < k; i++)
    {
      Sw[i*k+i] = 1.0/s[i];
    }

  X = malloc (k * n * sizeof (double));
  cube_host_dgemm (ctx, 'T', 'N', n, n, n, 1.0, Vt, ldvt, Sw, k, 0.0, X, k);

  //Ai = malloc (m * n * sizeof (double));
  Ai = cube_matrix_get_data (mAi);
  cube_host_dgemm (ctx, 'N', 'T',  m, n, n, 1.0, X, k, U, ldu, 0.0, Ai, m);

  //cube_memcpy (ctx, mAi->dev_ptr, Ai, m * n * sizeof (double), CMK_HOST_2_DEVICE);

 out:
  free (A);
  free (X);
  free (Sw);
  free (s);
  free (U);
  free (Vt);
}

void
cube_matrix_pinv (cube_t        *ctx,
		  cube_matrix_t *mA,
		  cube_matrix_t *mAi)
{
  double *A, *Ai, *s, *U, *Vt, *Sw, *X;
  double alpha, beta;
  int m, n, k;
  int lda, ldu, ldvt;

 
  if (! cube_context_check (ctx))
    return;

  m = cube_matrix_get_m (mA);
  n = cube_matrix_get_n (mA);

  lda  = m;
  ldu  = m;
  ldvt = n;

  k = min (m, n);
  
  A = cube_malloc_device (ctx, lda * n * sizeof (double));
  s = cube_malloc_device (ctx, k * sizeof (double));
  U = cube_malloc_device (ctx, ldu * m * sizeof (double));
  Vt = cube_malloc_device (ctx, ldvt * n * sizeof (double));

  cube_memcpy (ctx, A, mA->dev_ptr, m * n * sizeof (double), CMK_DEVICE_2_DEVICE);

  cube_gpu_dgesdd (ctx, 'S', m, n, A, lda, s, U, ldu, Vt, ldvt);

/* create diag matrix out of S */
  Sw = cube_malloc_device (ctx, k * k * sizeof (double));
  cube_gpu_diag (ctx, k, Sw, k, 1, 1, s, 1);

  X = cube_malloc_device (ctx, k * n * sizeof (double));
  Ai = mAi->dev_ptr;

  alpha = 1.0;
  beta  = 0.0;
  cube_blas_d_gemm (ctx, CUBE_BLAS_OP_T, CUBE_BLAS_OP_N, n, n, n, &alpha, Vt, ldvt, Sw,   k, &beta,  X, k);
  cube_blas_d_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_T, m, n, m, &alpha,  X,    k,  U, ldu, &beta, Ai, m);


  cube_free_device (ctx, A);
  cube_free_device (ctx, s);
  cube_free_device (ctx, U);
  cube_free_device (ctx, Vt);
  cube_free_device (ctx, Sw);
  cube_free_device (ctx, X);
}

void
cube_matrix_diag (cube_t *ctx, cube_matrix_t *diag, int inv, double alpha, double *x, int incx)
{
  double *D;
  int     n;

  if (! cube_context_check (ctx))
    return;

  n = cube_matrix_get_m (diag);
  D = (double *) diag->dev_ptr;

  cube_gpu_diag (ctx, n, D, n, inv, alpha, x, incx);
}


void
cube_matrix_dump (cube_matrix_t *matrix, int m_max, int n_max)
{
  double *A;
  int m;
  int n;
  int row, col;

  if (matrix == NULL)
    {
      printf ("Empty matrix!\n");
      return;
    }

  A = matrix->data;

  m = min (matrix->m, m_max);
  n = min (matrix->n, n_max);

  for (row = 0; row < m; row++)
    {
      for (col = 0; col < n; col++)
	{
	  int pos = (col * matrix->m) + row;
	  printf ("%lf ", A[pos]); //A[IDX2F(row, col, n)]);
	}

      printf ("\n");
    }
  printf ("\n");
}
