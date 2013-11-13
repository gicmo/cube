#include <cube.h>
#include <cube_matrix.h>
#include <cube_matlab.h>
#include <cube_blas.h>
#include <cube_math.h>
#include <cube_ica_kernels.h>

#include <stdio.h>
#include <string.h>

#include "cube_private.h"


#define min(a,b) ((a)>(b)?(b):(a))

void
matrix_dump (double *matrix, int m, int n, int m_max, int n_max, int m_s = 0, int n_s = 0)
{
  int row, col;
  int im, in;

  if (matrix == NULL)
    return;

  im = min (m, m_max + m_s);
  in = min (n, n_max + n_s);

  for (row = m_s; row < im; row++)
    {
      printf ("[");
      for (col = n_s; col < in; col++)
	{
	  int pos = (col * m) + row;
	  printf (" %0.1lf", matrix[pos]); //A[IDX2F(row, col, n)]);
	}

      printf ("], \n");
    }
  printf ("\n");
}

int
main (int argc, char **argv)
{
  cudaError_t res;
  cube_matrix_t *diag;
  double *x, *X, *iter;
  cube_t *ctx;
  int n;

  ctx = cube_context_new (0);
  
  n = 50;

  diag = cube_matrix_new_on_device (ctx, n, n);
  x = (double *) cube_malloc_device (ctx, n * sizeof (double));

  for (iter = x; iter < (x + n); iter++)
    {
      double d = 0.5;
      cube_memcpy (ctx, iter, &d, sizeof (double), CMK_HOST_2_DEVICE);
    }

  cube_matrix_diag (ctx, diag, 1, 0.25, x, 1);

  res = cudaPeekAtLastError ();
  cube_cuda_check (ctx, res);

  X = (double *) malloc (n * n * sizeof (double));
  cube_memcpy (ctx, X, diag->dev_ptr, n * n * sizeof (double), CMK_DEVICE_2_HOST);

  matrix_dump (X, n, n, 50, 50);

  free (X);
  cube_free_device (ctx, x);
  cube_matrix_destroy (ctx, diag);
  cube_context_destroy (&ctx);

  return 0;
}
