// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include <cube.h>
#include <cube_matrix.h>
#include <cube_matlab.h>
#include <cube_blas.h>
#include <cube_math.h>
#include <cube_ica_kernels.h>

#include <stdio.h>
#include <string.h> //memset
#include <matrix.h>

#include <time.h>

#define min(a,b) ((a)>(b)?(b):(a))

void
mat_array_dump (mxArray *matrix, int m_max, int n_max)
{
  double *A;
  int m;
  int n;
  int row, col;

  if (matrix == NULL)
    return;

  A = (double *) mxGetData (matrix);

  m = min (mxGetM (matrix), m_max + 10);
  n = min (mxGetN (matrix), n_max + 10);

  for (row = 10; row < m; row++)
    {
      for (col = 10; col < n; col++)
	{
	  int pos = (col * mxGetM(matrix)) + row;
	  printf ("%lf ", A[pos]); //A[IDX2F(row, col, n)]);
	}

      printf ("\n");
    }
  printf ("\n");
}

int
main (int argc, char **argv)
{
  cube_matfile_t *fd;
  cube_matrix_t  *S, *Sp;
  cube_t *ctx;
  char *filename;
  mxArray *mS, *mSp;
  int n, alen;
  int index;
  int c;

  ctx = cube_context_new (0);
  filename = argv[1];

  fd = cube_matfile_open (ctx, filename);

  c = cube_context_check (ctx);
  printf ("c=%d [%s]\n", c, filename);

  cube_matfile_get_vars (ctx, fd, "S", &mS, "Sp", &mSp, NULL);

  S = cube_matrix_from_array (ctx, mS);
  Sp = cube_matrix_from_array (ctx, mSp);

  mat_array_dump(mS, 10, 10);

  alen = cube_matrix_get_m (Sp);
  n = cube_matrix_get_n (S);

  printf ("%d %d\n", alen, n);

  cube_matrix_sync (ctx, S, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, Sp, CUBE_SYNC_DEVICE);

  for (index = 0; index < alen; index += n)
    {
      printf ("index %d n: %d\n", index, n);
      gpu_collect_prior (ctx, S, Sp, index);
    }

  cube_matrix_sync (ctx, Sp, CUBE_SYNC_HOST);
  cube_matfile_put_var (ctx, fd, "Sres", mSp);

  mat_array_dump (mSp, 10, 10);

  cube_matfile_close (ctx, fd);
  cube_context_destroy (&ctx);

  return 0;
}
