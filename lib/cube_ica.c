
#include "cube.h"
#include "cube_blas.h"
#include "cube_math.h"
#include "cube_private.h"
#include "cube_ica.h"
#include "cube_ica_kernels.h"
#include "cube_io.h"

#include <string.h>
#include <stdio.h>

void
cube_ica_prior_collect (cube_t        *ctx,
                        cube_matrix_t *S,
                        cube_matrix_t *coeffs,
                        int            index)
{
   if (! cube_context_check (ctx))
    return;

    gpu_collect_prior (ctx, S, coeffs, index);
}

void
cube_ica_prior_adapt (cube_t        *ctx,
                      double a, double b,
                      cube_matrix_t *coeffs,
                      cube_matrix_t *beta,
                      cube_matrix_t *mu,
                      cube_matrix_t *sigma,
                      double         tol)
{
  double *coeffs_data, *beta_data;
  double mu_s, sigma_s;
  int m, n;

  if (! cube_context_check (ctx))
    return;

  m = cube_matrix_get_m (coeffs);
  n = cube_matrix_get_n (coeffs);
  coeffs_data = coeffs->dev_ptr;
  beta_data = beta->dev_ptr;

  mu_s = mu->data[0];
  sigma_s = mu->data[0];

  gpu_adapt_prior (ctx, coeffs_data, m, n, mu_s, sigma_s, tol, a, b, beta_data);
}

void
cube_ica_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma,
		 cube_matrix_t *Z)
{
  if (! cube_context_check (ctx))
    return;


  cube_gpu_calc_Z (ctx, S, Z, mu, beta, sigma);
}

void
cube_ica_calc_dA (cube_t        *ctx,
		  cube_matrix_t *A,
		  cube_matrix_t *S,
		  cube_matrix_t *mu,
		  cube_matrix_t *beta,
		  cube_matrix_t *sigma,
		  cube_matrix_t *dA)
{
  cube_matrix_t *Z, *X;
  double a, b;
  double npats;
  int m, n;
  
  if (! cube_context_check (ctx))
    return;

  if (!A || !S || !mu || !beta || !sigma)
    return;

  m = cube_matrix_get_m (S);
  n = cube_matrix_get_n (S);

  npats = n;

  Z = cube_matrix_new_on_device (ctx, m, n);
  X = cube_matrix_new_on_device (ctx, m, n);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);

  cube_matrix_copy (ctx, A, dA, CUBE_SYNC_DEVICE);

  /** 1) **/
  cube_ica_calc_Z (ctx, S, mu, beta, sigma, Z);

  /** 2) **/
  a = -1.0;
  b = 0.0;

  /*  X = -1*A*Z  */
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, A, Z, &b, X);

  /* dA = X * S' - npats * A" (with A" = copy of A) */
  a = 1.0;
  b = -1.0 * npats;
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_T, &a, X, S, &b, dA);

  a = 1.0/npats;
  cube_matrix_scale (ctx, dA, &a);

  cube_matrix_destroy (ctx, X);
  cube_matrix_destroy (ctx, Z);
}

void
cube_ica_AdA (cube_t        *ctx,
	      cube_matrix_t *A,
	      cube_matrix_t *dA,
	      double         eps)
{
  double *epsilon;
  int maxidx, *iamax;

  maxidx = 0;
  cube_matrix_iamax (ctx, dA, &maxidx);

  iamax = cube_host_register (ctx, &maxidx, sizeof (maxidx));
  epsilon = cube_host_register (ctx, &eps, sizeof (double));

  gpu_update_A_with_delta_A (ctx, A, dA, epsilon, iamax);

  /* cleanup */
  cube_host_unregister (ctx, &maxidx);
  cube_host_unregister (ctx, &eps);
}


void
cube_ica_update_A (cube_t        *ctx,
		   cube_matrix_t *A,
		   cube_matrix_t *S,
		   cube_matrix_t *mu,
		   cube_matrix_t *beta,
		   cube_matrix_t *sigma,
		   const double  *npats,
		   const double  *epsilon)
{
  cube_matrix_t *Z, *dA, *X;
  double a, b;
  int maxidx, *iamax;
  int m, n;

  if (! cube_context_check (ctx))
    return;

  if (!A || !S || !mu || !beta || !sigma || !npats || !epsilon)
    return;

  m = cube_matrix_get_m (S);
  n = cube_matrix_get_n (S);

  Z = cube_matrix_new_on_device (ctx, m, n);
  X = cube_matrix_new_on_device (ctx, m, n);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);
  dA = cube_matrix_new_on_device (ctx, m, n);
  cube_matrix_copy (ctx, A, dA, CUBE_SYNC_DEVICE);

  /** 1) **/
  cube_ica_calc_Z (ctx, S, mu, beta, sigma, Z);

  /** 2) **/
  a = -1.0;
  b = 0.0;

  /*  X = -1*A*Z  */
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, A, Z, &b, X);

  /* dA = X * S' - npats * A" (with A" = copy of A) */
  a = 1.0;
  b = -1.0 * cube_matrix_get_n (S);
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_T, &a, X, S, &b, dA);

  cube_matrix_scale (ctx, dA, npats);

  /** 3) **/
  maxidx = 0;
  cube_matrix_iamax (ctx, dA, &maxidx);

  iamax = cube_host_register (ctx, &maxidx, sizeof (maxidx));

  gpu_update_A_with_delta_A (ctx, A, dA, epsilon, iamax);

  /* cleanup */
  cube_host_unregister (ctx, &maxidx);
  cube_matrix_destroy (ctx, dA);
  cube_matrix_destroy (ctx, X);
  cube_matrix_destroy (ctx, Z);
}



ica_dataset_t *
cube_ica_read_dataset (const char *filename)
{
  ica_dataset_t *ds;
  cube_h5_t h5;


  h5 = cube_h5_open (filename, 0);
  ds = malloc (sizeof (ica_dataset_t));

  ds->indicies = cube_h5_read_array (h5, "/ds/0/indicies");
  ds->imgdata  = cube_h5_read_array (h5, "/ds/0/imgdata");
  ds->patsperm = cube_h5_read_array (h5, "/ds/0/patsperm");

  ds->blocksize = 100; //FIXME
  ds->patchsize = 7; //FIXME

  cube_h5_close (h5);
  return ds;
}


void
cube_ica_extract_patches (cube_t        *ctx,
			  ica_dataset_t *dataset,
			  int            cluster,
			  cube_matrix_t *patches)
{
  cube_matrix_t *tmp;
  cube_array_t  *idx;
  cube_array_t  *img;
  cube_array_t  *prm;
  double        *D, *P;
  double std;
  size_t bsize;
  int i, m, n, npats;
  int nimg;
  int ppi; //pats-per-image
  int nch; //number of channels
  int ps;  //patchsize
  int *xx;

  idx = dataset->indicies;
  img = dataset->imgdata;
  prm = dataset->patsperm;

  nimg = cube_array_get_dim (img, 3);
  ppi = cube_array_get_dim (idx, 0);
  nch = cube_array_get_dim (img, 0);
  ps = dataset->patchsize;

  m = cube_matrix_get_m (patches);
  n = cube_matrix_get_n (patches);
  tmp = cube_matrix_create (ctx, m, n);

  xx = cube_array_get_data (idx);
  D = cube_matrix_get_data (tmp);

  bsize = ppi*ps*ps*nch;

  for (i = 0; i < nimg; i++)
    {
      size_t boff;
      boff = bsize * i;

      for (n = 0; n < ppi; n++)
	{
	  int xs, ys, x, y;

	  xs = cube_array_get_uint16 (idx, n, 0, cluster, i) - 1;
	  ys = cube_array_get_uint16 (idx, n, 1, cluster, i) - 1;

	  for (x = 0; x < ps; x++)
	    {
	      for (y = 0; y < ps; y++)
		{
		  double *simg;
		  size_t  doff;
		  size_t  soff;

		  simg = cube_array_get_data (img);
		  doff = boff + n*ps*ps*nch + y*ps*nch + x*nch;
		  soff = cube_array_index (img, 0, xs+x, ys+y, i);
		  memcpy (D + doff, simg + soff, nch * sizeof (double));
		}
	    }
	}

      cube_host_dvizm (ctx, bsize, D + boff, 1);
      cube_host_dvstd (ctx, bsize, D + boff, 1, &std);
      cube_host_dscal (ctx, bsize, 1.0/std, D + boff, 1);
    }

  npats = ppi * nimg;
  P = cube_matrix_get_data (patches);
  m = ps*ps*nch;

  for (n = 0; n < npats; n++)
    {
      uint16_t dst;
      size_t poff, doff;

      dst = cube_array_get_int (prm, n, cluster) - 1;
      doff = dst * m;
      poff = n * m;

      memcpy (P + poff, D + doff, sizeof (double) * m);
    }

  for (i = 0; i < m; i++)
    cube_host_dvizm (ctx, npats, P+i, m);

  cube_host_dvstd (ctx, (m*npats), P, 1, &std);
  cube_host_dscal (ctx, npats*m, 1.0/std, P, 1);

  free (tmp->data); //FIXME
  cube_matrix_destroy (ctx, tmp);
}


static double
interpolate_log(double n, double a, double b, double mi, double ma)
{
  double x, y;
  x = (n - a)/(b - a);
  y = (1-x)*log(mi)+x*log(ma);
  return exp(y);
}

double
cube_ica_inter_epsilon (cube_t *ctx, cube_array_t *epsilon, int iter)
{
  int k;
  int m, n;
  int a, b;
  double *iter_points;
  double *e_data;

  if (! cube_context_check (ctx))
    return 0;

  m = cube_array_get_dim (epsilon, 0);
  n = cube_array_get_dim (epsilon, 1);

  /* FIXME: check that epsilon has the same dimensions */

  k = max (m, n);
  iter_points = cube_array_get_data (epsilon);

  b = 1;
  while (iter_points[b] < iter && b < k)
    b++;

  a = b-1;

  e_data = iter_points + k;
  return interpolate_log (iter, iter_points[a], iter_points[b],
			  e_data[a], e_data[b]);

}
