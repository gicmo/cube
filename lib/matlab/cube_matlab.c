#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_matlab.h"
#include "cube_ica.h"

#include "cube_ica_kernels.h"

#include <mat.h>
#include <stdarg.h>

typedef struct MATFile _cube_matfile_t;


cube_matfile_t *
cube_matfile_open (cube_t *ctx, const char *filename)
{
  MATFile *fd;

  if (! cube_context_check (ctx))
    return NULL;

  fd = matOpen (filename, "u");

  if (fd == NULL)
    ctx->status = CUBE_ERROR_FAILED; 

  return (cube_matfile_t *) fd;
}

void
cube_matfile_close (cube_t *ctx, cube_matfile_t *mfd)
{
  MATFile *fd = (MATFile *) mfd;

  if (! cube_context_check (ctx) || fd == NULL)
    return;

  matClose (fd);
}

const char **
cube_matfile_get_dir (cube_t *ctx, cube_matfile_t *mfd, int *n)
{
  MATFile *fd = (MATFile *) mfd;
  const char **dir;
    
  if (! cube_context_check (ctx))
    return NULL;

  dir = (const char **) matGetDir (fd, n);

  if (dir == NULL)
    {
      ctx->status = CUBE_ERROR_FAILED;
      *n = 0;
    }

  return dir;
}

mxArray *
cube_matfile_get_var (cube_t         *ctx,
		      cube_matfile_t *mfd,
		      const char     *name)
{
  MATFile *fd = (MATFile *) mfd;
  mxArray *a;

  if (! cube_context_check (ctx) || fd == NULL || name == NULL)
    return NULL;

  a = matGetVariable (fd, name);

  return a;
}


int
cube_matfile_get_vars (cube_t         *ctx,
		       cube_matfile_t *mfd,
		       ...)
{
  va_list ap;
  char    *var;
  int      count;

  count = 0;

  va_start (ap, mfd); 

  while ((var = va_arg (ap, char *)) != NULL)
    {
      mxArray **map;

      map = va_arg (ap, mxArray **);

      if (ap == NULL)
	break;

      *map = cube_matfile_get_var (ctx, mfd, var);
      count++;
    }
 
  va_end(ap);
  return count;
}

void
cube_matfile_put_var (cube_t         *ctx,
		      cube_matfile_t *mfd,
		      const char     *name,
		      mxArray        *a)
{
  MATFile *fd = (MATFile *) mfd;
  int res;

  if (! cube_context_check (ctx) || fd == NULL || name == NULL)
    return;

  res = matPutVariable (fd, name, a);

  if (res != 0)
    ctx->status = CUBE_ERROR_FAILED;
}

cube_matrix_t *
cube_matrix_from_array (cube_t *ctx, mxArray *A)
{
  int m, n, ndims;
  void *data;
  cube_matrix_t *matrix;

  if (! cube_context_check (ctx) || A == NULL)
    return NULL;

  if (! mxIsDouble (A))
    return NULL;

  ndims = mxGetNumberOfDimensions (A);
  
  if (ndims != 2)
    return NULL;

  m = (int) mxGetM (A);
  n = (int) mxGetN (A);
  data = mxGetData (A);

  matrix = cube_matrix_new_from_data (ctx, m, n, data, 'F');

  return matrix;
}


int
cube_matlab_ica_update_A (cube_t  *ctx,
			  mxArray *m_A,
			  mxArray *m_S,
			  mxArray *m_mu,
			  mxArray *m_beta,
			  mxArray *m_sigma,
			  double   m_epsilon)
{
  cube_matrix_t *A, *S, *mu, *beta, *sigma;
  double *epsilon, npats;

  if (! cube_context_check (ctx))
    return -1;

  A     = cube_matrix_from_array (ctx, m_A);
  S     = cube_matrix_from_array (ctx, m_S);
  mu    = cube_matrix_from_array (ctx, m_mu);
  beta  = cube_matrix_from_array (ctx, m_beta);
  sigma = cube_matrix_from_array (ctx, m_sigma);

  npats   = 1.0 / (double) cube_matrix_get_n (S);
  epsilon = cube_host_register (ctx, &m_epsilon, sizeof (double));

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, S, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, mu, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, beta, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, sigma, CUBE_SYNC_DEVICE);

  cube_ica_update_A (ctx, A, S, mu, beta, sigma, &npats, epsilon);

  cube_matrix_sync (ctx, A, CUBE_SYNC_HOST);

  cube_host_unregister (ctx, &m_epsilon);
  cube_matrix_destroy (ctx, A);
  cube_matrix_destroy (ctx, S);
  cube_matrix_destroy (ctx, mu);
  cube_matrix_destroy (ctx, sigma);

  return cube_context_check (ctx);
}

int
cube_matlab_ica_adapt_prior (cube_t  *ctx,
			     mxArray *Sp,
			     double   mu,
			     double   sigma,
			     double   tol,
			     double   a,
			     double   b,
			     mxArray *beta)
{
  int res;
  if (! cube_context_check (ctx))
    return -1;

  res = gpu_adapt_prior_host (ctx,
			      mxGetPr(Sp),
			      mxGetM(Sp),
			      mxGetN(Sp),
			      mu,
			      sigma,
			      tol,
			      a,
			      b,
			      mxGetPr(beta));

  return res;
}

int
cube_matlab_ica_calc_S (cube_t *ctx,
			mxArray *m_A,
			mxArray *m_D,
			mxArray *m_S)
{
  cube_matrix_t *A, *S, *D, *Ai;
  double a, b;
  int m, n;

  if (! cube_context_check (ctx))
    return -1;
  
  A = cube_matrix_from_array (ctx, m_A);
  D = cube_matrix_from_array (ctx, m_D);
  S = cube_matrix_from_array (ctx, m_S);

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, D, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, S, CUBE_SYNC_DEVICE);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);

  Ai = cube_matrix_new_on_device (ctx, m, n);
  cube_matrix_pinv (ctx, A, Ai);

  a = 1.0;
  b = 0.0;
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, Ai, D, &b, S);

  cube_matrix_sync (ctx, S, CUBE_SYNC_HOST);

  cube_matrix_destroy (ctx, A);
  cube_matrix_destroy (ctx, S);
  cube_matrix_destroy (ctx, D);
  cube_matrix_destroy (ctx, Ai);

  return cube_context_check (ctx);
}

int cube_matlab_ica_pinv (cube_t *ctx,
			  mxArray *m_A,
			  mxArray *m_Ai)
{
  cube_matrix_t *A, *Ai;

  if (! cube_context_check (ctx))
    return -1;
  
  A = cube_matrix_from_array (ctx, m_A);
  Ai = cube_matrix_from_array (ctx, m_Ai);

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, Ai, CUBE_SYNC_DEVICE); 

  cube_matrix_pinv (ctx, A, Ai);

  cube_matrix_sync (ctx, Ai, CUBE_SYNC_HOST);

  cube_matrix_destroy (ctx, A);
  cube_matrix_destroy (ctx, Ai);

  return cube_context_check (ctx);
}
