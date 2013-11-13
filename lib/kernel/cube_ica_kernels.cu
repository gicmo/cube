
#include "cube.h"
#include "cube_blas.h"
#include "cube_matrix.h"

#include "cube_ica_kernels.h"
#include "cube_private.h"

#include <cuda.h>
#include <stdio.h>

__device__ double k_dsign (double v)
{
  if (v > 0)
    return 1;
  else if (v < 0)
    return -1;
  else
    return 0;
}

typedef struct _exp_param
{
  double *y;
  int    n;

  double mu;
  double sigma;
  double a;
  double b;

  double *sum;
  int     bs;

} exp_param;

__device__ double
sumpower_dev (const double *x,
	      int           n,
	      double        p,
	      double        mu,
	      double        sigma,
	      double       *sum,
	      int           bs)
{
  int i, off;

  // zero out shared mem so we can sum on int directly
  sum[threadIdx.x] = 0;

  __syncthreads (); // necessary?

  for (i = 0; i < bs; i++)
    {
      int block = threadIdx.x*bs;
      int col = block+i;
      double u;

      if (!(col < n))
	break;

      u = pow (fabs ((x[col] - mu) / sigma), p);
      sum[threadIdx.x] += u;
    }

  // b-tree result calculation
  for (off = 1; off < blockDim.x; off = off << 1)
    {
      __syncthreads ();

      if (threadIdx.x < (blockDim.x/(off*2)))
        {
          int off_x = (threadIdx.x * off * 2);
	  int off_y = off_x + off;

          sum[off_x] += sum[off_y];
        }
    }

  // write memory back to device memory
  __syncthreads (); // not sure that is needed either
  // sum[0] will hold the result!
  return sum[0];
}

__device__ double
exp_pwr_l_beta (double beta, const exp_param params)
{
  double *y;
  double mu, sigma, a, b;
  double c, logw, uas, l;
  double p;
  int n;

  y     = params.y;
  n     = params.n;
  mu    = params.mu;
  sigma = params.sigma;
  a     = params.a;
  b     = params.b;


  p = 2.0/(1+beta);

  c = pow ((tgamma(3.0/p)/tgamma(1/p)), (p/2.0));
  logw = 0.5 * lgamma (3.0/p) - 1.5 * lgamma (1/p) - log (1+beta);

  uas = sumpower_dev (y, n, p, mu, sigma, params.sum, params.bs);

  l = (-1*lgamma(a)) - (a*log(b)) + ((a-1.0)*log(1.0+beta)) +
    n*logw - n*log(sigma) - ((1.0+beta)/b) - (c * uas);

  return l;
}

__device__ double
exp_pwrlbeta_fmin (double x, const exp_param fparams)
{
  double l, beta;

  beta = exp (x) - 1;

  l = -1 *  exp_pwr_l_beta (beta, fparams);

  return l;
}

/* *************************************************************************** */
// Adapted from SciPy's fminbound function
// in SciPy's optimize module (cf. /scipy/optimize/optimize.py)
// Its license notice:
// ******NOTICE***************
// optimize.py module by Travis E. Oliphant
//
// You may copy and use this module as you see fit with no
// guarantee implied provided you keep this notice in all copies.
// *****END NOTICE************

#define mean_2(a,b) 0.5*(a+b)

/* No input checking, lower > upper required  */
__device__ double
f_min_bound (double lower, double upper, double tol, exp_param fparams)
{
 double tol1, tol2;
  int maxfun,  maxiter;
  double a, b, d, e;
  double fluc, ffluc, nfc, fnfc, x, xf, fx, xm;
  int count, iter;
  const double seps = 1.4901e-08; // nfcorth calculating?
  const double golden = 0.5 * (3.0 - sqrt (5.0));

  count = 0.0;
  iter = 0;

  maxiter = maxfun = 500;

  a = lower;
  b = upper;
  fluc = nfc = x = xf = a + golden * (b - a);
  d = e = 0.0;

  fx = exp_pwrlbeta_fmin(x, fparams);

  ffluc = fnfc = fx;

  xm = mean_2(a,b); //0.5 * (a + b);
  tol1 = seps * fabs(xf) + tol/3.0;
  tol2 = 2.0 * tol1;

  count++;
  while (fabs (xf-xm) > (tol2 - 0.5*(b-a)))
    {
      double r, q, p, si, fu;
      bool do_gs = true;

      if (fabs(e) > tol1)
	{
	  r = (xf - nfc)  * (fx - ffluc);
	  q = (xf - fluc) * (fx - fnfc);
	  p = (xf - fluc) * q - (xf - nfc) * r;
	  q = 2.0 * (q - r);

	  if (q > 0.0)
	    p = -p;

	  q = fabs(q);
	  r = e; e = d;

	  if ((fabs(p) < fabs(0.5 * q * r)) &&
	      (p > q * (a - xf)) &&
	      (p < q * (b - xf)))
	    {
	      d = p/q;
	      x = xf + d;

	      if (((x - a) < tol2) || ((b - x) < tol2))
		{
		  si = k_dsign (xm - xf) + ((xm - xf) == 0);
		  d = tol1 * si;
		}

	      do_gs = false;
	    }
	}

      if (do_gs)
	{
	  e = xf >= xm ? a - xf : b - xf;
	  d = golden * e;
	}

      si = k_dsign(d) + (d == 0);
      x = xf + si * fmax (fabs(d), tol1);

      fu = exp_pwrlbeta_fmin(x, fparams);
      count++;
      iter++;

      if (fu <= fx)
	{
	  (x >= xf ? a : b) = xf;

	  fluc = nfc; ffluc = fnfc;
	  nfc = xf; fnfc = fx;
	  xf = x; fx = fu;
	}
      else // fu > fx
	{
	  (x < xf ? a : b) = x;

	  if ((fu <= fnfc) || (nfc == xf))
	    {
	      fluc = nfc; ffluc = fnfc;
	      nfc = x; fnfc = fu;
	    }
	  else if ((fu <= ffluc) || (fluc == xf) || (fluc == nfc))
	    {
	     fluc = x; ffluc = fu;
	    }

	}

      xm = mean_2(a,b);
      tol1 = seps * fabs(xf) + tol/3.0;
      tol2 = 2.0 * tol1;

      if (count > maxfun || iter > maxiter)
	break;
    }

  return xf;
}

/* x, y are memory references (C layout, row-major),
   m, n are matrix dimensions,
   (col-major means m == y and n == x) */
__global__ void
adapt_prior_kernel (const double *in,
		    const int     m,
		    const int     bs,
		    const double  ax,
		    const double  bx,
		    const double  a,
		    const double  b,
		    double       *out)
{
  exp_param fparams;
  extern __shared__ double data[];
  int i;
  double res;

  // read data from global memory
  for (i = 0; i < bs; i++)
    {
      int x = threadIdx.x * bs + i;
      int y = blockIdx.y * m;

      if (x < m)
	data[x] = in[y + x];
    }

  // now the calculation

  fparams.y = &data[0];
  fparams.n = m;
  fparams.bs = bs;

  fparams.sum = &data[m];
  fparams.a = a;
  fparams.b = b;
  fparams.mu = 0.0;
  fparams.sigma = 1.0;

  res = f_min_bound (ax, bx, 0.1, fparams);

  if (threadIdx.x == 0)
    out[blockIdx.y] = exp(res) - 1;
}

int
gpu_adapt_prior_host (cube_t *ctx, const double *in, int m, int n, double mu, double sigma, double tol, double a, double b, double *beta)
{
  cudaError_t r;
  double *devp, *out;
  double betamin, betamax;
  double xmin, xmax;
  dim3 grid, block;
  size_t smem;
  int bs;

  if (! cube_context_check (ctx))
    return -1;

  betamin = -0.9;
  betamax = 20.0;

  xmin = log (1 + betamin);
  xmax = log (1 + betamax);

  out = (double *) cube_malloc_device (ctx, sizeof (double) * n);
  devp = (double *) cube_malloc_device (ctx, sizeof (double) * n * m);
  cube_memcpy (ctx, devp, (void *) in, sizeof (double) * n * m, CMK_HOST_2_DEVICE);

  grid.y = n;
  block.x = 512;

  smem = (block.x + m) * sizeof (double);
  bs = ceil ((double) m / block.x);

  adapt_prior_kernel<<<grid, block, smem>>>(devp, m, bs, xmin, xmax, a, b, out);

  r = cudaPeekAtLastError ();
  cube_cuda_check (ctx, r);

  cube_memcpy (ctx, beta, out, sizeof (double) * n, CMK_DEVICE_2_HOST);
  cube_free_device (ctx, devp);
  cube_free_device (ctx, out);

  return cube_cuda_check (ctx, r);
}

int
gpu_adapt_prior (cube_t *ctx, const double *in, int m, int n, double mu, double sigma, double tol, double a, double b, double *beta)
{
  cudaError_t r;
  double betamin, betamax;
  double xmin, xmax;
  dim3 grid, block;
  size_t smem;
  int bs;

  if (! cube_context_check (ctx))
    return -1;

  betamin = -0.9;
  betamax = 20.0;

  xmin = log (1 + betamin);
  xmax = log (1 + betamax);

  grid.y = n;
  block.x = 512;

  smem = (block.x + m) * sizeof (double);
  bs = ceil ((double) m / block.x);

  adapt_prior_kernel<<<grid, block, smem>>>(in, m, bs, xmin, xmax, a, b, beta);

  r = cudaPeekAtLastError ();
  
  return cube_cuda_check (ctx, r);
}



__global__ void
sumpower_kernel (const double *in, int n, int bs, double p, double *out)
{
  int tid;
  extern __shared__ double data[];
  double *sum;
  int i;

  tid = threadIdx.x;

  // read data from global memory

  for (i = 0; i < bs; i++)
    {
      int col = tid*bs+i;

      if (col < n)
	data[col] = in[col];
    }

  sum = &data[n];

  sumpower_dev (data, n, p, 0, 1, sum, bs);

  if (tid == 0)
    out[blockIdx.x] = sum[0];

}

double
gpu_sumpower (cube_t *ctx, const double *in, int n, double p)
{
  cudaError_t r;
  double *devp, res, *out;
  dim3 grid, block;
  size_t smem;
  int bs;

  if (! cube_context_check (ctx))
    return -1;

  out = (double *) cube_host_register (ctx, &res, sizeof (res));
  devp = (double *) cube_malloc_device (ctx, sizeof (double) * n);
  cube_memcpy (ctx, devp, (void *) in, sizeof (double) * n, CMK_HOST_2_DEVICE);

  block.x = 512;
  smem = (block.x + n) * sizeof (double);
  bs = ceil ((double) n / block.x);

  sumpower_kernel<<<grid, block, smem>>>(devp, n, bs, p, out);
  
  r = cudaPeekAtLastError ();
  cube_cuda_check (ctx, r);

  cube_host_unregister (ctx, &res);

  return res;
}





__global__ void
kernel_calc_z (const double *S_g,
	       int           m,
	       int           n,
	       const double *mu_g,
	       const double *beta_g,
	       const double *sigma_g,
	       double       *Z)
{
  extern __shared__ double smem[];
  double mu, beta, sigma;
  double *mu_s, *beta_s, *sigma_s, *S_s, *Z_s;
  double s, q, c, z;
  int    global_x, global_y, lid, gid;

    /* calculate global and local ids */
  global_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  global_y = (blockDim.y * blockIdx.y) + threadIdx.y;

  if (global_x > n)
    return;

  gid = (n * global_y) + global_x;
  lid = (threadIdx.y * blockDim.x) + threadIdx.x;

  mu_s = &smem[0];
  beta_s = &smem[blockDim.x];
  sigma_s = &smem[2*blockDim.x];
  S_s = &smem[3 * blockDim.x];
  Z_s = &smem[(3 + blockDim.y) *blockDim.x];

  mu_s[threadIdx.y] = mu_g[global_y];
  beta_s[threadIdx.y] = beta_g[global_y];
  sigma_s[threadIdx.y] = sigma_g[global_y];

  S_s[lid] = S_g[gid];

  __syncthreads();

  if (global_y > m)
    return;

  mu = mu_s[threadIdx.y];
  beta = beta_s[threadIdx.y];
  sigma = sigma_s[threadIdx.y];

  s = S_s[lid];

  /* do the computation */
  s -= mu;
  q = (2.0/(1.0+beta));
  c = pow ((tgamma(3.0/q)/tgamma(1.0/q)), (q/2.0));
  z = -1 * (q*c/pow (sigma,q)) * pow (abs (s), q-1.0) * k_dsign (s);

  Z_s[lid] = z;

  __syncthreads();

  Z[gid] = Z_s[lid];
}


int
cube_gpu_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *Z,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma)
{
  cudaError_t res;
  double *devS, *devZ, *devmu, *devbeta, *devsigma;
  dim3 grid, block;
  int m, n;
  size_t smem;

  if (! cube_context_check (ctx))
    return cube_context_check (ctx);

  m = cube_matrix_get_m (Z);
  n = cube_matrix_get_n (Z);

  block.x = 16;
  block.y = 16;

  grid.x = ceil (n / (double) block.x);
  grid.y = ceil (m / (double) block.y);

  smem = block.y * sizeof (double) * (3 + 2*block.x);

  devS = (double *) S->dev_ptr;
  devZ = (double *) Z->dev_ptr;
  devmu = (double *) mu->dev_ptr;
  devbeta = (double *) beta->dev_ptr;
  devsigma = (double *) sigma->dev_ptr;

  kernel_calc_z<<<grid, block, smem>>>(devS, m, n, devmu, devbeta, devsigma, devZ);

  res = cudaPeekAtLastError ();
  return cube_cuda_check (ctx, res);
}

__global__ void
update_AdA_kernel (double       *A,
		   const double *dA,
		   int           m,
		   int           n,
		   const double *epsilon,
		   const int    *iamax)
{
  double max;
  const double eps = *epsilon;
  extern __shared__ double smem[];
  double *dA_data;
  double *A_data;
  int     global_x, global_y, lid, gid;

  /* calculate global and local ids */
  global_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  global_y = (blockDim.y * blockIdx.y) + threadIdx.y;

  /* see if we are inside the boundaries */
  if (global_x > n || global_y > m)
    return;

  gid = (n * global_y)  + global_x;
  lid = (threadIdx.y * blockDim.x) + threadIdx.x;

  /* set up shared memory addresses  */
  A_data = &smem[0];
  dA_data = &smem[blockDim.x * blockDim.y];

  dA_data[lid] = dA[gid];
  A_data[lid] = A[gid];

  __syncthreads();

  /* do the computation */
  max = fabs(dA[*iamax - 1]); /* global read, but LDU hopefully (FIXME, not sure) */
  A_data[lid] += (eps / max) * dA_data[lid];


  /* write result back */
  __syncthreads();
  A[gid] = A_data[lid];

}

void
gpu_update_A_with_delta_A (cube_t        *ctx,
			   cube_matrix_t *A,
			   cube_matrix_t *dA,
			   const double  *epsilon,
			   const int     *iamax)
{
  cudaError_t res;
  double *devA, *devdA;
  dim3 grid, block;
  size_t smem;
  int  m, n;

  if (! cube_context_check (ctx))
    return;

  m = A->m;
  n = A->n;

  block.x = 16;
  block.y = 16;

  grid.x = ceil (n / (double) block.x);
  grid.y = ceil (m / (double) block.y);

  block.z = grid.z = 1;

  smem = 2 * block.x * block.y * sizeof (double);

  devA = (double *) A->dev_ptr;
  devdA = (double *) dA->dev_ptr;

  //printf ("%u, %u, %zu\n", grid.x, grid.y, smem);

  update_AdA_kernel<<<grid, block, smem>>>(devA, devdA, m, n, epsilon, iamax);
  res = cudaPeekAtLastError ();

  cube_cuda_check (ctx, res);
}

__global__
void k_collect_prior (double *S, int m, int n, int lds, double *Sp, int ldsp, int index)
{
  extern __shared__ double smem[];
  int     global_x, global_y, lid, gid;

  /* calculate global and local ids */
  global_x = (blockDim.x * blockIdx.x) + threadIdx.y; //n
  global_y = (blockDim.y * blockIdx.y) + threadIdx.x; //m

  gid = (lds * global_x) + global_y;
  lid = (threadIdx.y * blockDim.x) + threadIdx.x;

  /* read memory form S and write to smem */
  /* see if we are inside the boundaries */
  if (global_x < n && global_y < m)
    smem[lid] = S[gid];

  __syncthreads();

  /* recaculate gid, lid and write to glboal memory */
  global_x = (blockDim.x * blockIdx.y) + threadIdx.y;
  global_y = (blockDim.y * blockIdx.x) + threadIdx.x;

  gid = (ldsp * global_x) + index + global_y;
  lid = (threadIdx.x * blockDim.x) + threadIdx.y;

  if (global_x < m && global_y < (ldsp - index))
    Sp[gid] = smem[lid];
}


void
gpu_collect_prior (cube_t *ctx,
		   cube_matrix_t *S,
		   cube_matrix_t *priorS,
		   int            index)
{
  cudaError_t res;
  double *devS, *devSp;
  dim3 grid, block;
  int m, n, lds, ldsp;
  size_t smem;

  if (! cube_context_check (ctx))
    return;

  m = cube_matrix_get_m (S);
  n = cube_matrix_get_n (S);
  lds = m;

  ldsp = cube_matrix_get_m (priorS);
  
  block.x = 16;
  block.y = 16;

  grid.x = ceil (n / (double) block.x);
  grid.y = ceil (m / (double) block.y);

  smem = block.x * block.y * sizeof (double);

  devS = (double *) S->dev_ptr;
  devSp = (double *) priorS->dev_ptr;

  k_collect_prior<<<grid, block, smem>>>(devS, m, n, lds, devSp, ldsp, index);

  res = cudaPeekAtLastError ();
  cube_cuda_check (ctx, res);
}
