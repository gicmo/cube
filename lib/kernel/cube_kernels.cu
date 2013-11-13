#include "cube.h"
#include "cube_blas.h"
#include "cube_matrix.h"

#include "cube_kernels.h"
#include "cube_private.h"

#include <cuda.h>
#include <stdio.h>

__device__ double
d_inv (double x, int inv)
{
  return inv ? 1.0 / x : x;
}

__global__ void
k_diag (int n, double *D, int ldd, int inv, double alpha, double *x, int incx)
{
  extern __shared__ double smem[];
  double *s;

  int     global_x, global_y, lid, gid;

  /* calculate global and local ids */
  global_x = (blockDim.x * blockIdx.x) + threadIdx.y; //n
  global_y = (blockDim.y * blockIdx.y) + threadIdx.x; //m

  gid = (ldd * global_x) + global_y;
  lid = (threadIdx.y * blockDim.x) + threadIdx.x;

  smem[lid] = 0;

  if (blockIdx.x == blockIdx.y && threadIdx.x < warpSize && global_y < n)
    {
      s = &smem[blockDim.x * blockDim.y];
      s[threadIdx.x] = x[(blockIdx.x * blockDim.x + threadIdx.x) * incx];
      smem[threadIdx.x * blockDim.x + threadIdx.x] = d_inv (s[threadIdx.x] * alpha, inv);
    }

  if (global_x < n && global_y < n)
    D[gid] = smem[lid];
}

void
cube_gpu_diag (cube_t *ctx, int n, double *diag, int ldd, int inv, double alpha, double *x, int incx)
{
  size_t  smem;
  dim3    block, grid;

  if (! cube_context_check (ctx))
    return;

  block.x = 32;
  block.y = 32;

  grid.x = ceil (n / (double) block.x);
  grid.y = grid.x;

  smem = (block.x + 1) * block.y * sizeof (double);

  k_diag<<<block, grid, smem>>> (n, diag, ldd, inv, alpha, x, incx);
}

