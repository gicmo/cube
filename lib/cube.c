// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include "cube.h"
#include "cube_error.h"
#include "cube_private.h"

#ifdef HAVE_CULA
#include <cula_status.h>
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static cube_t ctx_no_mem = {0, CUBE_ERROR_NO_MEMORY, CUBLAS_STATUS_NOT_INITIALIZED, 0, 0};

cube_t *
cube_context_new (int gpu)
{
  cube_t *ctx;

  ctx = malloc (sizeof(cube_t));

  if (ctx == NULL)
    return &ctx_no_mem;

  memset (ctx, 0, sizeof (cube_t));

  ctx->gpu = gpu;

  if (gpu)
    {
#ifdef HAVE_CULA
      culaStatus s;
#endif

      ctx->e_blas = cublasCreate(&(ctx->h_blas));

      if (! cube_blas_check (ctx, ctx->e_blas))
        return ctx;

#ifdef HAVE_CULA
      s = culaInitialize();

      if (s != culaNoError)
	{
	  printf ("Error: CULA init failed!\n");
	  ctx->status = CUBE_ERROR_BLAS;
	}
#endif
    }
  else
    ctx->e_blas = CUBLAS_STATUS_NOT_INITIALIZED;

  //FIXME: hack for AMD CPUS & OMP
  setenv ("KMP_AFFINITY", "none", 0);

  return ctx;
}

void
cube_context_destroy (cube_t **ctx)
{
  cube_t *ctxp;

  if (ctx == NULL || *ctx == NULL)
    return;

  ctxp = *ctx;
  *ctx = NULL;

  if (ctxp == &ctx_no_mem)
    return;

  if (ctxp->e_blas != CUBLAS_STATUS_NOT_INITIALIZED)
    cublasDestroy(ctxp->h_blas);

  free (ctxp);
}


/* Return 0 in case of error, 1 otherwise */
int
cube_context_check (cube_t *ctx)
{
  if (ctx == NULL)
    return 0;

  if (ctx->status != CUBE_STATUS_OK)
    return 0;

  return 1;
}

int
cube_blas_check (cube_t *ctx, cublasStatus_t blas_status)
{
  if (blas_status == CUBLAS_STATUS_SUCCESS)
    return 1;

  ctx->status = CUBE_ERROR_BLAS;
  ctx->e_blas = blas_status;

  printf ("WARNING: cuda_blas error encountered %d\n", blas_status);

  return 0;
}

int
cube_cuda_check (cube_t *ctx, cudaError_t cuda_error)
{
  if (cuda_error == cudaSuccess)
      return 1;
  
  ctx->status = CUBE_ERROR_CUDA;
  ctx->e_cuda = cuda_error;

  printf ("WARNING: cuda_error encountered:\n\t%s\n",
	  cudaGetErrorString (cuda_error));

  return 0;
}

void *
cube_malloc_device (cube_t *ctx, size_t size)
{
  cudaError_t res; 
  void *dev_ptr;

  if (! cube_context_check (ctx))
    return NULL;

  if (ctx->gpu == 0)
    {
      return malloc (size);
    }

  res = cudaMalloc (&dev_ptr, size);

  if (! cube_cuda_check (ctx, res))
    dev_ptr = NULL;

  return dev_ptr;
}

void
cube_memset_device (cube_t *ctx, void *s, int c, size_t n)
{
  cudaError_t res;

  if (! cube_context_check (ctx))
    return;

  if (ctx->gpu == 0)
    {
      memset (s, c, n);
      return;
    }

  res = cudaMemset (s, c, n);
  cube_cuda_check (ctx, res);
}


void
cube_free_device (cube_t *ctx, void *dev_ptr)
{
  if (! cube_context_check (ctx))
    return;

  if (ctx->gpu == 0)
    {
      free (dev_ptr);
      return;
    }

  cudaFree(dev_ptr);
}

void *
cube_host_register (cube_t *ctx, void *host, size_t len)
{
  cudaError_t res;
  void *dev_ptr;

  if (! cube_context_check (ctx))
    return NULL;

  if (ctx->gpu == 0)
    return host;

  cudaHostRegister (host, len, cudaHostRegisterMapped);
  res = cudaHostGetDevicePointer (&dev_ptr, host, 0);

  if (! cube_cuda_check (ctx, res))
    dev_ptr = NULL;

  return dev_ptr;
}

int 
cube_host_unregister (cube_t *ctx, void *host)
{
  cudaError_t res;

  if (! cube_context_check (ctx))
    return cube_context_check (ctx);

  if (ctx->gpu == 0)
    return 1;

  res = cudaHostUnregister (host);
   
  return cube_cuda_check (ctx, res);
}

void *
cube_memcpy (cube_t *ctx,
	     void   *dest,
	     void   *src,
	     size_t  n,
	     cube_memcpy_kind_t kind)
{
  cudaError_t res;

  if (! cube_context_check (ctx))
    return NULL;


  if (ctx->gpu == 0)
    return memcpy (dest, src, n);


  res = cudaMemcpy (dest, src, n, (enum cudaMemcpyKind) kind);

  if (! cube_cuda_check (ctx, res))
    dest = NULL;

  return dest;
}

int
cube_context_is_gpu (cube_t *ctx)
{
  return ctx->gpu;
}
