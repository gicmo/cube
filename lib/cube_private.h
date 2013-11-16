// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifndef CUBE_PRIVATE_H
#define CUBE_PRIVATE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_v2.h>

#include "cube_error.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _cube_t {

  int            gpu;

  cube_status_t  status;
  cublasStatus_t e_blas;
  cudaError_t    e_cuda;

  cublasHandle_t h_blas;
};

int cube_context_check (cube_t *ctx);
int cube_blas_check (cube_t *ctx, cublasStatus_t blas_status);
int cube_cuda_check (cube_t *ctx, cudaError_t    cuda_error);

#ifdef __cplusplus
}
#endif

#endif
