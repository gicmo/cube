// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifndef CUDA_ICA_KERNEL_H
#define CUDA_ICA_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

int
cube_gpu_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *Z,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma);

void
gpu_update_A_with_delta_A (cube_t        *ctx,
			   cube_matrix_t *A,
			   cube_matrix_t *dA,
			   const double  *epsilon,
			   const int     *iamax);
double
gpu_sumpower (cube_t *ctx, const double *in, int n, double p);
double
gpu_aprior (cube_t *ctx, const double *in, int n, double tol, double a, double b);

int
gpu_adapt_prior_host (cube_t *ctx, const double *in, int m, int n, double mu, double sigma, double tol, double a, double b, double *beta);
int
gpu_adapt_prior (cube_t *ctx, const double *in, int m, int n, double mu, double sigma, double tol, double a, double b, double *beta);

void gpu_collect_prior (cube_t *ctx,
			cube_matrix_t *S,
			cube_matrix_t *priorS,
			int            index);

#ifdef __cplusplus
}
#endif

#endif
