#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void cube_gpu_diag (cube_t *ctx, int n, double *diag, int ldd, int inv, double alpha, double *x, int incx);

#ifdef __cplusplus
}
#endif

#endif
