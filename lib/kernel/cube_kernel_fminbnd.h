
#ifndef CUBE_KERNEL_FMINBND_H
#define CUBE_KERNEL_FMINBND_H

typedef double (*fminbnd_func_t)(double x, void *params);
typedef double (*fminimizer) (double lower, double upper, double tol, fminbnd_func_t func, void *fparams);

fminimizer get_min_f_bounded();

#endif
