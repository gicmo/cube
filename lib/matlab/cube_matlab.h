#ifndef CUBE_MATLAB_H
#define CUBE_MATLAB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <matrix.h>

#include "cube_matrix.h"

typedef struct _cube_matfile_t cube_matfile_t;

cube_matfile_t * cube_matfile_open  (cube_t *ctx, const char *file);
void             cube_matfile_close (cube_t *ctx, cube_matfile_t *fd);
const char **    cube_matfile_get_dir (cube_t *ctx, cube_matfile_t *fd, int *n);
mxArray *        cube_matfile_get_var (cube_t *ctx, cube_matfile_t *fd, const char *name);
int              cube_matfile_get_vars (cube_t *ctx, cube_matfile_t *mfd, ...);

void             cube_matfile_put_var (cube_t         *ctx,
				       cube_matfile_t *mfd,
				       const char     *name,
				       mxArray        *a);

cube_matrix_t *  cube_matrix_from_array (cube_t *ctx, mxArray *array);

int cube_matlab_ica_adapt_prior (cube_t  *ctx,
				 mxArray *Sp,
				 double   mu,
				 double   sigma,
				 double   tol,
				 double   a,
				 double   b,
				 mxArray *beta);

int  cube_matlab_ica_update_A (cube_t  *ctx,
			       mxArray *m_A,
			       mxArray *m_S,
			       mxArray *m_mu,
			       mxArray *m_beta,
			       mxArray *m_sigma,
			       double   m_epsilon);

int cube_matlab_ica_calc_S (cube_t *ctx,
			    mxArray *m_A,
			    mxArray *m_D,
			    mxArray *m_S);

int cube_matlab_ica_pinv (cube_t *ctx,
			  mxArray *m_A,
			  mxArray *m_ai);

#ifdef __cplusplus
}
#endif

#endif
