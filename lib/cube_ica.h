
#ifndef CUBE_ICA_H
#define CUBE_ICA_H

#include <cube_matrix.h>
#include <cube_array.h>

typedef struct ica_dataset_t {

  char *id;

  int dim;
  int patchsize;
  int blocksize;
  int nclusters;
  int npats;
  int nchannel;
  int maxiter;

  cube_array_t *channels;

  cube_array_t *imgdata;
  cube_array_t *indicies;
  cube_array_t *patsperm;
  cube_array_t *Ainit;

} ica_dataset_t;


ica_dataset_t * cube_ica_read_dataset (const char *filename);


void cube_ica_extract_patches (cube_t        *ctx,
			       ica_dataset_t *dataset,
			       int            cluster,
			       cube_matrix_t *patches);

void cube_ica_prior_collect (cube_t        *ctx,
                             cube_matrix_t *S,
                             cube_matrix_t *coeffs,
                             int            index);

void cube_ica_prior_adapt (cube_t        *ctx,
                           double a, double b,
                           cube_matrix_t *coeffs,
                           cube_matrix_t *beta,
                           cube_matrix_t *mu,
                           cube_matrix_t *sigma,
                           double         tol);

void cube_ica_calc_Z (cube_t        *ctx,
		      cube_matrix_t *S,
		      cube_matrix_t *mu,
		      cube_matrix_t *beta,
		      cube_matrix_t *sigma,
		      cube_matrix_t *Z);
void
cube_ica_calc_dA (cube_t        *ctx,
		  cube_matrix_t *A,
		  cube_matrix_t *S,
		  cube_matrix_t *mu,
		  cube_matrix_t *beta,
		  cube_matrix_t *sigma,
		  cube_matrix_t *dA);
void
cube_ica_AdA (cube_t        *ctx,
	      cube_matrix_t *A,
	      cube_matrix_t *dA,
	      const double   eps);

void cube_ica_update_A (cube_t        *ctx,
			cube_matrix_t *A,
			cube_matrix_t *S,
			cube_matrix_t *mu,
			cube_matrix_t *beta,
			cube_matrix_t *sigma,
			const double  *npats,
			const double  *epsilon);

double cube_ica_inter_epsilon (cube_t        *ctx,
                               cube_array_t  *epsilon,
                               int            iter);

#endif
