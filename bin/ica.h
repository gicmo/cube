// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifndef ICA_H
#define ICA_H

#include <cube.h>
#include <cube_matrix.h>
#include <cube_blas.h>
#include <cube_math.h>
#include <cube_ica.h>
#include <cube_array.h>
#include <cube_io.h>

#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <list.h>

typedef enum ICAStatusType {
  ICAStatusIterationsUpdate   = 0,
  ICAStatusExtractPatches     = 1,
  ICAStatusExtractPatchesDone = 2,
  ICAStatusAdaptPrior         = 3,
  ICAStatusAdaptPriorDone     = 4

} ICAStatusType;

typedef struct ica_t_ ica_t;
typedef struct ica_monitor_t ica_monitor_t;

typedef int (*ica_monitor_func) (ica_monitor_t *monitor,
				 ICAStatusType stype,
				 int i,
				 int imax,
				 ica_t *ica,
				 cube_t *ctx);

typedef void (*ica_monitor_finish) (ica_monitor_t *monitor);

struct ica_monitor_t {
  struct list_node   mlist;
  ica_monitor_func   mfunc;
  ica_monitor_finish finish;

  int                req_update_freq;
};


typedef struct ica_prior_t {

  cube_matrix_t *mu;
  cube_matrix_t *sigma;
  cube_matrix_t *beta;

  double a;
  double b;

  double tol;

  double adapt_size;

} ica_prior_t;


typedef struct ica_model_t {
  char          *id;
  char          *cfg;
  char          *ds;

  cube_array_t  *A;
  cube_matrix_t *beta;

  cube_array_t  *channels;

} ica_model_t;

struct ica_t_
{
  char          *id;
  char          *cfgid;

  cube_matrix_t *A;

  ica_dataset_t *dataset;
  ica_prior_t   *prior;

  cube_array_t  *epsilon;

  int            prior_adapt_size;
  int            blocksize;

  char          *ctime;
  double         fit_time;
  int            gpu;
  char          *creator;

  cube_h5_t     fd;
  struct list_head  monitors;
};

  /* utils */
double calc_t_elapsed (struct timeval *tic);

#ifdef __cplusplus
}
#endif

#endif
