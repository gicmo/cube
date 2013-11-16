// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifndef CUBE_IO_H
#define CUBE_IO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cube_matrix.h>
#include <cube_array.h>
#include <cube_ica.h>

#include <sys/types.h>

typedef int cube_h5_t;

cube_h5_t       cube_h5_open (const char *filename, int create);
void            cube_h5_close (cube_h5_t h5);
cube_array_t *  cube_h5_read_array (cube_h5_t h5, const char *path);
cube_matrix_t * cube_h5_read_matrix (cube_t *ctx, cube_h5_t h5, const char *path);
int             cube_h5_write_matrix (cube_h5_t h5, cube_matrix_t *matrix, const char *path);
ica_dataset_t * cube_h5_read_dataset (cube_h5_t h5, const char *base);
int             cube_h5_ds_find (cube_h5_t h5, const char *name);
int             cube_h5_link_delete (cube_h5_t h5, const char *name);
int             cube_h5_group_nchildren (cube_h5_t fd, const char *group);
char *          cube_h5_loc_get_name_idx (cube_h5_t fd, const char *location, int index);
int             cube_h5_ds_write_array (cube_h5_t h5, const char *path, cube_array_t *array);
char *          cube_h5_ds_read_string (cube_h5_t h5, const char *location);
int             cube_h5_attr_read_int (cube_h5_t fd, const char *path, const char *attribute);
char *          cube_h5_attr_read_string (cube_h5_t fd, const char *path, const char *attribute);
cube_array_t *  cube_h5_attr_read_array (cube_h5_t fd, const char *path, const char *attribute);
int             cube_h5_attr_write_string (cube_h5_t fd, const char *path, const char *attribute, const char *value);
int             cube_h5_attr_write_double (cube_h5_t fd, const char *path, const char *attribute, double value);
int             cube_h5_attr_write_int16 (cube_h5_t fd, const char *path, const char *attribute, int16_t value);
int             cube_h5_attr_write_array (cube_h5_t fd, const char *path, const char *attribute, cube_array_t *array);
int             cube_h5_group_create (cube_h5_t fd, const char *path);
int             cube_h5_flush (cube_h5_t fd, int global);


#ifdef __cplusplus
}
#endif

#endif /* CUBE_IO_H */
