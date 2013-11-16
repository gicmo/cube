// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifndef CUBE_ERROR_H
#define CUBE_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

enum _cube_status_t {
  CUBE_STATUS_OK  = 0,

  CUBE_ERROR_CUDA = 1,
  CUBE_ERROR_BLAS = 2,

  CUBE_ERROR_NO_MEMORY = 3,

  CUBE_ERROR_FAILED = 255

};

#ifdef __cplusplus
}
#endif

#endif
