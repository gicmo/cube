// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifndef MONITOR_STDOUT_H
#define MONITOR_STDOUT_H

#include <ica.h>

#ifdef __cplusplus
extern "C" {
#endif

  void monitor_stdout_new (ica_t *ica, int verbose, int update_freq);

#ifdef __cplusplus
}
#endif

#endif
