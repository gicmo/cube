// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include <string.h> //memcpy, memset
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <termios.h>

#include "sca.h"
#include "monitor_sca.h"


typedef struct monitor_sca_t {

  ica_monitor_t  parent;

  cube_h5_t      log;
  int            log_interval;

} monitor_sca_t;


static int
monitor_sca_do (ica_monitor_t *ica_monitor, ICAStatusType stype, int i, int imax, ica_t *ica, cube_t *ctx)
{
  monitor_sca_t *monitor;

  monitor = (monitor_sca_t *) ica_monitor;

  if (stype != 0 || i % monitor->log_interval != 0)
    {
      return 0;
    }

  cube_matrix_sync (ctx, ica->A, CUBE_SYNC_HOST);
  cube_matrix_sync (ctx, ica->prior->beta, CUBE_SYNC_HOST);

  sca_write_log (monitor->log, ica, i);

  return 0;
}

void
monitor_sca_new (ica_t *ica, int log_interval)
{
  monitor_sca_t *monitor;
  ica_monitor_t *ica_monitor;

  if (log_interval < 1)
    return;

  monitor = malloc (sizeof (monitor_sca_t));

  ica_monitor = (ica_monitor_t *) monitor;

  ica_monitor->mfunc = monitor_sca_do;
  ica_monitor->finish = NULL;
  ica_monitor->req_update_freq = log_interval;

  monitor->log = ica->fd;
  monitor->log_interval = log_interval;

  list_add (&ica->monitors, &((ica_monitor_t *) monitor)->mlist);
}


