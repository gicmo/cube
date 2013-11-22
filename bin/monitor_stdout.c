// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include <string.h> //memcpy, memset
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <termios.h>

#include <math.h>

#include "monitor_stdout.h"

static int
term_get_chars ()
{
  struct winsize ws;
  int res;

  res = ioctl (STDOUT_FILENO, TIOCGWINSZ, &ws);

  if (res == -1)
    {
      return 80;
    }

  return ws.ws_col;
}

typedef struct monitor_std_t {

  ica_monitor_t  parent;

  int            nchars;
  struct timeval t_start;
  int            verbose;

} monitor_std_t;




static void
show_dataset_info (ica_dataset_t *ds)
{
  FILE *stream = stdout;
  int i, n;
  char *data;

  fprintf (stream, "Dataset [%s] {%p}\n", ds->id, (void *) ds);
  fprintf (stream, "\t Dim: %d\n", ds->dim);
  fprintf (stream, "\t Channel: %d\n", ds->nchannel);
  fprintf (stream, "\t Patchsize: %d\n", ds->patchsize);
  fprintf (stream, "\t Blocksize: %d\n", ds->blocksize);
  fprintf (stream, "\t imgdata: %d\n", cube_array_get_dim (ds->imgdata, 3));
  fprintf (stream, "\t channels: ");
  n = cube_array_get_dim (ds->channels, 1);
  data = (char *) cube_array_get_data (ds->channels);

  for (i = 0; i < n; i++)
    fprintf (stream, "%d ", data[i]);
  fprintf (stream, "\n");

}


static void
draw_status_bar (const char *cfgid,
		 char        status_char,
		 int         i,
		 int         imax,
		 double      t_elapsed)
{
  int nchars, pchars, n;
  int j;

  nchars = term_get_chars ();

  pchars = nchars - 44;
  n = ceil (i*((double) pchars/imax));

  printf ("%.7s (%5d/%5d) [", cfgid, i, imax);

  for (j = 0; j < n-1; j++)
    printf ("=");

  printf ("%c", status_char);

  while (n++ < pchars)
    printf (" ");

  printf ("] ");
  fflush (stdout);

  if (i == 0)
    printf ("\r");
  else
    printf ("%7.1fs (%6.0fs)\r", t_elapsed, (t_elapsed * imax)/i);
}

static int
monitor_stdout_do (ica_monitor_t *ica_monitor, ICAStatusType stype, int i, int imax, ica_t *ica, cube_t *ctx)
{
  monitor_std_t *monitor;
  double t_elapsed;
  char sc;

  monitor = (monitor_std_t *) ica_monitor;

  if (stype == 0 && i == 0)
    {
      gettimeofday (&monitor->t_start, NULL);
      if (monitor->verbose > 0)
	{
	  int n;
	  double *data;

	  show_dataset_info (ica->dataset);

	  n = cube_array_get_dim (ica->epsilon, 0);
	  data = cube_array_get_data (ica->epsilon);
	  printf ("Epsilon: ");
	  for (int i = 0; i < n; i++)
	    {
	      printf ("%f ", data[n+i]);
	    }
	  printf ("\n");
	}

    }

  switch (stype)
    {
    case 0:
    case 2:
    case 4:
      sc = '-';
      break;

    case 1:
      sc = 'E';
      break;

    case 3:
      sc = 'P';
      break;
    }

  t_elapsed =  calc_t_elapsed (&monitor->t_start);
  draw_status_bar (ica->cfgid, sc, i, imax, t_elapsed);

  if (i == imax)
    printf ("\n[model: %.7s, ds: %.7s, cfg: %.7s] done in %f seconds.\n",
	    ica->id, ica->dataset->id, ica->cfgid, ica->fit_time);

  return 0;
}

void
monitor_stdout_new (ica_t *ica, int verbose, int update_freq)
{
  monitor_std_t *monitor;
  ica_monitor_t *ica_monitor;

  monitor = malloc (sizeof (monitor_std_t));
  monitor->verbose = verbose;

  ica_monitor = (ica_monitor_t *) monitor;

  ica_monitor->mfunc = monitor_stdout_do;
  ica_monitor->finish = NULL;
  ica_monitor->req_update_freq = update_freq > 0 ? update_freq : 50;

  list_add (&ica->monitors, &((ica_monitor_t *) monitor)->mlist);
}
