// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#ifdef __APPLE__
  #include <CommonCrypto/CommonDigest.h>
#else
  #include <openssl/sha.h>
#endif

#include "version.h"
#include "sca.h"
#include "ica.h"
#include "monitor_oglui.h"
#include "monitor_stdout.h"
#include "monitor_sca.h"

#include <time.h>
#include <math.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

#define USEC 1000000.0


typedef struct ica_options_t {
  int verbose;
  int log;
  int glui;
  int update_freq;
} ica_options_t;

double
calc_t_elapsed (struct timeval *tic)
{
  struct timeval tac;
  double dt;

  gettimeofday (&tac, NULL);

  dt = ((tac.tv_sec*USEC + tac.tv_usec) -
	(tic->tv_sec*USEC + tic->tv_usec))/USEC;

    return dt;
}

static void
ica_call_monitor (ica_t *ica, ICAStatusType stype, int i, int imax, cube_t *ctx)
{
  ica_monitor_t *monitor = NULL;

  list_for_each (&ica->monitors, monitor, mlist)
    {
      if (monitor && monitor->mfunc)
	monitor->mfunc (monitor, stype, i, imax, ica, ctx);
    }
}

static void
ica_monitor_call_finish (ica_t *ica)
{
  ica_monitor_t *next, *monitor = NULL;

  list_for_each_safe (&ica->monitors, monitor, next, mlist)
    {
      list_del (&monitor->mlist);

      if (monitor && monitor->finish)
	monitor->finish (monitor);
    }
}

static int
ica_monitor_get_min_update_freq (ica_t *ica)
{
 ica_monitor_t *monitor = NULL;
 int freq = INT_MAX;

 list_for_each (&ica->monitors, monitor, mlist)
   freq = min (freq, monitor->req_update_freq);

 return freq;
}

static void
rescale_bfs (cube_t *ctx, int n, double *d, int incd, cube_matrix_t *A)
{
  double *v;
  double std;
  int m;

  cube_host_dvstd (ctx, n, d, incd, &std);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);
  v = cube_matrix_get_data (A);
  cube_host_dscal (ctx, m*n, std, v, 1);
}

#if 0
static int
epsilon_get_max_iter (cube_array_t *epsilon)
{
  int m;

  m = cube_array_get_dim (epsilon, 0);
  return (int) cube_array_get_double (epsilon, m-1, 0) + 1;
}
#endif

static void
do_ica (cube_t *ctx, ica_t *ica)
{
  struct timeval tic;
  cube_matrix_t *A;
  cube_matrix_t *dA;
  ica_dataset_t *dataset;
  cube_array_t  *epsilon;
  cube_matrix_t *patches;
  cube_matrix_t *S, *D;
  cube_matrix_t *Ai;
  cube_matrix_t *coeffs;
  ica_prior_t   *prior;
  double eps;
  double a, b;
  double *p, *d;
  double *tmp;
  int blocksize;
  int i;
  double npats;
  int m, n;
  int max_iter;
  int prior_adapt_size;
  int prior_index;
  int update_freq;
  size_t poff;

  prior_adapt_size = ica->prior->adapt_size;
  dataset = ica->dataset;
  blocksize = dataset->blocksize;
  npats = dataset->npats;
  max_iter = dataset->maxiter;
  prior = ica->prior;
  epsilon = ica->epsilon;

  m = cube_array_get_dim (dataset->Ainit, 0);
  n = cube_array_get_dim (dataset->Ainit, 1);
  tmp = cube_array_get_data (dataset->Ainit);

  A = cube_matrix_new_from_data (ctx, m, n, tmp, 0);

  patches = cube_matrix_create (ctx, m, npats);

  /* rescale the basis functions, we need the first set of patches*/
  cube_ica_extract_patches (ctx, dataset, 0, patches);
  p = cube_matrix_get_data (patches);
  rescale_bfs (ctx, m*blocksize, p, 1, A);

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);

  Ai = cube_matrix_new_on_device (ctx, m, n);
  dA = cube_matrix_new_on_device (ctx, m, n);
  D =  cube_matrix_new_on_device (ctx, m, blocksize);
  S =  cube_matrix_new_on_device (ctx, m, blocksize);
  coeffs = cube_matrix_new_on_device (ctx, m, prior_adapt_size);

  cube_matrix_sync (ctx, prior->mu, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, prior->sigma, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, prior->beta, CUBE_SYNC_DEVICE);

  ica->A = A;

  /* all setup is done now */
  ica_call_monitor (ica, 0, 0, max_iter, ctx);
  gettimeofday (&tic, NULL);
  update_freq = ica_monitor_get_min_update_freq (ica);

  for (i = 1; i < max_iter; i++)
    {
       int blocks;
       int block;
       int cluster;
       int idx;

       if (!cube_context_check(ctx)) {
         break; //Error abort!
       }

       blocks = npats / blocksize;
       block = ((i - 1) % blocks) + 1;
       cluster = ((i - block) / blocks);
       idx = ((block - 1) * blocksize);

       if (i % update_freq == 0)
	 ica_call_monitor (ica, 0, i, max_iter, ctx);

       if (block == 1)
         {
	   ica_call_monitor (ica, 1, i, max_iter, ctx);
           cube_ica_extract_patches (ctx, dataset, cluster, patches);
	   ica_call_monitor (ica, 2, i, max_iter, ctx);
         }

       d = D->dev_ptr;
       p = cube_matrix_get_data (patches);

       poff = m*idx;
       //memcpy (d, p+poff, m*blocksize*sizeof(double)); //FIXME: matrix_copy_region
       cube_memcpy (ctx, d, p+poff, m*blocksize*sizeof(double), CMK_HOST_2_DEVICE);

       /* ************ */
       /* S = A^-1 * D */
       cube_matrix_pinv (ctx, A, Ai);

       a = 1.0;
       b = 0.0;
       cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, Ai, D, &b, S);

       /* ************ */
       /* collect coefficients for prior adaption */
       prior_index = (blocksize*(i-1)) % prior_adapt_size;
       cube_ica_prior_collect (ctx, S, coeffs, prior_index);

       /* adapt beta of the prior */
       //FIXME a, b
       if (prior_index + blocksize == prior_adapt_size)
	 {
	   ica_call_monitor (ica, 3, i, max_iter, ctx);
	   cube_ica_prior_adapt (ctx,
				 prior->a,
				 prior->b,
				 coeffs,
				 prior->beta,
				 prior->mu,
				 prior->sigma,
				 prior->tol);
	   ica_call_monitor (ica, 4, i, max_iter, ctx);
	 }
       /* calc dA and perform the update of A */
       cube_ica_calc_dA (ctx, A, S, prior->mu, prior->beta, prior->sigma, dA);

       eps = cube_ica_inter_epsilon (ctx, epsilon, i-1);
       cube_ica_AdA (ctx, A, dA, eps);
    }

  ica->fit_time = calc_t_elapsed (&tic);
  ica_call_monitor (ica, 0, i, max_iter, ctx);

  cube_matrix_sync (ctx, A, CUBE_SYNC_HOST);
  cube_matrix_sync (ctx, prior->beta, CUBE_SYNC_HOST);

  cube_matrix_destroy (ctx, Ai);
  cube_matrix_destroy (ctx, dA);
  cube_matrix_destroy (ctx, D);
  cube_matrix_destroy (ctx, S);
  cube_matrix_destroy (ctx, coeffs);
}


static void
usage (const char *prgname)
{
  printf ("cube ica :: version %s [%.7s]\n", version, version_git);
  printf ("usage: %s <datafile.h5>\n", prgname);
}

static char *
hex_encode_data (const unsigned char *data, int n)
{
  char *str;
  int i;
  int k;

  k = n*2+1;
  str = malloc (k);

  for (i = 0; i < n; i++)
    {
      int offset = 2*i;
      snprintf (str + offset, k - offset, "%02x", data[i]);
    }

  return str;
}

static char *
ctime_str ()
{
  char str[13] = {'\0', };
  struct tm ltime;
  time_t now;

  now = time(NULL);
  localtime_r (&now, &ltime);
  strftime (str, sizeof (str), "%Y%m%d%H%M", &ltime);

  return strdup (str);
}

static char *
creator_str ()
{
  char buffer[255] = {'\0', };
  char *str;

  snprintf (buffer, sizeof (buffer), "ica+cube [%.7s]", version_git);

  str = strdup (buffer);
  return str;
}

#ifdef __APPLE__
  #define SHA_CTX CC_SHA1_CTX
  #define SHA1_Init CC_SHA1_Init
  #define SHA1_Update CC_SHA1_Update
  #define SHA1_Final CC_SHA1_Final
#endif

static void
ica_gen_id(ica_t *ica)
{
  SHA_CTX sha1;
  unsigned char md[20];

  SHA1_Init (&sha1);
  SHA1_Update (&sha1, ica->cfgid, strlen (ica->cfgid));
  SHA1_Update (&sha1, ica->dataset->id, strlen (ica->dataset->id));
  SHA1_Update (&sha1, ica->creator, strlen (ica->creator));
  SHA1_Update (&sha1, ica->ctime, strlen (ica->ctime));
  SHA1_Update (&sha1, &ica->gpu, sizeof (ica->gpu));
  SHA1_Final (md, &sha1);

  ica->id = hex_encode_data (md, 20);
}

static ica_t *
setup_ica (cube_t *ctx, cube_h5_t fd, const char *cfgid, const char *dsid)
{
  ica_t *ica;
  ica_prior_t *prior;
  ica_dataset_t *ds;
  sca_config_t *cfg;

  ica = malloc (sizeof (ica_t));
  ica->gpu = cube_context_is_gpu (ctx);
  ica->creator = creator_str ();
  ica->ctime = ctime_str ();

  ds = sca_dataset_read (fd, cfgid, dsid);

  cfg = sca_config_read (fd, cfgid);
  prior = sca_config_create_prior (ctx, cfg, ds->dim);
  ica->epsilon = sca_config_create_gradient (ctx, cfg);
  ica->cfgid = strdup (sca_config_get_id (cfg));
  ica->prior = prior;
  ica->dataset = ds;

  sca_config_free (cfg);

  ica_gen_id (ica);

  ica->fd = fd;

  list_head_init (&ica->monitors);

  return ica;
}

static void
free_dataset (ica_dataset_t *dataset)
{
  free (dataset->id);

  cube_array_destroy (dataset->imgdata);
  cube_array_destroy (dataset->indicies);
  cube_array_destroy (dataset->patsperm);
  cube_array_destroy (dataset->Ainit);

  free (dataset);
}

static void
free_prior (cube_t *ctx, ica_prior_t *prior)
{
  cube_matrix_destroy (ctx, prior->mu);
  cube_matrix_destroy (ctx, prior->sigma);
  cube_matrix_destroy (ctx, prior->beta);

  free (prior);
}

static void
free_ica (cube_t *ctx, ica_t *ica)
{
  free (ica->id);
  free (ica->cfgid);

  cube_matrix_destroy (ctx, ica->A);

  free_dataset (ica->dataset);
  free_prior (ctx, ica->prior);

  cube_array_destroy (ica->epsilon);
  free (ica->ctime);
  free (ica->creator);

  free (ica);
}

static void
setup_and_run_ica (cube_t     *ctx,
                   const char *filename,
                   const char *cfgid,
                   const char *dsid,
                   ica_options_t *options)
{
  ica_t *ica;
  cube_h5_t fd;
  int   res;

  fd = cube_h5_open (filename, 0);

  if (fd < 0)
    {
      fprintf (stderr, "Could not open %s\n", filename);
      return;
    }

  if (cfgid == NULL || cfgid[0] == '\0')
    {
      if (sca_config_count (fd) > 1)
	{
	  fprintf (stderr, "Config id not specificed (and not unique in file)\n");
	  return;
	}

      cfgid = sca_config_name (fd, 0);
    }

  if (dsid == NULL)
    {
      if (sca_dataset_count (fd, cfgid) > 1)
	{
	  fprintf (stderr, "Dataset id not specificed (and not unique in file)\n");
          return;
	}

      dsid = sca_dataset_name (fd, cfgid, 0);
    }

  ica = setup_ica (ctx, fd, cfgid, dsid);

  monitor_stdout_new (ica, options->verbose, options->update_freq);

  if (options->glui)
    monitor_oglui_new (ica);

  if (options->log > 0)
    monitor_sca_new (ica, options->log);

  do_ica (ctx, ica);

  res = cube_context_check (ctx);
  if (res)
    sca_write_model (fd, ica);
  else
    fprintf(stderr, "\nError during calculation!\n");

  ica_monitor_call_finish (ica);

  free_ica (ctx, ica);
  cube_h5_close (fd);

}

static void
batch_process_ica (cube_t *ctx, const char *filename, ica_options_t *options)
{
  FILE *fd;
  char path[256];
  char cfgid[41];
  char dsid[41];
  int  n;

  fd = fopen (filename, "r");

  if (fd == NULL)
    {
      fprintf (stderr, "Could not open batch file\n");
      return;
    }

  do {
    memset (path, '\0', sizeof (path));
    memset (cfgid, '\0', sizeof (cfgid));
    memset (dsid, '\0', sizeof (dsid));

    n = fscanf (fd, "%255[^\n]%*c", path);
    if (n > 0)
      {
        printf ("Running ICA for [%s, %s, %s]\n\n", path, cfgid, dsid);
        setup_and_run_ica (ctx, path, cfgid, dsid, options);
      }

  } while (cube_context_check (ctx) && feof (fd) == 0);


  fclose (fd);
}

int
main (int argc, char **argv)
{
  ica_options_t options = {0, 0, 0, 0};
  cube_t *ctx;
  char *filename;
  char *cfgid;
  char *dsid;
  int   gpu;
  int   res;
  int   batch;
  int   c;

  cfgid = dsid = NULL;
  opterr = 0;
  batch = 0;
  gpu = 1;

  memset (&options, 0, sizeof (options));

  while ((c = getopt (argc, argv, "Bd:gvfhHl:U:")) != -1)
    switch (c)
      {
      case 'v':
        options.verbose = 1;
        break;

      case 'H':
        gpu = 0;
        break;

      case 'B':
        batch = 1;
        break;

      case 'l':
	options.log = atoi(optarg);
	break;

      case 'g':
	options.glui = 1;
	break;

      case 'U':
	options.update_freq = atoi (optarg);
	break;

      case 'd':
	dsid = optarg;
	break;

      case 'h':
      case '?':
      default:
	usage (argv[0]);
	return 0;
      }
  
  if (!(optind < argc))
    {
      usage (argv[0]);
      return -1;
    }


  filename = argv[optind];
  ctx = cube_context_new (gpu);
  
  if (! cube_context_check (ctx))
    {
      fprintf (stderr, "Error during cube context creation\n");
      cube_context_destroy (&ctx);
      return -1;
    }

  if (batch)
    batch_process_ica (ctx, filename, &options);
  else
    setup_and_run_ica (ctx, filename, cfgid, dsid, &options);

  res = cube_context_check (ctx);
  cube_context_destroy (&ctx);

  return ! res;
}
