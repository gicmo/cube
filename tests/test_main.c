#include <cube.h>
#include <cube_matrix.h>
#include <cube_blas.h>
#include <cube_math.h>
#include <cube_ica.h>
#include <cube_array.h>
#include <cube_io.h>

#include <time.h>
#include <math.h>

#include <string.h> //memcpy
#include <stdio.h>
#include <unistd.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

double max_iter_dev = 1.0e-15;
double max_math_dev = 1.0e-15;
double max_epts_dev = 3.0e-11;
double max_pinv_dev = 1.0e-15;

typedef enum TestFlags {

  TestFlag_Verbose = 1 << 0,
  TestFlag_Debug   = 1 << 1

} TestFlags;

typedef int (*TestFunction) (cube_t *, cube_h5_t, TestFlags);


typedef struct Test {

  const char *name;
  TestFunction func;

} Test;


static double
cube_matrix_compare_full (cube_matrix_t *mA, cube_matrix_t *mB, double tol, int *idx)
{
  double *A;
  double *B;
  double deviation;
  double cumdev;
  int m;
  int n;
  int i;

  if (mA == NULL && mB == NULL)
    return 0;

  if (mA == NULL || mB == NULL)
    return -1;

  m = cube_matrix_get_m (mA);
  n = cube_matrix_get_n (mB);

  if (m != cube_matrix_get_m (mB) ||
      n != cube_matrix_get_n (mB))
    return -2;

  A = cube_matrix_get_data (mA);
  B = cube_matrix_get_data (mB);

  cumdev = 0.0;
  deviation = 0.0;
  for (i = 0; i < m*n; i++)
    {
      double d = fabs (B[i] - A[i]);

      cumdev += d;

      if (d > deviation)
	{
	  printf ("New max dev: %e %d <%f %f>\n", d, i, B[i], A[i]);
	  if (idx)
	    *idx = i;

	  deviation = d;
	}
    }

  if (tol < 0)
    tol = 1.0e-15;

  deviation = cumdev/(m*n);
  printf ("mean dev: %e\n", cumdev/(m*n));

  return deviation < tol ? -1 * deviation : deviation;
}

static double
cube_matrix_compare (cube_matrix_t *mA, cube_matrix_t *mB, double tol)
{
  return cube_matrix_compare_full (mA, mB, tol, NULL);
}

static int
test_matrix (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_matrix_t *ones;
  cube_matrix_t *ref;
  double ref_data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double dev;
  int res;

  ones = cube_matrix_new_ones (ctx, 3, 2);
  ref = cube_matrix_new_from_data (ctx, 3, 2, ref_data, 0);

  dev = cube_matrix_compare (ones, ref, -1);

   if (flags & TestFlag_Debug)
     cube_matrix_dump (ones, 10, 10);

   res = dev < max_math_dev;
   printf (" * maximum deviation: %e [%e]\n", dev, max_math_dev);

  return !res;
}

static int
test_pinv (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_matrix_t *A;
  cube_matrix_t *Ai;
  cube_matrix_t *Ai_ref;
  double dev;
  int m, n;
  int res;

  A      = cube_h5_read_matrix (ctx, fd, "/test/pinv/A");
  Ai_ref = cube_h5_read_matrix (ctx, fd, "/test/pinv/Ai");

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);
  Ai = cube_matrix_create (ctx, m, n);

  if (flags & TestFlag_Debug)
    cube_matrix_dump (A, 10, 10);

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, Ai, CUBE_SYNC_DEVICE);
  cube_matrix_pinv (ctx, A, Ai);
  cube_matrix_sync (ctx, Ai, CUBE_SYNC_HOST);

  dev = cube_matrix_compare (Ai, Ai_ref, -1);

  if (flags & TestFlag_Debug)
    {
      cube_matrix_dump (Ai, 10, 10);
      cube_matrix_dump (Ai_ref, 10, 10);
    }

  res = dev < max_pinv_dev;
  printf (" * maximum deviation: %e [%e]\n", dev, max_pinv_dev);

  return !res;
}

static int
test_iiter (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_array_t *eps;
  cube_array_t *data;
  double       *iter;
  double       *ref;
  double        deviation;
  int m, n, i;
  int res;

  eps = cube_h5_read_array (fd, "test/interpolate/epsilon");

  m = cube_array_get_dim (eps, 0);
  n = cube_array_get_dim (eps, 1);

  if (flags & TestFlag_Verbose)
    {
      if (flags & TestFlag_Debug)
        printf ("m: %d n %d\n", m, n);

      for (i = 0; i < m; i++)
        {
          double step, epsilon;

          step = cube_array_get_double (eps, i, 0);
          epsilon = cube_array_get_double (eps, i, 1);

          printf ("\t%6.0f | %0.7f\n", step, epsilon);
        }
    }

  data = cube_h5_read_array (fd, "test/interpolate/samples");
  m = cube_array_get_dim (data, 0);
  n = cube_array_get_dim (data, 1);

  if (flags & TestFlag_Debug)
    printf ("m: %d n %d\n", m, n);

  iter = cube_array_get_data (data);
  ref = iter + m;

  deviation = 0.0;
  for (i = 0; i < m; i++)
    {
      double res;
      double cur_iter = iter[i];
      double cur_ref = ref[i];
      double cur_dev;

      res = cube_ica_inter_epsilon (ctx, eps, cur_iter);

      cur_dev = fabs (cur_ref - res);
      deviation = max (deviation, cur_dev);

      if (flags & TestFlag_Debug)
        printf (" %6.0f : %0.5f -> %0.5f [%e]\n", cur_iter, res, cur_ref, deviation);
    }

  res = deviation <  max_iter_dev;
  printf (" * maximum deviation: %e\n", deviation);

  return !res;
}

static int
test_epatches (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  ica_dataset_t *ds;
  cube_matrix_t *patches;
  cube_matrix_t *c0;
  double dev;
  int res;
  int idx;

  ds = cube_h5_read_dataset (fd, "/test/dataset");
  c0 = cube_h5_read_matrix (ctx, fd, "/test/dataset/c0");
  patches = cube_matrix_create (ctx, 294, ds->npats);

  cube_ica_extract_patches (ctx, ds, 0, patches);

  dev = cube_matrix_compare_full (patches, c0, -1, &idx);
  if (flags & TestFlag_Debug)
    {
      cube_matrix_dump (c0, 10, 10);
      cube_matrix_dump (patches, 10, 10);
    }

  res = dev < max_epts_dev;

  printf (" * maximum deviation: %e [%d]\n", fabs(dev), idx);

  //FIXME: free dataset
  return !res;
}

static int
test_host_math (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_matrix_t *X;
  cube_matrix_t *Xzm;
  cube_array_t  *ref_mean;
  cube_array_t  *ref_std;
  cube_array_t  *ref_var;
  double *x;
  double mean, std, var, rmean, rstd, rvar;
  double dev_mean, dev_std, dev_zm, dev_var;
  double max_dev;
  int m, n;
  int res;

  X = cube_h5_read_matrix (ctx, fd, "/test/math/data");

  m = cube_matrix_get_m (X);
  n = cube_matrix_get_n (X);
  x = cube_matrix_get_data (X);

  cube_host_dvmean (ctx, m*n, x, 1, &mean);
  cube_host_dvstd (ctx, m*n, x, 1, &std);
  cube_host_dvvar (ctx, m*n, x, 1, &var);

  ref_mean = cube_h5_read_array (fd, "/test/math/mean");
  ref_std = cube_h5_read_array (fd, "/test/math/std");
  ref_var = cube_h5_read_array (fd, "/test/math/var");

  rmean = cube_array_get_double (ref_mean, 0, 0);
  rstd = cube_array_get_double (ref_std, 0, 0);
  rvar = cube_array_get_double (ref_var, 0, 0);

  dev_mean = fabs (rmean - mean);
  dev_std = fabs (rstd - std);
  dev_var = fabs (rvar - var);

  if (flags & TestFlag_Verbose)
    {
      printf ("\t%f %f | %e\n", rmean, mean, dev_mean);
      printf ("\t%f %f | %e\n", rvar, var, dev_var);
      printf ("\t%f %f | %e\n", rstd, std, dev_std);
    }

  cube_host_dvizm (ctx, m*n, x, 1);
  Xzm = cube_h5_read_matrix (ctx, fd, "/test/math/data_zm");

  dev_zm = cube_matrix_compare (X, Xzm, -1);

  if (flags & TestFlag_Verbose)
    printf ("\t %e\n", dev_zm);

  max_dev = max(max(dev_mean, dev_std), max(dev_var, dev_zm));
  res = max_dev < max_math_dev;

  printf (" * maximum deviation: %e\n", max_dev);

  return !res;
}

static int
test_rescale_bfs (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_matrix_t *D;
  cube_matrix_t *A_ref;
  cube_matrix_t *A;
  double std;
  double dev;
  double *v;
  int m, n;
  int res;

  A = cube_h5_read_matrix (ctx, fd, "/test/dataset/Ainit");
  D = cube_h5_read_matrix (ctx, fd, "/test/dataset/D");
  A_ref   = cube_h5_read_matrix (ctx, fd, "/test/dataset/A_scaled");

  m = cube_matrix_get_m (D);
  n = cube_matrix_get_n (D);
  v = cube_matrix_get_data (D);
  cube_host_dvstd (ctx, m*n, v, 1, &std);

  if (flags & TestFlag_Verbose)
    printf ("\t std: %e\n", std);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);
  v = cube_matrix_get_data (A);
  cube_host_dscal (ctx, m*n, std, v, 1);


  dev = cube_matrix_compare (A, A_ref, -1);
  res = dev < max_math_dev;

  printf (" * maximum deviation: %e\n", dev);

  return !res;
}

static int
test_ica_calc_dA (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_matrix_t *A;
  cube_matrix_t *dA;
  cube_matrix_t *S;
  cube_matrix_t *mu;
  cube_matrix_t *sigma;
  cube_matrix_t *beta;
  cube_matrix_t *Aref;
  double dev;
  int m, n;
  int res;

  A = cube_h5_read_matrix (ctx, fd, "/test/ica/A");
  S = cube_h5_read_matrix (ctx, fd, "/test/ica/S");
  mu = cube_h5_read_matrix (ctx, fd, "/test/ica/mu");
  sigma = cube_h5_read_matrix (ctx, fd, "/test/ica/sigma");
  beta = cube_h5_read_matrix (ctx, fd, "/test/ica/beta");

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, S, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, mu, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, sigma, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, beta, CUBE_SYNC_DEVICE);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);
  dA = cube_matrix_create (ctx, m, n);
  cube_matrix_sync (ctx, dA, CUBE_SYNC_DEVICE);

  cube_ica_calc_dA (ctx, A, S, mu, beta, sigma, dA);

  cube_matrix_sync (ctx, dA, CUBE_SYNC_HOST);

  Aref = cube_h5_read_matrix (ctx, fd, "/test/ica/dA");

  dev = cube_matrix_compare (dA, Aref, -1);
  res = dev < max_math_dev;

  cube_matrix_destroy (ctx, dA);
  cube_matrix_destroy (ctx, A);
  cube_matrix_destroy (ctx, S);
  cube_matrix_destroy (ctx, mu);
  cube_matrix_destroy (ctx, sigma);

  res = dev < max_math_dev;
  printf (" * maximum deviation: %e\n", dev);
  return !res;
}

static int
test_ica_AdA (cube_t *ctx, cube_h5_t fd, TestFlags flags)
{
  cube_matrix_t *A;
  cube_matrix_t *dA;
  cube_matrix_t *Aref;
  cube_array_t  *epsilon;
  double eps;
  double dev;
  int res;

  A = cube_h5_read_matrix (ctx, fd, "/test/ica/A");
  dA = cube_h5_read_matrix (ctx, fd, "/test/ica/dA");
  epsilon = cube_h5_read_array (fd, "/test/ica/eps");

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, dA, CUBE_SYNC_DEVICE);

  eps = cube_array_get_double (epsilon, 0, 0);
  cube_ica_AdA (ctx, A, dA, eps);

  cube_matrix_sync (ctx, A, CUBE_SYNC_HOST);
  Aref = cube_h5_read_matrix (ctx, fd, "/test/ica/Aref");
  dev = cube_matrix_compare (A, Aref, -1);

  res = dev < max_math_dev;

  cube_matrix_destroy (ctx, dA);
  cube_matrix_destroy (ctx, A);

  res = dev < max_math_dev;
  printf (" * maximum deviation: %e\n", dev);
  return !res;
}

static double
find_eps()
{
  return nextafter(1.0, 2.0) - 1.0;
}

Test tests[] = {
  {"Matrix",          test_matrix},
  {"Math",            test_host_math},
  {"Interpolate",     test_iiter},
  {"Extract patches", test_epatches},
  {"Pseudo-Inverse",  test_pinv},
  {"Rescale BFS",     test_rescale_bfs},
  {"ICA: calc dA",    test_ica_calc_dA},
  {"ICA: AdA",        test_ica_AdA},
  {NULL, NULL}
};

static void
get_current_time (struct timespec *ts)
{
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;

  host_get_clock_service (mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time (cclock, &mts);
  mach_port_deallocate (mach_task_self(), cclock);

  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime (CLOCK_REALTIME, ts);
#endif

}

static void
usage(const char *prgname)
{
  printf ("usage %s [-vd] <test_data.h5>\n", prgname);
}

int
main (int argc, char **argv)
{
  TestFlags flags;
  cube_t *ctx;
  cube_h5_t fd;
  char *filename;
  double eps;
  Test  *test;
  int    fcount;
  int    tcount;
  int    c;
  int    gpu;

  flags = 0;
  gpu = 1;

  opterr = 0;

  while ((c = getopt (argc, argv, "dvH")) != -1)
    switch (c)
      {
      case 'd':
	flags |= TestFlag_Debug;
	break;

      case 'v':
	flags |= TestFlag_Verbose;
	break;

      case 'H':
        gpu = 0;
        break;

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

  eps = find_eps ();
  max_math_dev = eps * 10;
  max_iter_dev = eps;
  max_pinv_dev = eps * 1000;

  printf ("Machine epsilon: %e\n", eps);

  filename = argv[optind];

  ctx = cube_context_new (gpu);

  if (! cube_context_check (ctx))
    {
      fprintf (stderr, "Could not create cube context!\n");
      cube_context_destroy (&ctx);
      return -1;
    }

  fd = cube_h5_open (filename, 0);

  fcount = 0;
  for (test = tests; test->name; test++)
    {
      int failed;
      double ct;
      struct timespec tic, toc;

      printf ("\n** \033[34m%s\033[39m: \n", test->name);

      get_current_time (&tic);
      failed = ! cube_context_check (ctx) || test->func (ctx, fd, flags);
      get_current_time (&toc);

      if (failed)
        fcount++;

      ct = ((toc.tv_sec - tic.tv_sec)  + (double) (toc.tv_nsec - tic.tv_nsec) / 1000000000.0) * 1000.0;
      printf (" * DONE:%s [%f ms]\n", (!failed) ? "\033[32m pass\033[39m" : "\033[31m FAILED\033[39m", ct);
    }

  tcount = (sizeof (tests) / sizeof (Test)) -1;

  printf ("\n** result:");
  if (fcount > 0)
    printf ("\033[31m FAIL\033[39m");
  else
    printf ("\033[32m ok\033[39m");

  printf (" [%d tests, %d passed, %s%d\033[39m failed]\n",
	  tcount,
	  tcount - fcount,
	  fcount > 0 ? "\033[31m" : "",
	  fcount);

  cube_h5_close (fd);
  cube_context_destroy (&ctx);

  return 0;
}

