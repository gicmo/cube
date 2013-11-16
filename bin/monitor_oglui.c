// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include "cube.h"
#include "cube_matrix.h"
#include "cube_array.h"
#include "cube_io.h"
#include "cube_math.h"
#include "ica.h"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <math.h>
#include <limits.h>

#include "list.h"
#include "monitor_oglui.h"

/* prototypes */
typedef  void (*idle_callback_t) (void *data);
static void idle_add (idle_callback_t cb, void *data);

/* the globals */
static volatile int  need_update_textures = 0;
static cube_array_t *grid = NULL;
static float   spacing = 0.01;
static float   space;
static GLuint *textures = NULL;
static int     n_textures = 0;

/* THE grid! */

static void
grid_update (int w, int h)
{
  int x, y, n, i;
  int cols;
  float r;
  float border;
  float *gd;

  if (grid == NULL)
    return;

  n = cube_array_get_dim (grid, 0) / 2;

  r = (float) w / (float) h;
  cols = ceil (sqrt (n * r)) + 1;
  fprintf (stderr, "%d %d\n", cols, (int) ceil (sqrt (n / r)) + 1);
  border = min (w, h) * spacing;
  space = (w - (cols+1)*border) / cols;

  gd = cube_array_get_data (grid);

  for (i = 0; i < n; i++)
    {
      div_t q;
      int p;

      q = div (i, cols);

      x = q.rem;
      y = q.quot + 1;

      p = i * 2;
      gd[p] = x * space + (x+1) * border;
      gd[p+1] = h - (y * space + y * border);
    }
}

static void
grid_init (int n)
{
  GLint dim[4] = {0, };

  grid = cube_array_new (cube_dtype_float32, 1, n*2);
  glGetIntegerv (GL_VIEWPORT, dim);
  grid_update ((int) dim[2], (int) dim[3]);
}

/* Textures */
static void
free_textures ()
{
  int n;

  if (textures == NULL)
    return;

  n = n_textures;
  n_textures = 0;

  glDeleteTextures (n, textures);
  free (textures);

  textures = NULL;
}

static void
gen_textures (int n)
{
  int i;
  GLuint test = 0;

  glGenTextures (1, &test);
  textures = malloc (sizeof (GLuint) * n);
  glGenTextures (n, textures);

  for (i = 0; i < n; i++)
    {
      float bf[3*7*7] = {0.0, };

      glBindTexture (GL_TEXTURE_2D, textures[i]);

      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexImage2D (GL_TEXTURE_2D, 0, 3, 7, 7, 0, GL_BGR, GL_FLOAT, bf);
    }

  n_textures = n;
}

static void
update_textures (void *user_data)
{
  cube_array_t *nA;
  float *data;
  int i;
  int m;

  nA = user_data;
  m = cube_array_get_dim (nA, 0);
  data = cube_array_get_data (nA);

  for (i = 0; i < n_textures; i++)
    {
      float *bf;

      bf = data + (i * m);
      glBindTexture (GL_TEXTURE_2D, textures[i]);
      glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, 7, 7, GL_BGR, GL_FLOAT, bf);
    }

  cube_array_destroy (nA);
  glutPostRedisplay ();
}

typedef struct init_args_t {
  int m;
  int n;
} init_args_t;

static void
init_display (void *user_data)
{
  init_args_t *args = user_data;

  grid_init (args->n);
  gen_textures (args->n);

  free (args);
}

static void
glui_ica_start (ica_t *ica)
{
  init_args_t *args;
  int m, n;

  m = cube_matrix_get_m (ica->A);
  n = cube_matrix_get_n (ica->A);

  args = malloc (sizeof (init_args_t));
  args->m = m;
  args->n = n;

  idle_add (init_display, args);
}

extern void bf_to_lms_ (int *m, int *n, double *A, int *k, float *B, int *nch, char *C);
extern void bf_to_float_ (int *m, int *n, double *A, float *B);
extern void bff_normalize_ (int *m, int *n, float *A);


static void
glui_update_A (cube_t *ctx, ica_t *ica)
{
  cube_array_t *nA;
  double *source;
  float  *sink;
  char   *chans;
  int     n;
  int     m;
  int     c;
  int     k;

  if (need_update_textures == 1)
    return;

  cube_matrix_sync (ctx, ica->A, CUBE_SYNC_HOST);
  m = cube_matrix_get_n (ica->A);
  n = cube_matrix_get_m (ica->A);

  c = cube_array_get_dim (ica->dataset->channels, 1);
  k = (m / c) * 3; // L M S (or pseudo-BGR)

  nA = cube_array_new (cube_dtype_float32, 2, k, n);

  source = cube_matrix_get_data (ica->A);
  sink = cube_array_get_data (nA);
  chans = cube_array_get_data (ica->dataset->channels);

  bf_to_lms_ (&m, &n, source, &k, sink, &c, chans);
  bff_normalize_ (&k, &n, sink);

  need_update_textures = 1;
  idle_add (update_textures, nA);
}

/* idle handling */

static pthread_mutex_t idle_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct idle_queue_t {

  struct list_node list;

  idle_callback_t callback;
  void  *data;
} idle_queue_t;

static struct list_head idle_queue;

static void
idle ()
{
  idle_queue_t *next, *entry = NULL;

  if (pthread_mutex_trylock (&idle_lock) != 0)
    return;

  list_for_each_safe (&idle_queue, entry, next, list)
    {
      entry->callback (entry->data);
      list_del (&entry->list);
      free (entry);
    }

  pthread_mutex_unlock (&idle_lock);
  return;
}

static void
idle_add (idle_callback_t cb, void *data)
{
  idle_queue_t *entry;

  entry = malloc (sizeof (idle_queue_t));
  entry->data = data;
  entry->callback = cb;

  pthread_mutex_lock (&idle_lock);
  list_add_tail (&idle_queue, &entry->list);
  pthread_mutex_unlock (&idle_lock);
}

static void
display (void)
{
  int    i;
  float *gd;

  glClear (GL_COLOR_BUFFER_BIT);

  if (grid == NULL)
    {
      glutSwapBuffers ();
      return;
    }

  gd = cube_array_get_data (grid);

  for (i = 0; i < n_textures; i++)
    {
      float x, y;

      x = gd[2*i];
      y = gd[2*i + 1];

      glBindTexture (GL_TEXTURE_2D, textures[i]);

      glBegin(GL_QUADS);
      {
	glTexCoord2f (0.0, 0.0); glVertex2f (x, y);
	glTexCoord2f (0.0, 1.0); glVertex2f (x, y+space);
	glTexCoord2f (1.0, 1.0); glVertex2f (x+space, y+space);
	glTexCoord2f (1.0, 0.0); glVertex2f (x+space, y);
      }
      glEnd();
      //printf ("i:%d : x, y: %f %f\n", i, x, y);
    }

  //printf ("N: %d %d\n", i, n_textures);

  glutSwapBuffers ();
  need_update_textures = 0;
}

static void
resize(int w, int h)
{
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  glViewport (0, 0, w, h);
  gluOrtho2D (0.0, w, 0.0, h);
  glMatrixMode (GL_MODELVIEW);

  //create the grid
  grid_update (w, h);
}

static void *
gui_loop (void *arg)
{
  int wid;
  int argc;
  char *argv = {"ica"};

  argc = 1;

  glutInit (&argc, &argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize (640, 480);
  wid = glutCreateWindow ("ICA");

  glClearColor (1.0 ,1.0, 1.0, 0.0);
  glColor3f (0.0, 0.0, 0.0);
  glPointSize (4.0);
  glDisable (GL_DEPTH_TEST);
  glEnable (GL_TEXTURE_2D);
  glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei (GL_PACK_ALIGNMENT, 1);
  glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glClear (GL_COLOR_BUFFER_BIT);

  glutReshapeFunc (resize);
  glutDisplayFunc (display);
  glutIdleFunc (idle);

  glutMainLoop ();

  return NULL;
}

typedef struct oglui_monitor_t {
  ica_monitor_t  parent;
  pthread_t      thread;
} oglui_monitor_t;

static int
oglui_monitor (ica_monitor_t *m, ICAStatusType stype, int i, int imax, ica_t *ica, cube_t *ctx)
{
  oglui_monitor_t *monitor;

  monitor = (oglui_monitor_t *) m;

  if (stype == 0 && i == 0)
    glui_ica_start (ica);

  if (stype == 0)
    glui_update_A (ctx, ica);

  return 0;
}

static void
oglui_exit ()
{
  //free_textures ();
  pthread_exit (NULL);
}

static void
oglui_finish (ica_monitor_t *ica_monitor)
{
  oglui_monitor_t *monitor;
  void *retval;

  monitor = (oglui_monitor_t *) ica_monitor;

  idle_add (oglui_exit, NULL);
  pthread_join (monitor->thread, &retval);
  free_textures ();
  cube_array_destroy (grid);
  grid = NULL;
}

ica_monitor_t *
monitor_oglui_new (ica_t *ica)
{
  oglui_monitor_t *monitor;
  ica_monitor_t   *ica_monitor;

  monitor = malloc (sizeof (oglui_monitor_t));

  ica_monitor = (ica_monitor_t *) monitor;
  ica_monitor->mfunc = oglui_monitor;
  ica_monitor->finish = oglui_finish;
  ica_monitor->req_update_freq = INT_MAX;

  list_head_init (&idle_queue);

  pthread_create (&monitor->thread, NULL, gui_loop, NULL);

  list_add (&ica->monitors, &((ica_monitor_t *) monitor)->mlist);
  return (ica_monitor_t *) monitor;
}

