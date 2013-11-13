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

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <acml.h>
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>

#include <pthread.h>
#include <math.h>
#include <limits.h>

#include "list.h"


/* */
typedef  void (*idle_callback_t) (void *data);
static void idle_add (idle_callback_t cb, void *data);
static void idle ();

static pthread_mutex_t idle_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct idle_queue_t {

  struct list_head list;

  idle_callback_t callback;
  void  *data;
} idle_queue_t;

static idle_queue_t idle_queue;

static void
idle_add (idle_callback_t cb, void *data)
{
  idle_queue_t *entry;

  entry = malloc (sizeof (idle_queue_t));
  entry->data = data;
  entry->callback = cb;

  pthread_mutex_lock (&idle_lock);
  list_add_tail (&entry->list, &idle_queue.list);
  pthread_mutex_unlock (&idle_lock);
}

/* */
extern void bf_to_lms_ (int *m, int *n, double *A, int *k, float *B, int *nch, char *C);
extern void bff_normalize_ (int *m, int *n, float *A);

static int
argsrt_cmp_inv (const void *p1, const void *p2)
 {
   const double *a, *b;
   a = p1;
   b = p2;

   //FIXME: too naive
   if (*a < *b)
     return 1;
   else if (*a > *b)
     return -1;
   else
     return 0;
 }

static cube_array_t *
bf_sort (cube_array_t *A)
{
  cube_array_t *idx;
  double *data;
  char *args;
  int n, m;
  size_t elmsz;
  int    *idata;

  m = cube_array_get_dim (A, 0);
  n = cube_array_get_dim (A, 1);

  data = cube_array_get_data (A);

  elmsz = sizeof (int) + sizeof (double);
  args = malloc (elmsz * n);

  for (int i = 0; i < n; i++)
    {
      char *base;
      double nrm;

      nrm = cblas_dnrm2 (m, data + (i*m), 1);
      base = args + i*elmsz;
      *((double *) base) = nrm;
      *((int *) (base + sizeof (double))) = i;
    }

  qsort (args, n, elmsz, argsrt_cmp_inv);

  idx = cube_array_new (cube_dtype_int32, 1, n);
  idata = cube_array_get_data (idx);

  for (int i = 0; i < n; i++)
    {
      char *base;

      base = args + i*elmsz + sizeof (double);
      idata[i] = *((int *) base);
    }

  return idx;
}

/* */

static GLuint *
gen_textures (int n)
{
  int i;
  GLuint *textures;

  textures = malloc (sizeof (GLuint) * n);
  glGenTextures (n, textures);

  fprintf (stderr, "Before create_text: %s \n", gluErrorString (glGetError()));
  for (i = 0; i < n; i++)
    {
      float bf[3*7*7] = {0.3, 0.3, 0.3, 0.3, 0.3, };
      glBindTexture (GL_TEXTURE_2D, textures[i]);

      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexImage2D (GL_TEXTURE_2D, 0, 3, 7, 7, 0, GL_BGR, GL_FLOAT, bf);
    }
  fprintf (stderr, "Create textures done: %s \n", gluErrorString (glGetError()));
  return textures;
}


static void
update_textures (cube_array_t *A, int n_textures, GLuint *textures)
{
  float *data;
  int i;
  int m;

  m = cube_array_get_dim (A, 0);
  data = cube_array_get_data (A);

   for (i = 0; i < n_textures; i++)
    {
      float *bf;

      bf = data + (i * m);
      glBindTexture (GL_TEXTURE_2D, textures[i]);
      glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, 7, 7, GL_BGR, GL_FLOAT, bf);
    }
}

static float
grid_update (cube_array_t *grid, int w, int h, float spacing)
{
  int x, y, n, i;
  int cols;
  float r;
  float border;
  float *gd;
  float space;

  if (grid == NULL)
    return 0;

  n = cube_array_get_dim (grid, 0) / 2;

  r = (float) w / (float) h;
  cols = ceil (sqrt (n*r) + 0.5);

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

  return space;
}

static void
draw_bfs (cube_array_t *grid, int n_textures, GLuint *textures, float space, cube_array_t *sort)
{
  int i;
  float *gd;
  int   *idx;

  if (grid == NULL)
    return;

  if (sort)
    idx = cube_array_get_data (sort);
  else
    idx = NULL;

  gd = cube_array_get_data (grid);

  for (i = 0; i < n_textures; i++)
    {
      float x, y;
      int t_idx;

      x = gd[2*i];
      y = gd[2*i + 1];

      t_idx = idx ? idx[i] : i;
      glBindTexture (GL_TEXTURE_2D, textures[t_idx]);

      glBegin(GL_QUADS);
      {
	glTexCoord2f (0.0, 0.0); glVertex2f (x, y);
	glTexCoord2f (0.0, 1.0); glVertex2f (x, y+space);
	glTexCoord2f (1.0, 1.0); glVertex2f (x+space, y+space);
	glTexCoord2f (1.0, 0.0); glVertex2f (x+space, y);
      }
      glEnd();
    }
}

/* Window */
typedef struct BfXpWnd {

  cube_array_t *grid;
  int           n_textures;
  GLuint       *textures;
  float         space;

  /* data */
  cube_array_t *A;
  cube_array_t *chans;
  cube_array_t *sort;
  char         *path;
  cube_h5_t     sca;
  char         *cid;
  char         *mid;
  int           idx;

  /* for log */
  cube_array_t *log_points;
  int           log_idx;

  /*glut*/
  int           id; /* the window */
  int           model_menu;
} BfXpWnd;

static BfXpWnd wnd;


static void
bfxp_wnd_display (BfXpWnd *window)
{
  glClear (GL_COLOR_BUFFER_BIT);
  draw_bfs (window->grid,
	    window->n_textures,
	    window->textures,
	    window->space,
	    window->sort);
}

static void
bfxp_wnd_reshape (BfXpWnd *window, int width, int height)
{
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  glViewport (0, 0, width, height);
  gluOrtho2D (0.0, width, 0.0, height);
  glMatrixMode (GL_MODELVIEW);

  window->space = grid_update (window->grid,
			       width, height,
			       0.01);
}

static void
bfxp_wnd_set_matrix (BfXpWnd      *window,
		     cube_array_t *A)
{
  int n, w, h;
  GLint dims[4];

  n = cube_array_get_dim (A, 1);

  if (window->n_textures != n)
    {

      window->textures = gen_textures (n);
      window->n_textures = n;
    }

  update_textures (A, n, window->textures);

  if (cube_array_get_dim (window->grid, 0) != n*2)
    {
      cube_array_destroy (window->grid);
      window->grid = cube_array_new (cube_dtype_float32, 1, n*2);

      glGetIntegerv (GL_VIEWPORT, dims);
      w = dims[2];
      h = dims[3];

      fprintf (stderr, "%d %d\n", w, h);
      window->space = grid_update (window->grid, w, h, 0.01);
    }

  glutPostRedisplay();
}

static cube_array_t *
prepare_matrix (cube_array_t *A,
		cube_array_t *chans)
{
  int c, m, n, k;
  cube_array_t *nA;
  double *source;
  float  *sink;
  char   *cdata;

  if (chans == NULL)
    {
      fprintf (stderr, "Warning: old SCA, not channels attribute on model\n");
      c = 3;
    }
  else
    {
      m = cube_array_get_dim (chans, 0);
      n = cube_array_get_dim (chans, 1);
      c = max (m, n);
    }

  m = cube_array_get_dim (A, 0);
  n = cube_array_get_dim (A, 1);
  k = (m / c) * 3;

  nA = cube_array_new (cube_dtype_float32, 2, k, n);

  source = cube_array_get_data (A);
  sink = cube_array_get_data (nA);

  cdata = cube_array_get_data (chans);
  bf_to_lms_ (&m, &n, source, &k, sink, &c, cdata);
  bff_normalize_ (&k, &n, sink);

  return nA;
}

static void
bfxp_wnd_reset (BfXpWnd *window)
{
  char buffer[255];
  char *cid, *mid;
  cube_array_t *A, *nA;

  cid = window->cid;
  mid = window->mid;
  A = window->A;

  nA = prepare_matrix (A, window->chans);
  bfxp_wnd_set_matrix (window, nA);

  snprintf (buffer, sizeof (buffer), "%.7s - %.7s", cid, mid);
  glutSetWindowTitle (buffer);
  snprintf (buffer, sizeof (buffer), "BfXp %.7s - %.7s", cid, mid);
  glutSetIconTitle (buffer);
  glutPostRedisplay ();
}

static void
bfxp_wnd_load_model (BfXpWnd *window, const char *mid)
{
  char buffer[255];
  cube_array_t *A, *chans;
  cube_h5_t fd;
  char *cid;

  fd = window->sca;
  cid = window->cid;

  if (window->mid)
    free (window->mid);
  window->mid = strdup (mid);

  //fprintf (stderr, "Laoding model: %s %s\n", cid, mid);

  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/model/%.7s/A", cid, mid);
  A = cube_h5_read_array (fd, buffer);

  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/model/%.7s", cid, mid);
  chans = cube_h5_attr_read_array (fd, buffer, "channels");

  window->sort = bf_sort (A);
  window->A = A;
  window->chans = chans;

  bfxp_wnd_reset (window);
}

static int
cmp_int32 (const void *pa, const void *pb)
{
  int a, b;

  a = * (int *)pa;
  b = * (int *)pb;

  if (a < b)
    return -1;
  else if (a > b)
    return 1;
  else
    return 0;
}

static int
bfxp_wnd_load_log_points (BfXpWnd *window)
{
  cube_array_t *log_points;
  int  *lpdata;
  char buffer[255];
  int  n;
  cube_h5_t fd;
  char *cid, *mid;

  fd = window->sca;
  cid = window->cid;
  mid = window->mid;

  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/log/%.7s", cid, mid);

  n = cube_h5_group_nchildren (fd, buffer);
  printf ("%d log points\n", n);
  if (n == 0)
    return -1;

  log_points = cube_array_new (cube_dtype_int32, 1, n);
  lpdata = cube_array_get_data (log_points);

  for (int i = 0; i < n; i++)
    {
      char *point = cube_h5_loc_get_name_idx (fd, buffer, i);
      lpdata[i] = atoi (point);
      free (point);
    }

  qsort (lpdata, n, sizeof (int), cmp_int32);
  window->log_points = log_points;

  return 0;
}

static void
bfxp_wnd_log_load (BfXpWnd *window, int idx)
{
  char buffer[255];
  char *cid, *mid;
  cube_array_t *A, *nA;
  cube_h5_t fd;
  int log_point;
  int *lpdata;

  fd = window->sca;
  cid = window->cid;
  mid = window->mid;

  lpdata = cube_array_get_data (window->log_points);

  if (lpdata == NULL)
    {
      fprintf (stderr, "FIXME: log_laod\n");
      return;
    }

  log_point = lpdata[idx];

  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/log/%.7s/%d/A", cid, mid, log_point);
  A = cube_h5_read_array (fd, buffer);

  nA = prepare_matrix (A, window->chans);
  bfxp_wnd_set_matrix (window, nA);
  cube_array_destroy (nA);
  cube_array_destroy (A);

  snprintf (buffer, sizeof (buffer), "%.7s - %.7s @ %d", cid, mid, log_point);
  glutSetWindowTitle (buffer);
}

static void
play_log (int value)
{
  BfXpWnd *window = &wnd;
  int n;

  n = cube_array_get_dim (window->log_points, 0);
  if (!(value++ < n))
    {
      bfxp_wnd_reset (window);
      return;
    }

  bfxp_wnd_log_load (window, value);
  glutTimerFunc (10, play_log, value);
}

static void
bfxp_wnd_log_play (BfXpWnd *window)
{
  int res;

  res = bfxp_wnd_load_log_points (window);
  if (res)
    return;

  glutTimerFunc (100, play_log, 0);
}

static char *
model_id_for_index (cube_h5_t fd, const char *cid, int idx)
{
  char buffer[255];
  int n;
  char *id, *mid;

  if (idx < 0)
    return NULL;

  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/model", cid);
  n = cube_h5_group_nchildren (fd, buffer);
  if (n < (idx + 1))
    return NULL;

  id = cube_h5_loc_get_name_idx (fd, buffer, idx);
  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/model/%.7s", cid, id);
  free (id);

  mid = cube_h5_attr_read_string (fd, buffer, "id");
  return mid;
}

static void
bfxp_wnd_model_load_by_idx (BfXpWnd *window, int idx)
{
  char *mid;
  mid = model_id_for_index (window->sca, window->cid, idx);

  if (mid)
    {
      bfxp_wnd_load_model (window, mid);
      free (mid);
      window->idx = idx;
    }
}

static void
bfxp_wnd_model_select_next (BfXpWnd *window)
{
  int idx;
  idx = window->idx + 1;
  bfxp_wnd_model_load_by_idx (window, idx);
}

static void
bfxp_wnd_model_select_prev (BfXpWnd *window)
{
  int idx;
  idx = window->idx - 1;

  bfxp_wnd_model_load_by_idx (window, idx);
}

static void
on_model_menu (int value)
{
  bfxp_wnd_model_load_by_idx (&wnd, value);
}

static void
bfx_wnd_create_model_menu (BfXpWnd *window)
{
  char buffer[255];
  cube_h5_t fd;
  int n, i;

  fd = window->sca;
  snprintf (buffer, sizeof (buffer), "/ICA/%.7s/model", window->cid);
  n = cube_h5_group_nchildren (fd, buffer);

  glutCreateMenu (on_model_menu);

  for (i = 0; i < n; i++)
    {
      char *name = cube_h5_loc_get_name_idx (fd, buffer, i);
      glutAddMenuEntry (name, i);
      free (name);
    }
  glutAttachMenu (GLUT_LEFT_BUTTON);
}

static void
bfxp_wnd_load_file (BfXpWnd *window, const char *filename)
{
  char buffer[255];
  cube_h5_t fd;
  int n;
  char *path, *cid, *mid;

  if (window->sca > 0)
    cube_h5_close (window->sca);

  fprintf (stderr, "load data \n");
  fd = cube_h5_open (filename, 0);
  window->sca = fd;

  n = cube_h5_group_nchildren (fd, "/ICA");
  if (n < 1)
    return;

  path = cube_h5_loc_get_name_idx (fd, "/ICA", 0);
  snprintf (buffer, sizeof (buffer), "/ICA/%.7s", path);
  free (path);
  cid = cube_h5_attr_read_string (fd, buffer, "id");

  if (window->cid)
    free (window->cid);
  window->cid = strdup (cid);

  window->idx = 0;
  mid = model_id_for_index (fd, cid, 0);
  if (mid == NULL)
    return;

  window->path = strdup (filename);
  bfxp_wnd_load_model (window, mid);
  bfx_wnd_create_model_menu (window);
  free (cid);
}

static void
spawn_process (const char *path, char * const argv[])
{
  pid_t pid;
  int res;

  pid = fork ();

  if (pid < 0)
    {
      fprintf (stderr, "Error during fork()");
      return;
    }
  else if (pid == 0)
    {
      pid_t gc_pid = 1;
      /* child*/

      gc_pid = fork ();
      if (gc_pid != 0)
	_exit (gc_pid < 0);

      /* grandchild  */
      for (int i = 3; i < sysconf (_SC_OPEN_MAX); i++)
      	fcntl (i, F_SETFD, FD_CLOEXEC);

      execv (path, argv);
    }

  /* parent */
 godot:
  if (waitpid (pid, &res, 0) < 0)
    {
      if (errno != ECHILD)
	{
	  if (errno == EINTR)
	    goto godot;
	  else
	    fprintf (stderr, "Error during waitpid");
	}
    }
}

static void
bfxp_wnd_spawn_plotter (BfXpWnd *window)
{
  char *argv[5];

  argv[0] = "bfPlotter";
  argv[1] = window->path;
  argv[2] = window->mid;
  argv[3] = window->cid;
  argv[4] = NULL;

  spawn_process ("bfPlotter", argv);
}

static void
display (void)
{
  bfxp_wnd_display (&wnd);
  glutSwapBuffers ();
}

static void
reshape (int width, int height)
{
  bfxp_wnd_reshape (&wnd, width, height);
}

static void
keyboard (unsigned char key, int x, int y)
{
  switch (key)
    {
    case 'q':
      exit (0);
      break;

    case 'n':
      bfxp_wnd_model_select_next (&wnd);
      break;

    case 'p':
      bfxp_wnd_model_select_prev (&wnd);
      break;

    case 'P':
      bfxp_wnd_spawn_plotter (&wnd);
      break;

    case 'l':
      bfxp_wnd_log_play (&wnd);
      break;
    }
}

static void
bfxp_wnd_init (BfXpWnd *window)
{
  int id;

  memset (window, 0, sizeof (BfXpWnd));
  id = glutCreateWindow ("Bf Explorer");
  window->id = id;
  window->sca = -1;

  /* basic gl init  */
  glClearColor (1.0 ,1.0, 1.0, 0.0);
  glColor3f (0.0, 0.0, 0.0);
  glPointSize (4.0);
  glDisable (GL_DEPTH_TEST);
  glEnable (GL_TEXTURE_2D);
  glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei (GL_PACK_ALIGNMENT, 1);
  glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glClear (GL_COLOR_BUFFER_BIT);

  //ext = glGetString (GL_EXTENSIONS);
  //printf ("EXT: %s\n", ext);
  fprintf (stderr, "init done: %s \n", gluErrorString (glGetError()));

  glutDisplayFunc (display);
  glutReshapeFunc (reshape);
  glutKeyboardFunc (keyboard);
  glutIdleFunc (idle);
}
/*  */

static void
load_data (void *data)
{
  bfxp_wnd_load_file (&wnd, data);
}

/* glut callback */

static void
idle ()
{
  struct list_head *iter, *p;

  if (pthread_mutex_trylock (&idle_lock) != 0)
    return;

  list_for_each_safe (iter, p, &idle_queue.list) {
    idle_queue_t *entry;

    entry = list_entry (iter, idle_queue_t, list);

    entry->callback (entry->data);

    list_del (iter);
    free (entry);
  }

  pthread_mutex_unlock (&idle_lock);
  return;
}

int
main (int argc, char** argv)
{
  glutInit (&argc, argv);
  INIT_LIST_HEAD (&idle_queue.list);

  glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize (800, 600);
  bfxp_wnd_init (&wnd);


  if (argc > 1)
    idle_add (load_data, argv[1]);

  glutMainLoop ();
  return EXIT_SUCCESS;
}
