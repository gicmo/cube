// This file is part of the cube - ica/cuda - software package
// Copyright © 2010-2013 Christian Kellner <kellner@bio.lmu.de>
// License: MIT (see LICENSE.BSD-MIT)

#include <cube_math.h>
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

#include <math.h>

#include <stdlib.h>

/* cube include  */
#include <cube.h>
#include <cube_array.h>
#include <cube_io.h>
#include <cube_math.h>

#include <bfxpwnd.h>
#include <cube_plot_bfspcr.h>

struct _BfXpWindowPrivate
{
  CubePlotBfSpCr *bf_plot;

};

enum
{
  PROP_0,
};

G_DEFINE_TYPE (BfXpWindow, bfxp_window, GTK_TYPE_WINDOW);


/* columns */
enum
{
  FILENAME_COL = 0,
  N_COLUMNS
};

static void
bfxp_window_init (BfXpWindow* self)
{
  BfXpWindowPrivate *priv;
  GtkWidget *vbox;
  GtkWidget *plot;
  GtkWidget *button;
  GtkWidget *paned;
  GtkWidget *tree;


  self->priv = priv = G_TYPE_INSTANCE_GET_PRIVATE (self,
						   BFXP_TYPE_WINDOW,
						   BfXpWindowPrivate);

  gtk_window_set_title (GTK_WINDOW (self), "Basis Function Explorer");

  paned = gtk_hpaned_new ();
  gtk_container_add (GTK_CONTAINER (self), paned);
  gtk_widget_show (paned);

  /* file list */
  tree = gtk_tree_view_new ();
  gtk_widget_show (tree);
  gtk_paned_add1 (GTK_PANED (paned), tree);

  /* bfxp plot */
  vbox = gtk_vbox_new (FALSE, 0);
  gtk_paned_add2 (GTK_PANED (paned), vbox);
  gtk_widget_show (vbox);

  plot = cube_plot_bfspcr_new ();
  gtk_box_pack_start (GTK_BOX (vbox), plot, TRUE, TRUE, 0);
  gtk_widget_show (plot);

  priv->bf_plot = CUBE_PLOT_BF_SP_CR (plot);

  button = gtk_button_new_with_label ("Quit");

  g_signal_connect (G_OBJECT (button), "clicked",
                    G_CALLBACK (gtk_main_quit), NULL);

  gtk_box_pack_start (GTK_BOX (vbox), button, FALSE, FALSE, 0);
  //gtk_paned_add1 (GTK_PANED (paned), button);

  gtk_widget_show (button);
}


static void
destroy (GtkObject* object)
{
  BfXpWindow *wnd;
  BfXpWindowPrivate *priv;

  wnd = BFXP_WINDOW (object);
  priv = wnd->priv;

  GTK_OBJECT_CLASS (bfxp_window_parent_class)->destroy (object);
}

static void
bfxp_window_class_init (BfXpWindowClass* self_class)
{
  GObjectClass* object_class = G_OBJECT_CLASS (self_class);
  GtkObjectClass* gtk_object_class = GTK_OBJECT_CLASS (self_class);

  gtk_object_class->destroy = destroy;
  g_type_class_add_private (self_class, sizeof (BfXpWindowPrivate));
}

GtkWidget*
bfxp_window_new (void)
{
  return g_object_new (BFXP_TYPE_WINDOW,
                       NULL);
}

 static int
argsrt_cmp (const void *p1, const void *p2)
 {
   const float *a, *b;
   a = p1;
   b = p2;

   //FIXME: too naive
   if (*a < *b)
     return -1;
   else if (*a > *b)
     return 1;
   else
     return 0;

 }

static cube_array_t *
bf_sort (cube_array_t *A)
{
  cube_array_t *idx;
  char *data;
  char *args;
  int n, m;
  size_t elmsz;
  int    *idata;

  m = cube_array_get_dim (A, 0);
  n = cube_array_get_dim (A, 1);

  data = cube_array_get_data (A);

  elmsz = sizeof (int) + sizeof (float);
  args = malloc (elmsz * n);

  for (int i = 0; i < n; i++)
    {
      char *base;

      base = args + i*elmsz;
      *((float *) base) = cblas_snrm2 (m, (float *) data + (i*m), 1);
      *((int *) (base + sizeof (float))) = i;
    }

  qsort (args, n, elmsz, argsrt_cmp);

  idx = cube_array_new (cube_dtype_int32, 1, n);
  idata = cube_array_get_data (idx);

  for (int i = 0; i < n; i++)
    {
      char *base;

      base = args + i*elmsz + sizeof (float);
      idata[i] = *((int *) base);
    }

  return idx;
}

extern void bf_to_lms_ (int *m, int *n, double *A, int *k, float *B, int *nch, char *C);
extern void bff_normalize_ (int *m, int *n, float *A);

void
bfxp_window_load_data (BfXpWindow *self)
{
  BfXpWindowPrivate *priv;
  CubePlotBfSpCr *plot;
  cube_h5_t fd;
  char path[255];
  cube_array_t *A, *nA, *idx;
  int m, n, k, c;
  double *source;
  float  *sink;
  int *idata;

  priv = self->priv;
  plot = priv->bf_plot;

  fprintf (stderr, "load adata \n");
  fd = cube_h5_open ("../results/cbde32b.sca.h5", 0);

  snprintf (path, sizeof (path), "/ICA/cbde32b/model/22d24f0/A");

  A = cube_h5_read_array (fd, path);

  m = k = cube_array_get_dim (A, 0);
  n = cube_array_get_dim (A, 1);
  c = 3;

  nA = cube_array_new (cube_dtype_float32, 2, k, n);

  source = cube_array_get_data (A);
  sink = cube_array_get_data (nA);

  bf_to_lms_ (&m, &n, source, &k, sink, &c, NULL);
  bff_normalize_ (&k, &n, sink);

  idx = bf_sort (nA);
  idata = cube_array_get_data (idx);
  for (int i = 0; i < n; i++)
    {
      g_print ("%d ", idata[i]);
    }
  g_print ("\n");

  cube_plot_bfspcr_set_A (plot, nA);
//cube_plot_bfspcr_set_sortidx (plot, idx);

  cube_h5_close (fd);
}

/* vim:set et sw=2 cino=t0,f0,(0,{s,>2s,n-1s,^-1s,e2s: */
