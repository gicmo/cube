#include <json/json.h>
#include <cube_math.h>

#include "sca.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROOT "/ICA"

struct _sca_config_t {
  char *id;
  json_object *root;
};

/*  */

static int
json_property_get_double (json_object *obj, const char *key, double *value)
{
  json_object *val_obj;

  val_obj = json_object_object_get (obj, key);

  if (val_obj == NULL)
    return -1;

  *value = json_object_get_double (val_obj);

  return 0;
}

static int
json_object_read_double_array (json_object *array_obj,
			       double      *array,
			       int          n)
{
  int i, na;

  na = json_object_array_length (array_obj);

  for (i = 0; i < min (na, n); i++)
    {
      json_object *val;

      val = json_object_array_get_idx (array_obj, i);
      if (val == NULL)
	break;

      array[i] = (double) json_object_get_double (val);
    }

  return i;
}


/*  */

int
sca_config_count (cube_h5_t sca)
{
  return cube_h5_group_nchildren (sca, "/ICA");
}

char *
sca_config_name (cube_h5_t sca, int index)
{
  char *str;
  str = cube_h5_loc_get_name_idx (sca, ROOT, index);
  return str;
}

sca_config_t *
sca_config_read (cube_h5_t sca, const char *id)
{
  char buf[4069];
  char *str;
  sca_config_t *cfg;
  json_object *root;
  int n;

  n = snprintf (buf, sizeof (buf), "%s/%s/config", ROOT, id);

  if (n < 0)
    return NULL;

  str = cube_h5_ds_read_string (sca, buf);

  if (str == NULL)
    return NULL;

  root = json_tokener_parse (str);

  if (root == NULL)
    {
      free (str);
      return NULL;
    }

  cfg = malloc (sizeof (sca_config_t));

  snprintf (buf, sizeof (buf), "%s/%s", ROOT, id);
  cfg->id = cube_h5_attr_read_string (sca, buf, "id");
  cfg->root = root;

  free (str);

  return cfg;
}

void
sca_config_free (sca_config_t *cfg)
{
  free (cfg->id);
  json_object_put (cfg->root);
  free (cfg);
}

const char *
sca_config_get_id (sca_config_t *cfg)
{
  return cfg->id;
}

ica_prior_t *
sca_config_create_prior (cube_t *ctx, sca_config_t *cfg, int dimension)
{
  ica_prior_t *prior;
  json_object *cfg_pr;
  double mu, beta, sigma;
  double a, b;
  double tol;
  double adapt_size;
  int res;

  cfg_pr = json_object_object_get (cfg->root, "prior");

  if (cfg_pr == NULL)
    return NULL;

  res  = json_property_get_double (cfg_pr, "mu", &mu);
  res += json_property_get_double (cfg_pr, "beta", &beta);
  res += json_property_get_double (cfg_pr, "sigma", &sigma);
  res += json_property_get_double (cfg_pr, "a", &a);
  res += json_property_get_double (cfg_pr, "b", &b);
  res += json_property_get_double (cfg_pr, "tol", &tol);
  res += json_property_get_double (cfg_pr, "adapt_size", &adapt_size);

  if (res > 0)
    {
      return NULL;
    }

  prior = malloc (sizeof (ica_prior_t));

  prior->mu = cube_matrix_new_fill (ctx, dimension, 1, mu);
  prior->beta = cube_matrix_new_fill (ctx, dimension, 1, beta);
  prior->sigma = cube_matrix_new_fill (ctx, dimension, 1, sigma);

  prior->a = a;
  prior->b = b;
  prior->tol = tol;
  prior->adapt_size = adapt_size;

  return prior;
}

cube_array_t *
sca_config_create_gradient (cube_t *ctx, sca_config_t *cfg)
{
  cube_array_t *gradient;
  json_object *cfg_gradient;
  json_object *array_obj;
  double *data;
  double scale;
  int m, j;

  cfg_gradient = json_object_object_get (cfg->root, "gradient");

  if (cfg_gradient == NULL)
    return NULL;

  array_obj = json_object_object_get (cfg_gradient, "iter_points");

  if (array_obj == NULL)
    {
      return NULL;
    }

  m = json_object_array_length (array_obj);

  gradient = cube_array_new (cube_dtype_float | cube_dtype_size_64, 2, m, 2);
  data = cube_array_get_data (gradient);

  json_object_read_double_array (array_obj, data, m);
  //json_object_put (array_obj);


  array_obj = json_object_object_get (cfg_gradient, "epsilon");

  if (array_obj == NULL)
    {
      cube_array_destroy (gradient);
      gradient = NULL;
      goto out;
    }

  j = json_object_array_length (array_obj);

  if (j != m)
    {
      //json_object_put (array_obj);
      cube_array_destroy (gradient);
      gradient = NULL;
      goto out;
    }

  json_object_read_double_array (array_obj, data + m, m);

  json_property_get_double (cfg_gradient, "eps_scale", &scale);
  cube_host_dscal (ctx, m, scale, data + m, 1);

 out:
  return gradient;
}


int
sca_dataset_count (cube_h5_t sca, const char *config)
{
  char buf[4069];
  int  n;

  n = snprintf (buf, sizeof (buf), "%s/%s/dataset", ROOT, config);

  if (n < 0)
    return 0;

  return cube_h5_group_nchildren (sca, buf);
}

char *
sca_dataset_name (cube_h5_t sca, const char *config, int index)
{
  char  buf[4069];
  char *str;
  int   n;

  n = snprintf (buf, sizeof (buf), "%s/%s/dataset", ROOT, config);

  if (n < 0)
    return 0;

  str = cube_h5_loc_get_name_idx (sca, buf, index);

  return str;
}

ica_dataset_t *
sca_dataset_read (cube_h5_t h5, const char *config, const char *dataset)
{
  ica_dataset_t *ds;
  char path[255];
  char base[255];

  snprintf (base, sizeof (base), "%s/%.7s/dataset/%.7s",
	    ROOT, config, dataset);

  ds = malloc (sizeof (ica_dataset_t));

  ds->id = cube_h5_attr_read_string (h5, base, "id");

  snprintf (path, sizeof (path), "%s/%s", base, "indicies");
  ds->indicies = cube_h5_read_array (h5, path);

  snprintf (path, sizeof (path), "%s/%s", base, "imgdata");
  ds->imgdata  = cube_h5_read_array (h5, path);

  snprintf (path, sizeof (path), "%s/%s", base, "patsperm");
  ds->patsperm = cube_h5_read_array (h5, path);

  snprintf (path, sizeof (path), "%s/%s", base, "A_guess");
  ds->Ainit = cube_h5_read_array (h5, path);

  ds->dim = cube_h5_attr_read_int (h5, base, "dim");
  ds->blocksize = cube_h5_attr_read_int (h5, base, "blocksize");
  ds->patchsize = cube_h5_attr_read_int (h5, base, "patchsize");
  ds->nchannel = cube_array_get_dim (ds->imgdata, 0);
  ds->maxiter = cube_h5_attr_read_int (h5, base, "maxiter");
  ds->npats = cube_array_get_dim (ds->indicies, 0) *
              cube_array_get_dim (ds->indicies, 3);

  ds->channels = cube_h5_attr_read_array (h5, base, "channels");

  return ds;
}

int
sca_write_model (cube_h5_t h5, ica_t *ica)
{
  char *id;
  char base[255];
  char path[255];

  id = ica->id;

  snprintf (base, sizeof (base), "%s/%.7s/model/%.7s",
	    ROOT, ica->cfgid, id);

  cube_h5_group_create (h5, base);
  cube_h5_attr_write_string (h5, base, "id", id);
  cube_h5_attr_write_string (h5, base, "cfg", ica->cfgid);
  cube_h5_attr_write_string (h5, base, "ds", ica->dataset->id);
  cube_h5_attr_write_double (h5, base, "fit_time", ica->fit_time);
  cube_h5_attr_write_string (h5, base, "creator", ica->creator);
  cube_h5_attr_write_string (h5, base, "ctime", ica->ctime);
  cube_h5_attr_write_int16 (h5, base, "gpu", ica->gpu);
  cube_h5_attr_write_array (h5, base, "channels", ica->dataset->channels);

  snprintf (path, sizeof (path), "%s/%s", base, "A");
  cube_h5_write_matrix (h5, ica->A, path);

  snprintf (path, sizeof (path), "%s/%s", base, "beta");
  cube_h5_write_matrix (h5, ica->prior->beta, path);

  return 0;
}

int
sca_write_log (cube_h5_t h5, ica_t *ica, int i)
{
  char base[255];
  char path[255];

  snprintf (base, sizeof (base), "%s/%.7s/log/%.7s/%d",
	    ROOT, ica->cfgid, ica->id, i);

  cube_h5_group_create (h5, base);

  cube_h5_attr_write_array (h5, base, "channels", ica->dataset->channels);
  cube_h5_attr_write_string (h5, base, "ds", ica->dataset->id);

  snprintf (path, sizeof (path), "%s/A", base);
  cube_h5_write_matrix (h5, ica->A, path);

  snprintf (path, sizeof (path), "%s/beta", base);
  cube_h5_write_matrix (h5, ica->prior->beta, path);

  cube_h5_flush (h5, 0);

  return 0;

}

int
sca_model_read (cube_h5_t h5, ica_model_t *model, const char *cfgid, const char *id)
{
  char base[255];
  char path[255];

  snprintf (base, sizeof (base), "%s/%.7s/model/%.7s",
	    ROOT, cfgid, id);

  snprintf (path, sizeof (path), "%s/A", base);
  model->A = cube_h5_read_array (h5, path);

  return 0;
}
