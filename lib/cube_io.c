
#include "cube_io.h"
#include "cube_private.h"


#include <hdf5.h>
#include <hdf5_hl.h>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

static void
flip_dim_hsize (int ndim, hsize_t *dims)
{
  int i, n;

  n = ceil (ndim/2.0);
  ndim -= 1;

  for (i = 0; i < n; i++)
    {
      int t = dims[i];
      dims[i] = dims[ndim-i];
      dims[ndim-i] = t;
    }
}

static cube_dtype_t
dtype_from_sized_klass (H5T_class_t dklass, size_t esize)
{
  cube_dtype_t dtype = 0;

  switch (dklass)
    {
    case H5T_INTEGER:
      dtype = cube_dtype_integer;
      break;

    case H5T_FLOAT:
      dtype = cube_dtype_float;
      break;

    default:
      printf ("Warning: Implement me!\n");
      assert (false);
      break;
    }

  switch (esize)
    {

    case 1:
      dtype |= cube_dtype_size_8;
      break;

    case 2:
      dtype |= cube_dtype_size_16;
      break;

    case 4:
      dtype |= cube_dtype_size_32;
      break;

    case 8:
      dtype |= cube_dtype_size_64;
      break;

    case 16:
      dtype |= cube_dtype_size_128;
      break;
    }

  return dtype;
}

static hid_t
h5type_from_dtype (cube_dtype_t dtype)
{
  hid_t type = 0;

  switch (dtype)
    {
    case cube_dtype_int8:
      type = H5T_NATIVE_CHAR;
      break;

    case cube_dtype_int16:
      type = H5T_NATIVE_SHORT;
      break;

    case cube_dtype_int32:
      type = H5T_NATIVE_INT;
      break;

    case cube_dtype_single:
      type = H5T_NATIVE_FLOAT;
      break;

    case cube_dtype_double:
      type = H5T_NATIVE_DOUBLE;
      break;

    default:
      printf ("Warning: implement me! %d\n", dtype);
      assert (false);
    }

  return type;
}

cube_h5_t
cube_h5_open (const char *filename, int create)
{
  herr_t (*err_func)(hid_t, void*);
  void *data;
  hid_t fd = -1;

  if (create)
    {
      H5Eget_auto (H5E_DEFAULT, &err_func, &data);
      H5Eset_auto (H5E_DEFAULT, NULL, NULL);

      fd = H5Fcreate (filename, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
      H5Eset_auto (H5E_DEFAULT, err_func, data);
    }

  if (fd < 0)
    fd = H5Fopen (filename, H5F_ACC_RDWR, H5P_DEFAULT); //FIXME read only: H5F_ACC_RDONLY

  return (cube_h5_t) fd;
}

void
cube_h5_close (cube_h5_t h5)
{
  hid_t fd;
  fd = (hid_t) h5;
  H5Fclose (fd);
}

cube_array_t *
cube_h5_read_array (cube_h5_t h5, const char *path)
{
  cube_dtype_t  dtype;
  int           ndims;
  herr_t        status;
  hsize_t      *dims;
  cube_array_t *array;
  size_t        esize;
  H5T_class_t   dklass;
  hid_t         type_id;
  hid_t         fd;
  void *        data;

  fd = (hid_t) h5;

  status = H5LTget_dataset_ndims (fd, path, &ndims);

  if (status != 0)
    return NULL;

  dims = malloc (sizeof (hsize_t) * ndims);
  status = H5LTget_dataset_info (fd, path, dims, &dklass, &esize);

  if (status != 0)
    return NULL;

  flip_dim_hsize (ndims, dims);

  dtype = dtype_from_sized_klass (dklass, esize);
  array = cube_array_newa (dtype, ndims, dims);
  data = cube_array_get_data (array);
  type_id = h5type_from_dtype (dtype);
  status = H5LTread_dataset (fd, path, type_id, data);

  free (dims);

  if (status != 0)
    return NULL;

  return array;
}

int
cube_h5_ds_write_array (cube_h5_t h5, const char *path, cube_array_t *array)
{
  hsize_t dims[H5S_MAX_RANK];
  int ndim;
  hid_t dtype;
  hid_t fd;
  void *data;
  int i;

  if (array == NULL)
    return 0;

  fd = (hid_t) h5;

  ndim = cube_array_get_ndim (array);

  if (ndim > (int) sizeof (dims))
    return -1;

  for (i = 0; i < ndim; i++)
    dims[i] = cube_array_get_dim (array, i);

  flip_dim_hsize (ndim, dims);

  dtype = cube_array_get_dtype (array);
  data = cube_array_get_data (array);

  H5LTmake_dataset (fd, path, ndim, dims, dtype, data);
  return 0;
}

cube_matrix_t *
cube_h5_read_matrix (cube_t *ctx, cube_h5_t h5, const char *path)
{
  cube_matrix_t *matrix;
  hsize_t       *h5_dims;
  void          *data;
  cube_dtype_t   dtype;
  H5T_class_t    dklass;
  herr_t         status;
  size_t         esize;
  hid_t          type_id;
  hid_t          fd;
  int            ndims;
  int            m, n;

  if (! cube_context_check (ctx))
    return NULL;

  fd = (hid_t) h5;

  status = H5LTget_dataset_ndims (fd, path, &ndims);

  if (status != 0)
    return NULL;

  if (ndims != 2)
    return NULL;

  h5_dims = malloc (sizeof (hsize_t) * ndims);
  status = H5LTget_dataset_info (fd, path, h5_dims, &dklass, &esize);

  /* h5 is row-major */
  m = h5_dims[1];
  n = h5_dims[0];

  dtype = dtype_from_sized_klass (dklass, esize);
  matrix = cube_matrix_create (ctx, m, n);
  data = cube_matrix_get_data (matrix);
  type_id = h5type_from_dtype (dtype);
  status = H5LTread_dataset (fd, path, type_id, data);

  free (h5_dims);

  if (status != 0)
    return NULL;

  return matrix;

}

int
cube_h5_write_matrix (cube_h5_t h5, cube_matrix_t *matrix, const char *path)
{
  hid_t   fd;
  hsize_t dims[2];
  double *data;

  fd = (hid_t) h5;

  dims[0] = cube_matrix_get_n (matrix);
  dims[1] = cube_matrix_get_m (matrix);
  data = cube_matrix_get_data (matrix);

  H5LTmake_dataset (fd, path, 2, dims, H5T_NATIVE_DOUBLE, data);

  return 0;
}

ica_dataset_t *
cube_h5_read_dataset (cube_h5_t h5, const char *base)
{
  ica_dataset_t *ds;
  char path[255];

  if (base == NULL || base[0] == '\0')
    return NULL;

  ds = malloc (sizeof (ica_dataset_t));

  snprintf (path, sizeof (path), "%s/%s", base, "indicies");
  ds->indicies = cube_h5_read_array (h5, path);

  snprintf (path, sizeof (path), "%s/%s", base, "imgdata");
  ds->imgdata  = cube_h5_read_array (h5, path);

  snprintf (path, sizeof (path), "%s/%s", base, "patsperm");
  ds->patsperm = cube_h5_read_array (h5, path);

  snprintf (path, sizeof (path), "%s/%s", base, "Ainit");
  ds->Ainit = cube_h5_read_array (h5, path);

  ds->blocksize = 100; //FIXME
  ds->patchsize = 7; //FIXME
  ds->nchannel = cube_array_get_dim (ds->imgdata, 0);
  ds->npats = cube_array_get_dim (ds->indicies, 0) *
              cube_array_get_dim (ds->indicies, 3);

  return ds;
}

int
cube_h5_ds_find (cube_h5_t h5, const char *name)
{
  herr_t res;
  hid_t  fd;

  fd = (hid_t) h5;

  res = H5LTfind_dataset (fd, name);

  return res == 1;
}

int
cube_h5_link_delete (cube_h5_t h5, const char *name)
{
  herr_t res;
  hid_t  fd;

  fd = (hid_t) h5;
  
  res = H5Ldelete (fd, name, H5P_DEFAULT);
  
  return res > -1;
}

int
cube_h5_group_nchildren (cube_h5_t fd, const char *group)
{
  hid_t head;
  hsize_t n;

  head = H5Gopen (fd, group, H5P_DEFAULT);

  if (head < 0)
    return 0;

  H5Gget_num_objs (head, &n);

  H5Gclose (head);

  return (int) n;
}

char *
cube_h5_loc_get_name_idx (cube_h5_t fd, const char *location, int index)
{
  char *name;
  char buf[255];
  ssize_t n;

  n = H5Lget_name_by_idx (fd,
			  location,
			  H5_INDEX_NAME,
			  H5_ITER_NATIVE,
			  index,
			  buf,
			  sizeof (buf),
			  H5P_LINK_ACCESS_DEFAULT);

  name = NULL;
  if (n > 0)
    name = strndup (buf, n);

  return name;
}

static hid_t
h5tstr_copy_metadata (hid_t strtype, hid_t type)
{
  H5T_str_t   spad;
  H5T_cset_t  cset;
  htri_t      vstr;
  size_t      slen;
  hid_t       ctype;

  ctype = H5Tcopy (strtype);

  slen = H5Tget_size (type);
  spad = H5Tget_strpad (type);
  cset = H5Tget_cset (type);
  vstr = H5Tis_variable_str (type);

  ctype = H5Tcopy(H5T_C_S1);
  if (vstr)
    H5Tset_size (ctype, H5T_VARIABLE);
  else
    H5Tset_size (ctype, slen);

  H5Tset_cset (ctype, cset);
  H5Tset_strpad (ctype, spad);

  return ctype;
}

static int
h5tstr_equal (hid_t str, hid_t ref)
{
  hid_t ctype;
  ctype = h5tstr_copy_metadata (str, ref);

  if (H5Tequal (ctype, ref))
    return 1;

  H5Tclose (ctype);
  return 0;
}

static char *
cube_h5_alloc_string (hid_t ftype, hid_t *type_out)
{
  hid_t klass;
  size_t slen;
  char  *str = NULL;

  klass = H5Tget_class (ftype);

  if (klass != H5T_STRING)
    {
      fprintf (stderr, "H5IO: Not a string class!\n");
      return NULL;
    }


  slen = H5Tget_size (ftype);

  if (! h5tstr_equal (H5T_C_S1, ftype))
    {
      if (h5tstr_equal (H5T_FORTRAN_S1, ftype))
	slen++;
      else
	{
	  fprintf (stderr, "H5IO: Unkown string type!\n");
	  return NULL;
	}
    }

  str = malloc (sizeof (char) * slen);

  if (type_out)
    {
      hid_t mtype = H5Tcopy (H5T_C_S1);
      H5Tset_size (mtype, slen);
      *type_out = mtype;
    }

  return str;
}

char *
cube_h5_ds_read_string (cube_h5_t h5, const char *location)
{
  hid_t ds;
  hid_t ftype, mtype;
  char  *str = NULL;

  ds = H5Dopen (h5, location, H5P_DEFAULT);

  if (ds < 0)
    {
      fprintf (stderr, "Unkown dataset\n");
      return NULL;
    }

  ftype = H5Tcopy (ds);
  str = cube_h5_alloc_string (ftype, &mtype);

  H5Tclose (ftype);

  if (str == NULL)
    return NULL;

  H5Dread (ds, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, str);
  H5Tclose (mtype);

  return str;
}

static int
h5_space_is_scalar (hid_t space)
{
  int ndims;
  hsize_t dims[2];

  ndims = H5Sget_simple_extent_ndims (space);

  if (ndims == 0)
    return 1;
  else if (ndims != 2)
    return 0;

  H5Sget_simple_extent_dims (space, dims, NULL);
  return dims[0] == 1 && dims[1] == 0;
}

int
cube_h5_attr_read_int (cube_h5_t fd, const char *path, const char *attribute)
{
  hid_t dset;
  hid_t attr;
  hid_t space;
  int value = -1;

  dset = H5Oopen (fd, path, H5P_DEFAULT);
  attr = H5Aopen (dset, attribute, H5P_DEFAULT);
  space = H5Aget_space (attr);

  if (! h5_space_is_scalar (space))
    goto out;

  H5Aread (attr, H5T_NATIVE_INT, &value);

 out:
  H5Aclose (attr);
  H5Oclose (dset);
  H5Sclose (space);

  return value;
}

char *
cube_h5_attr_read_string (cube_h5_t   fd,
			  const char *path,
			  const char *attribute)
{
  hid_t dset;
  hid_t attr;
  hid_t space;
  hid_t ftype;
  hid_t mtype;
  char *str;

  dset = H5Oopen (fd, path, H5P_DEFAULT);
  attr = H5Aopen (dset, attribute, H5P_DEFAULT);
  space = H5Aget_space (attr);

  ftype = H5Aget_type (attr);
  str = cube_h5_alloc_string (ftype, &mtype);
  H5Tclose (ftype);

  if (str == NULL)
    goto out;

  H5Aread (attr, mtype, str); //FIXME check status
  H5Tclose (mtype);

 out:
  H5Oclose (dset);
  H5Aclose (attr);
  return str;
}

cube_array_t *
cube_h5_attr_read_array (cube_h5_t   fd,
			 const char *path,
			 const char *attribute)
{
  cube_array_t *array;
  H5T_class_t dklass;
  hid_t    obj;
  hid_t    attr;
  hid_t    ftype;
  hid_t    mtype;
  hid_t    dtype;
  hid_t    space;
  int      ndims;
  hsize_t *dims;
  size_t   esize;
  herr_t   status;
  void    *data;


  array = NULL;

  obj = H5Oopen (fd, path, H5P_DEFAULT);
  attr = H5Aopen (obj, attribute, H5P_DEFAULT);

  ftype = H5Aget_type (attr);
  esize = H5Tget_size (ftype);
  dklass = H5Tget_class (ftype);

  space = H5Aget_space (attr);
  ndims = H5Sget_simple_extent_ndims (space);
  dims = malloc (sizeof (hsize_t) * ndims);
  status = H5Sget_simple_extent_dims (space, dims, NULL);

  H5Sclose (space);
  if (status < 0)
    goto out;

  flip_dim_hsize (ndims, dims);

  dtype = dtype_from_sized_klass (dklass, esize);
  mtype = h5type_from_dtype (dtype);

  array = cube_array_newa (dtype, ndims, dims);
  data = cube_array_get_data (array);

  //mtype = H5Tarray_create (H5T_NATIVE_INT, ndims, dims);
  status = H5Aread (attr, mtype, data);

  if (status)
    {
      cube_array_destroy (array);
      array = NULL;
    }

 out:
  H5Aclose (attr);
  H5Oclose (obj);
  free (dims);
  return array;
}


int
cube_h5_attr_write_string (cube_h5_t   fd,
			   const char *path,
			   const char *attribute,
			   const char *value)
{
  hid_t dset;
  hid_t space;
  hid_t mtype;
  hid_t attr;

  dset = H5Oopen (fd, path, H5P_DEFAULT);

  if (dset < 0)
    {
      return -1;
    }

  mtype = H5Tcopy (H5T_C_S1);
  H5Tset_size (mtype, strlen (value));

  space = H5Screate (H5S_SCALAR);
  attr = H5Acreate (dset, attribute, mtype, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite (attr, mtype, value);

  H5Tclose (mtype);
  H5Oclose (dset);
  H5Aclose (attr);
  return 0;
}

int
cube_h5_attr_write_double (cube_h5_t   fd,
			   const char *path,
			   const char *attribute,
			   double value)
{
  hid_t dset;
  hid_t space;
  hid_t mtype;
  hid_t attr;

  dset = H5Oopen (fd, path, H5P_DEFAULT);

  if (dset < 0)
    {
      return -1;
    }

  mtype = H5T_NATIVE_DOUBLE;
  space = H5Screate (H5S_SCALAR);
  attr = H5Acreate (dset, attribute, mtype, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite (attr, mtype, &value);

  H5Oclose (dset);
  H5Aclose (attr);
  return 0;
}

int
cube_h5_attr_write_int16 (cube_h5_t   fd,
			  const char *path,
			  const char *attribute,
			  int16_t     value)
{
  hid_t dset;
  hid_t space;
  hid_t mtype;
  hid_t attr;

  dset = H5Oopen (fd, path, H5P_DEFAULT);

  if (dset < 0)
    {
      return -1;
    }

  mtype = H5T_NATIVE_SHORT;
  space = H5Screate (H5S_SCALAR);
  attr = H5Acreate (dset, attribute, mtype, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite (attr, mtype, &value);

  H5Oclose (dset);
  H5Aclose (attr);
  return 0;
}

int
cube_h5_attr_write_array (cube_h5_t h5, const char *path, const char *attribute, cube_array_t *array)
{
  hid_t dset;
  hid_t space;
  hid_t mtype;
  hid_t attr;
  hsize_t dims[H5S_MAX_RANK];
  int ndim;
  hid_t dtype;
  hid_t fd;
  void *data;
  int i;

  if (array == NULL)
    return 0;

  fd = (hid_t) h5;

  dset = H5Oopen (fd, path, H5P_DEFAULT);
  if (dset < 0)
    {
      return -1;
    }

  ndim = cube_array_get_ndim (array);

  if (ndim > (int) sizeof (dims))
    return -1;

  for (i = 0; i < ndim; i++)
    dims[i] = cube_array_get_dim (array, i);

  flip_dim_hsize (ndim, dims);

  dtype = cube_array_get_dtype (array);
  mtype = h5type_from_dtype (dtype);
  data = cube_array_get_data (array);

  space = H5Screate_simple (ndim, dims, NULL);
  attr = H5Acreate (dset, attribute, mtype, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite (attr, mtype, data);

  H5Sclose (space);
  H5Oclose (dset);
  H5Aclose (attr);
  return 0;
}


int
cube_h5_group_create (cube_h5_t fd, const char *path)
{
  hid_t group;
  hid_t lcpl;

  lcpl = H5Pcreate (H5P_LINK_CREATE);
  H5Pset_create_intermediate_group (lcpl, 1);
  group = H5Gcreate (fd, path, lcpl, H5P_DEFAULT, H5P_DEFAULT);
  H5Pclose (lcpl);

  if (group < 0)
    return -1;

  H5Gclose (group);
  return 0;
}

int
cube_h5_flush (cube_h5_t fd, int global)
{
  int res;
  res = H5Fflush (fd, (global ? H5F_SCOPE_GLOBAL : H5F_SCOPE_LOCAL));
  return (int) res;
}
