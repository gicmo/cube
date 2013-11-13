
#include <cube_array.h>

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

size_t
cube_dtype_calc_size (cube_dtype_t dtype, int ndim, int *dims)
{
  size_t elm_size;
  size_t total;
  int    i;

  elm_size = dtype & cube_dtype_size_mask;

  for (total = 0, i = 0; i < ndim; i++)
    total = total * dims[i];

  total = total * elm_size;

  return total;
}

cube_array_t *
cube_array_newa (cube_dtype_t dtype, int ndim, unsigned long long *dims)
{
  cube_array_t *array;
  int           i;
  int           total;
  size_t        size;

  array = malloc (sizeof (cube_array_t));

  if (array == NULL)
    return NULL;

  array->dims = malloc (ndim * sizeof (int));

  if (array->dims == NULL)
    {
      free (array);
      return NULL;
    }

  array->ndim = ndim;
  total = 1;

  for (i = 0; i < ndim; i++)
    {
      array->dims[i] = dims[i];
      total = total * dims[i];
    }

  size = total * (dtype & cube_dtype_size_mask);
  array->data = malloc (size);
  array->dtype = dtype;

  return array;
}

cube_array_t *
cube_array_new (cube_dtype_t dtype, int ndim, ...)
{
  cube_array_t *array;
  va_list       ap;
  int           i;
  int           total;
  size_t        size;

  array = malloc (sizeof (cube_array_t));

  if (array == NULL)
    return NULL;

  array->dims = malloc (ndim * sizeof (int));

  if (array->dims == NULL)
    {
      free (array);
      return NULL;
    }

  array->ndim = ndim;

  total = 1;
  va_start (ap, ndim);

  for (i = 0; i < ndim; i++)
    {
      int n = va_arg (ap, int);
      array->dims[i] = n;
      total = total * n;
    }

  size = total * (dtype & cube_dtype_size_mask);
  array->data = malloc (size);
  array->dtype = dtype;

  return array;
}

void
cube_array_destroy (cube_array_t *array)
{
  if (array == NULL)
    return;

  free (array->dims);
  free (array->data);
  free (array);
}

static int
calc_offset (int ndim, int *dims, int *p)
{
  int offset;
  int i, j;

  offset = 0;

  for (i = 0; i < ndim; i++)
    {
      int x = 1;
      for (j = 0; j < i; j++)
	x *= dims[j];
      offset += x * p[i];
    }

  return offset;
}

int
cube_array_index_va (cube_array_t *array, va_list ap)
{
  int           p[array->ndim];
  int i;

  if (array == NULL)
    return 0;

  for (i = 0; i < array->ndim; i++)
    p[i] = va_arg (ap, int);

  return calc_offset (array->ndim, array->dims, p);
}

int
cube_array_index (cube_array_t *array, ...)
{
  va_list       ap;
  int           p[array->ndim];
  int i;

  if (array == NULL)
    return 0;

  va_start (ap, array);

  for (i = 0; i < array->ndim; i++)
    p[i] = va_arg (ap, int);

  return calc_offset (array->ndim, array->dims, p);
}

cube_dtype_t
cube_array_get_dtype (cube_array_t *array)
{
  if (array == NULL)
    return 0;

  return array->dtype;
}

size_t
cube_array_get_offset (cube_array_t *array, ...)
{
  va_list       ap;
  int           p[array->ndim];
  int i;
  size_t        offset;

  if (array == NULL)
    return 0;

  va_start (ap, array);

  for (i = 0; i < array->ndim; i++)
    p[i] = va_arg (ap, int);

  offset = calc_offset (array->ndim, array->dims, p);
  return offset * (array->dtype & cube_dtype_size_mask);
}

double
cube_array_get_double (cube_array_t *array, ...)
{
  va_list  ap;
  int      offset;

  if (array == NULL)
    return 0;

  va_start (ap, array);

  offset = cube_array_index_va (array, ap);
  
  return *((double *) array->data + offset);
}

int
cube_array_get_int (cube_array_t *array, ...)
{
  va_list  ap;
  int      offset;

  if (array == NULL)
    return 0;

  va_start (ap, array);

  offset = cube_array_index_va (array, ap);
  
  return *((int *) array->data + offset);
}

uint16_t
cube_array_get_uint16 (cube_array_t *array, ...)
{
  va_list  ap;
  int      offset;
  
  if (array == NULL)
    return 0;

  va_start (ap, array);

  offset = cube_array_index_va (array, ap);
  
  return *((uint16_t *) array->data + offset);
}

void *
cube_array_get_data (cube_array_t *array)
{
  if (array == NULL)
    return NULL;

  return array->data;
}

int
cube_array_get_ndim (cube_array_t *array)
{
  if (array == NULL)
    return 0;

  return array->ndim;
}

int
cube_array_get_dim (cube_array_t *array, int d)
{
  if (array == NULL)
    return 0;

  return array->dims[d];
}
