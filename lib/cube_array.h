#ifndef CUBE_ARRAY_H
#define CUBE_ARRAY_H

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cube_dtype_t {

  cube_dtype_size_8   = 1,
  cube_dtype_size_16  = 2,
  cube_dtype_size_32  = 4,
  cube_dtype_size_64  = 8,
  cube_dtype_size_128 = 16,

  cube_dtype_size_mask = 0xFF,

  cube_dtype_integer = 0x100,
  cube_dtype_float   = 0x200,

  cube_dtype_int8    = cube_dtype_integer | cube_dtype_size_8,
  cube_dtype_int16   = cube_dtype_integer | cube_dtype_size_16,
  cube_dtype_int32   = cube_dtype_integer | cube_dtype_size_32,
  cube_dtype_int64   = cube_dtype_integer | cube_dtype_size_64,
  cube_dtype_float32 = cube_dtype_float   | cube_dtype_size_32,
  cube_dtype_float64 = cube_dtype_float   | cube_dtype_size_64,
  
  cube_dtype_single = cube_dtype_float32,
  cube_dtype_double = cube_dtype_float64
} cube_dtype_t;
 
typedef struct _cube_array_t cube_array_t;

struct _cube_array_t {

  cube_dtype_t dtype;
  int     ndim;
  int    *dims;

  void   *data;
};

cube_array_t * cube_array_newa (cube_dtype_t dtype, int ndim, unsigned long long *dims);
cube_array_t * cube_array_new (cube_dtype_t dtype, int ndim, ...);
void           cube_array_destroy (cube_array_t *array);
int            cube_array_index (cube_array_t *array, ...);
cube_dtype_t   cube_array_get_dtype (cube_array_t *array);
size_t         cube_array_get_offset (cube_array_t *array, ...);
void *         cube_array_get_data (cube_array_t *array);
int            cube_array_get_ndim (cube_array_t *array);
int            cube_array_get_dim (cube_array_t *array, int d);
double         cube_array_get_double (cube_array_t *array, ...);
int            cube_array_get_int (cube_array_t *array, ...);
uint16_t       cube_array_get_uint16 (cube_array_t *array, ...);

#ifdef __cplusplus
}
#endif
#endif /* CUBE_ARRAY_H */
